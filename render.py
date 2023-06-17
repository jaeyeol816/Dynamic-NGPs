import json
import os
from os import path
import subprocess
import copy
import datetime

f_config = open('./render_config.json', 'r')
config_data = json.load(f_config)
f_config.close()

train_id = config_data["train_exp_id"]
render_id = config_data["render_exp_id"]

result_dir = config_data["result_dir"]
frame_start = config_data["frame_start"]
frame_end = config_data["frame_end"]
ingp_home_dir = '.'


# 기본적으로 디렉토리 만들고 있을거 있는지 체크
if not os.path.exists(f'{result_dir}'):
	pass
if not os.path.exists(f'{result_dir}/train_{train_id}'):
  pass
if not os.path.exists(f'{result_dir}/train_{train_id}/log.txt'):
	pass
if not os.path.exists(f'{result_dir}/train_{train_id}/transforms.json'):
	pass
os.system(f'mkdir {result_dir}/train_{train_id}/render_{render_id}')
os.system(f'mkdir {result_dir}/train_{train_id}/render_{render_id}/temp')
train_log_file = f'{result_dir}/train_{train_id}/log.txt'
render_log_file = f'{result_dir}/train_{train_id}/render_{render_id}/log.txt'

# 시작 시간을 로그 파일에 기록
start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
with open(render_log_file, "w") as log_file:
	log_file.write(f"[{start_time}] - Start!!\n")

# 기본 세팅을 로그 파일에 기록
with open(render_log_file, "a") as log_file:
	log_file.write('\n')
	for key, value in config_data.items():
		log_file.write(f"{key}: {value}\n")

# Step1. view start index와 num of views 등등 파악
view_start_idx = None
num_of_views = None
miv_json_path = None
with open(train_log_file, 'r') as log_file:
	lines = log_file.readlines()
	for line in lines:
		if "view_start_idx" in line:
			view_start_idx = int(line.split(":")[1].strip())
		elif "num_of_views" in line:
			num_of_views = int(line.split(":")[1].strip())
		elif "path_of_MIV_json_file" in line:
			miv_json_path = line.split(":")[1].strip()
		elif "initial_n_iters" in line:
			initial_n_iters = int(line.split(":")[1].strip())
		elif "transfer_learning_n_iters"	in line:
			transfer_n_iters = int(line.split(":")[1].strip())
		elif "input_frame_rate" in line:
			input_frame_rate = int(line.split(":")[1].strip())
		elif "images_base_dir" in line:
			image_base_dir = line.split(":")[1].strip()
if view_start_idx == None or num_of_views == None or miv_json_path == None:
	# 향후 에러처리
	pass

# pose trace render인지 test view render인지 플래그
pose_render_flag = False
test_render_flag = False
if config_data["render_pose_trace"] == "true" or config_data["render_pose_trace"] == "True":
	pose_render_flag = True
if config_data["render_test_set"] == "true" or config_data["render_test_set"] == "True":
	test_render_flag = True

# 디렉토리 만들기
for F in range(frame_start, frame_end+1):
	os.system(f'mkdir {result_dir}/train_{train_id}/render_{render_id}/temp/frame{F}')


# Step2. 각 프레임별 렌더링용 카메라 파라미터 파일 생성 (transforms_train.json 또는 transforms_pose.json)
if pose_render_flag:
	pose_csv_path = config_data["path_of_MIV_pose_trace_file"]
	subprocess.call(f'python ./camorph/main.py 1 {miv_json_path} \
		  {result_dir}/train_{train_id}/render_{render_id}/transforms_poses.json \
		  {num_of_views}  {pose_csv_path}', shell=True)
	for F in range(frame_start, frame_end + 1):
		os.system(f'cp {result_dir}/train_{train_id}/render_{render_id}/transforms_poses.json \
	    {result_dir}/train_{train_id}/render_{render_id}/temp/frame{F}')
	for F in range(frame_start, frame_end + 1):
		f_ex_read = open(f"{result_dir}/train_{train_id}/render_{render_id}/transforms_poses.json", 'r')
		data = json.load(f_ex_read)
		f_ex_read.close()
		data["frames"] = data["frames"][F-1:F]
		f_ex_write = open(f"{result_dir}/train_{train_id}/render_{render_id}/temp/frame{F}/transforms_poses.json", "w+")
		json.dump(data, f_ex_write, ensure_ascii=False, indent='\t')
		f_ex_write.close()
if test_render_flag:
	test_render_views = config_data["test_set_view"]
	for F in range(frame_start, frame_end + 1):
		with open(f"{result_dir}/train_{train_id}/transforms.json", "r") as f_orig:
			orig = json.load(f_orig)
			test = copy.deepcopy(orig)
			orig_frames = orig["frames"]
			for j, data_frame in enumerate(orig_frames):
				test["frames"][j]["file_path"] = \
					f"{image_base_dir}/images/v{j+view_start_idx}/image-v{j+view_start_idx}-f{str(F).zfill(3)}.png"
			for i in reversed(range(len(orig_frames))):
				if i not in test_render_views:
					del(test["frames"][i])
			f_test = open(f"{result_dir}/train_{train_id}/render_{render_id}/temp/frame{F}/transforms_test.json", "w")
			json.dump(test, f_test, ensure_ascii=False, indent='\t')
			f_test.close()
# 로그 파일에 기록
start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
with open(render_log_file, "a") as log_file:
	log_file.write(f"\n[{start_time}] - Success on translating camera parameter for rendering views\n")


# Step3. Instant-NGP 실행 (저장된 모델 불러와 렌더링)
if pose_render_flag:
	for F in range(frame_start, frame_end + 1):
		print(f"Rendering Frame{F} ...")
		#로그 파일에 시작 시간 기록
		time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
		with open(render_log_file, "a")	as log_file:
			log_file.write(f"[{time}] - Start rendering frame{F}. (from pose trace {pose_csv_path}) \n\n")
		if F == frame_start:
			accumulated_iter = initial_n_iters
		else:
			accumulated_iter += transfer_n_iters
		os.system(f'mkdir {result_dir}/train_{train_id}/render_{render_id}/temp/frame{F}/poses_outputs')
		os.system(f'python {ingp_home_dir}/scripts/run.py \
	    --network {ingp_home_dir}/configs/nerf/dyngp_render.json \
			--scene {result_dir}/train_{train_id}/frames/frame{F}/transforms_train.json \
			--n_steps 0 \
			--load_snapshot {result_dir}/train_{train_id}/models/frame{F}/frame{F}.msgpack \
			--screenshot_transforms {result_dir}/train_{train_id}/render_{render_id}/temp/frame{F}/transforms_poses.json \
			--screenshot_dir {result_dir}/train_{train_id}/render_{render_id}/temp/frame{F}/poses_outputs \
			> {result_dir}/train_{train_id}/render_{render_id}/temp/frame{F}/frame{F}_log.txt ')
		# 로그 파일에 종료 시각 기록
		time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
		with open(render_log_file, "a")	as log_file:
			log_file.write(f"[{time}] - Finish rendering frame{F}. (png saved at {result_dir}/train_{train_id}/render_{render_id}/temp/frame{F}/poses_outputs) \n\n")
if test_render_flag:
	for F in range(frame_start, frame_end + 1):
		print(f"Rendering Frame{F} ...")
		#로그 파일에 시작 시간 기록
		time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
		with open(render_log_file, "a")	as log_file:
			log_file.write(f"[{time}] - Start rendering frame{F}. (for test view {test_render_views}) \n\n")
		if F == frame_start:
			accumulated_iter = initial_n_iters
		else:
			accumulated_iter += transfer_n_iters
		os.system(f'mkdir {result_dir}/train_{train_id}/render_{render_id}/temp/frame{F}/testview_outputs')
		os.system(f'python {ingp_home_dir}/scripts/run.py \
	    --network {ingp_home_dir}/configs/nerf/dyngp_render.json \
			--scene {result_dir}/train_{train_id}/frames/frame{F}/transforms_train.json \
			--n_steps 0 \
			--load_snapshot {result_dir}/train_{train_id}/models/frame{F}/frame{F}.msgpack \
			--screenshot_transforms {result_dir}/train_{train_id}/render_{render_id}/temp/frame{F}/transforms_test.json \
			--screenshot_dir {result_dir}/train_{train_id}/render_{render_id}/temp/frame{F}/testview_outputs \
			> {result_dir}/train_{train_id}/render_{render_id}/temp/frame{F}/frame{F}_log.txt ')
		# 로그 파일에 종료 시각 기록
		time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
		with open(render_log_file, "a")	as log_file:
			log_file.write(f"[{time}] - Finish rendering frame{F}. (png saved at {result_dir}/train_{train_id}/render_{render_id}/temp/frame{F}/testview_outputs) \n\n")

# Step4. 동영상으로 취합
if pose_render_flag:
	os.system(f"mkdir {result_dir}/train_{train_id}/render_{render_id}/temptemp")
	for F in range(frame_start, frame_end+1):
		os.system(f"cp {result_dir}/train_{train_id}/render_{render_id}/temp/frame{F}/poses_outputs/*.png \
	    {result_dir}/train_{train_id}/render_{render_id}/temptemp")
	os.system(f"ffmpeg -framerate {input_frame_rate} -pattern_type glob \
		-i '{result_dir}/train_{train_id}/render_{render_id}/temptemp/*.png' -c:v libx264 -pix_fmt yuv420p \
		{result_dir}/train_{train_id}/render_{render_id}/poses_video.mp4")
	os.system(f"rm -rf {result_dir}/train_{train_id}/render_{render_id}/temptemp")
if test_render_flag:
	for V in test_render_views:
		os.system(f"mkdir {result_dir}/train_{train_id}/render_{render_id}/temp_v{V}")
	for i, V in enumerate(test_render_views):
		for F in range(frame_start, frame_end+1):
			os.system(f"cp {result_dir}/train_{train_id}/render_{render_id}/temp/frame{F}/testview_outputs/image-v{V}-f{str(F).zfill(3)}.png \
	    {result_dir}/train_{train_id}/render_{render_id}/temp_v{V}")
		os.system(f"ffmpeg -framerate {input_frame_rate} -pattern_type glob \
			-i '{result_dir}/train_{train_id}/render_{render_id}/temp_v{V}/*.png' -c:v libx264 -pix_fmt yuv420p \
			{result_dir}/train_{train_id}/render_{render_id}/view{V}_video.mp4")
	for V in test_render_views:
		os.system(f"rm -rf {result_dir}/train_{train_id}/render_{render_id}/temp_v{V}")

# 로그 파일에 전체 종료 시각 기록
time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
with open(render_log_file, "a") as log_file:
	log_file.write(f"[{time}] - Success on generating 6DoF Video(s)\n\n")
