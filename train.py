import json
import os
from os import path
import subprocess
import copy

f_config = open('./train_config.json', 'r')
config_data = json.load(f_config)
f_config.close()

train_id = config_data["train_exp_id"]

image_base_dir = config_data["images_base_dir"]
result_dir = config_data["result_dir"]
miv_json_path = config_data["path_of_MIV_json_file"]
input_frame_rate = config_data["input_frame_rate"]
initial_n_iters = config_data["initial_n_iters"]
transfer_n_iters = config_data["transfer_learning_n_iters"]
frame_start = config_data["frame_start"]
frame_end = config_data["frame_end"]
ingp_home_dir = '.'

# num_of_original_frames = 0		 # will be modified below. don't touch.
view_start_idx = 0		 # will be modified below. don't touch.
num_of_views = 0			 # will be modified below. don't touch.

# 에러 처리
if frame_start <= 0:
	print('frame_start should be 1 or bigger')
	exit()
if frame_start >= frame_end:
	print('frame_start should be smaller than frame_end')
	exit()


# Step1. 필요시 yuv를 png로 변환해 images에 저장
if path.exists(f'{image_base_dir}/images'):
	views_list = os.listdir(f'{image_base_dir}/images')	
	views_list.sort()
	if '_v0' in views_list[0]:
		view_start_idx = 0
	elif '_v1' in views_list[0]:
		view_start_idx = 1
	else:
		# 향후 에러 처리 코드 작성
		pass
	num_of_views = len(views_list)
else:
	yuv_dir = config_data["path_of_dir_containing_only_texture_yuv"] 
	input_video_size = config_data["input_video_width_height"]
	input_frame_rate = config_data["input_frame_rate"]
  
	video_file_list = os.listdir(yuv_dir)
	video_file_list = [x for x in video_file_list if x[-3:]=='yuv' and 'depth' not in x]
	num_of_views = len(video_file_list)
  
	start_with_zero_file=  [x for x in video_file_list if 'v0_' in x or 'v00_' in x]
	if len(start_with_zero_file) > 0:
		view_start_idx = 0
	else:
		view_start_idx = 1
	
	os.system(f'sudo mkdir {image_base_dir}')
	os.system(f'sudo mkdir {image_base_dir}/images')

	for V in range(view_start_idx, num_of_views + view_start_idx):
		video_file = [x for x in video_file_list if f'v{V}_' in x or f'v0{V}_' in x]
		video_file = video_file[0]
		os.system(f'sudo mkdir {image_base_dir}/images/v{V}; \
			sudo ffmpeg -pixel_format yuv420p10le \
		-video_size {input_video_size} -framerate {input_frame_rate} \
		-i {yuv_dir}/{video_file} \
		-f image2 -pix_fmt rgba \
		{image_base_dir}/images/v{V}/image-v{V}-f%3d.png')

# Step2. frames 디렉토리 생성후 카메라 파라미터 파일 변환하기
# num_of_original_frames = len(os.listdir(f'{image_base_dir}/images/v{view_start_idx}'))

os.system(f'mkdir {result_dir}')
os.system(f'mkdir {result_dir}/train_{train_id}')
os.system(f'mkdir {result_dir}/train_{train_id}/frames')
for F in range(frame_start, frame_end + 1):
	os.system(f'mkdir {result_dir}/train_{train_id}/frames/frame{F}')

subprocess.call(f'python ./camorph/main.py 0 {miv_json_path} {result_dir}/train_{train_id}/transforms.json \
		 {num_of_views} 0', shell=True)

f_tr_json = open(f'{result_dir}/train_{train_id}/transforms.json', 'r')
orig_tr_json = json.load(f_tr_json)
tr_json = orig_tr_json
f_tr_json.close()
tr_json["aabb_scale"] = 64
# tr_json["offset"] = [-1, -2, 0]
translation_val_sample = \
	( abs(tr_json["frames"][0]["transform_matrix"][0][3]) \
		+ abs(tr_json["frames"][0]["transform_matrix"][1][3]) \
		+ abs(tr_json["frames"][1]["transform_matrix"][0][3]) \
		+ abs(tr_json["frames"][1]["transform_matrix"][1][3]) ) / 4			
if translation_val_sample > 15:
	tr_json["scale"] = 0.01
f_tr_write = open(f'{result_dir}/train_{train_id}/transforms.json', 'w+')
json.dump(tr_json, f_tr_write, ensure_ascii=False, indent='\t')
f_tr_write.close()

# for F in range(frame_start, frame_end + 1):
# 	os.system(f'sudo cp {result_dir}/train_{train_id}/transforms.json  {result_dir}/train_{train_id}/frames/frame{F}')

f_fp_read = open(f'{result_dir}/train_{train_id}/transforms.json', 'r')
orig = json.load(f_fp_read)
f_fp_read.close()
# os.system(f'sudo chmod 777 {result_dir}/train_{train_id}/frames/*')
for F in range(frame_start, frame_end + 1):
	new = orig
	orig_frames = orig["frames"]
	for j, data_frame in enumerate(orig_frames):
		new["frames"][j]["file_path"] = \
				f"{image_base_dir}/images/v{j+view_start_idx}/image-v{j+view_start_idx}-f{str(F).zfill(3)}.png"
	# os.system(f'sudo chmod 777 {result_dir}/train_{train_id}/frames/frame{F}/*')
	f_fp_write = open(f"{result_dir}/train_{train_id}/frames/frame{F}/transforms_train.json", 'w+')
	json.dump(new, f_fp_write, ensure_ascii=False, indent='\t')
	f_fp_write.close()


# Step3: 일부 view를 exclude하는 경우 관련 사항 처리
if config_data["exclude_specific_views"] == "true" or config_data["exclude_specific_views"] == "True":
	exclude_views = config_data["views_to_exclude"]
	train_views =  [i for i in range(view_start_idx, view_start_idx+num_of_views+1) if i not in exclude_views]
	for F in range(frame_start, frame_end + 1):
		f_orig= open(f'{result_dir}/train_{train_id}/frames/frame{F}/transforms_train.json', 'r')
		orig = json.load(f_orig)
		f_orig.close()
		train = copy.deepcopy(orig)
		orig_frames = orig["frames"]
		for i in reversed(range(len(orig_frames))):
			if i not in train_views:
				del(train["frames"][i])
		f_train = open(f"{result_dir}/train_{train_id}/frames/frame{F}/transforms_train.json", "w+")
		json.dump(train, f_train, ensure_ascii=False, indent='\t')
		f_train.close()


# Step4: Instant-NGP 실행 (train, 모델 저장)
# 첫 프레임
print(f"Training Frame{frame_start} ...")
accumulated_iter = initial_n_iters
os.system(f"mkdir {result_dir}/train_{train_id}/models")
os.system(f"mkdir {result_dir}/train_{train_id}/models/frame{frame_start}")
os.system(f"python {ingp_home_dir}/scripts/run.py \
		--scene {result_dir}/train_{train_id}/frames/frame{frame_start}/transforms_train.json \
		--n_steps {accumulated_iter} \
		--save_snapshot {result_dir}/train_{train_id}/models/frame{frame_start}/frame{frame_start}.msgpack \
		> {result_dir}/train_{train_id}/frames/frame{frame_start}/frame{frame_start}_log.txt ")

# 이후 프레임
for F in range(frame_start + 1, frame_end + 1):
	print(f"Training Frame{F} ...")
	accumulated_iter += transfer_n_iters
	os.system(f"mkdir {result_dir}/train_{train_id}/models/frame{F}")
	os.system(f"python {ingp_home_dir}/scripts/run.py \
		--scene {result_dir}/train_{train_id}/frames/frame{F}/transforms_train.json \
		--load_snapshot {result_dir}/train_{train_id}/models/frame{F-1}/frame{F-1}.msgpack \
		--n_steps {accumulated_iter} \
		--save_snapshot {result_dir}/train_{train_id}/models/frame{F}/frame{F}.msgpack \
		> {result_dir}/train_{train_id}/frames/frame{F}/frame{F}_log.txt ")

