import json
import os
from os import path
import subprocess
import copy

f_config = open('./config.json', 'r')
config_data = json.load(f_config)
f_config.close()

content_name = config_data["content_name"]
image_base_dir = config_data["images_base_dir"]
result_dir = config_data["result_dir"]

miv_json_path = config_data["path_of_MIV_json_file"]

num_of_original_frames = 0		 # will be modified below. don't touch.
view_start_idx = 0		 # will be modified below. don't touch.
num_of_views = 0			 # will be modified below. don't touch.

ingp_home_dir = '.'
input_frame_rate = config_data["input_frame_rate"]
initial_n_iters = config_data["experiment_config"]["initial_n_iters"]
transfer_n_iters = config_data["experiment_config"]["transfer_learning_n_iters"]

if path.exists(f'{image_base_dir}/images'):
	views_list = os.listdir(f'{image_base_dir}/images')	
	# views_list = [x for x in views_list if x[-3:]=='yuv']
	views_list.sort()
	if '_v0' in views_list[0]:
		view_start_idx = 0
	else:
		view_start_idx = 1
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

	for V in range(num_of_views):
		video_file = [x for x in video_file_list \
		 if f'v{V+view_start_idx}_' in x or f'v0{V+view_start_idx}_' in x]
		video_file = video_file[0]
		os.system(f'sudo mkdir {image_base_dir}/images/{content_name}_v{V+view_start_idx}; \
			sudo ffmpeg -pixel_format yuv420p10le \
		-video_size {input_video_size} -framerate {input_frame_rate} \
		-i {yuv_dir}/{video_file} \
		-f image2 -pix_fmt rgba \
		{image_base_dir}/images/{content_name}_v{V+view_start_idx}/image-v{V+view_start_idx}-f%3d.png')

	num_of_original_frames = len(os.listdir(f'{image_base_dir}/images/{content_name}_v{view_start_idx}'))

	os.system(f'sudo mkdir {result_dir}/frames')
	for F in range(1, num_of_original_frames+1):
		os.system(f'sudo mkdir {result_dir}/frames/frame{F}')
	

	subprocess.call(f'python ./camorph/main.py 0 {miv_json_path} {result_dir}/transforms.json {num_of_views} 0', shell=True)

	f_tr_json = open(f'{result_dir}/transforms.json', 'r')
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
	f_tr_write = open(f'{result_dir}/transforms.json', 'w+')
	json.dump(tr_json, f_tr_write, ensure_ascii=False, indent='\t')
	f_tr_write.close()
	
	for F in range(1, num_of_original_frames+1):
		os.system(f'sudo cp {result_dir}/transforms.json  {result_dir}/frames/frame{F}')
	

	f_fp_read = open(f'{result_dir}/transforms.json', 'r')
	orig = json.load(f_fp_read)
	f_fp_read.close()
	os.system(f'sudo chmod 777 {result_dir}/frames/*')
	for F in range(1, num_of_original_frames+1):
		new = orig
		orig_frames = orig["frames"]
		for j, data_frame in enumerate(orig_frames):
			new["frames"][j]["file_path"] = \
				  f"{image_base_dir}/images/{content_name}_v{j+view_start_idx}/image-v{j+view_start_idx}-f{str(F).zfill(3)}.png"
		os.system(f'sudo chmod 777 {result_dir}/frames/frame{F}/*')
		f_fp_write = open(f"{result_dir}/frames/frame{F}/transforms.json", 'w+')
		json.dump(new, f_fp_write, ensure_ascii=False, indent='\t')
		f_fp_write.close()

num_of_original_frames = len(os.listdir(f'{image_base_dir}/images/{content_name}_v{view_start_idx}'))
exp_name = config_data["experiment_config"]["experiment_name"]

frame_start = config_data["experiment_config"]["frame_start"]
frame_end = config_data["experiment_config"]["frame_end"]
if frame_start == 0:
	print('frame_start cannot be 0 (frame_start >= 1)')
	exit()
if frame_start >= frame_end:
	print('frame_start should be smaller than frame_end')
	exit()


if config_data["experiment_config"]["render_pose_trace"] == "true" or config_data["experiment_config"]["render_pose_trace"] == "True":
	pose_flag = True
else:
	pose_flag = False

if pose_flag:
	pose_csv_path = config_data["experiment_config"]["path_of_MIV_pose_trace_file"]
	subprocess.call(f'python ./camorph/main.py 1 {miv_json_path} {result_dir}/{exp_name}_poses.json \
		  {num_of_views}  {pose_csv_path}', shell=True)
	for F in range(1, num_of_original_frames+1):
		os.system(f'sudo cp {result_dir}/{exp_name}_poses.json  {result_dir}/frames/frame{F}')
		os.system(f'sudo chmod 777  {result_dir}/frames/frame{F}/*')
	
	for F in range(1, num_of_original_frames+1):
		f_ex_read = open(f"{result_dir}/frames/frame{F}/{exp_name}_poses.json", 'r')
		data = json.load(f_ex_read)
		f_ex_read.close()
		data["frames"] = data["frames"][F-1:F]
		f_ex_write = open(f"{result_dir}/frames/frame{F}/{exp_name}_poses.json", "w+")
		json.dump(data, f_ex_write, ensure_ascii=False, indent='\t')
		f_ex_write.close()

	print(f"Training Frame{frame_start} ...")
	accumulated_iter = initial_n_iters
	os.system(f"sudo mkdir {result_dir}/frames/frame{frame_start}/result_{exp_name}_{accumulated_iter};")
	os.system(f"sudo chmod 777 {result_dir}/frames/frame{frame_start}/result_{exp_name}_{accumulated_iter};")
	os.system(f"python {ingp_home_dir}/scripts/run.py \
		--scene {result_dir}/frames/frame{frame_start}/transforms.json \
		--n_steps {accumulated_iter} \
		--screenshot_transforms {result_dir}/frames/frame{frame_start}/{exp_name}_poses.json \
		--screenshot_dir {result_dir}/frames/frame{frame_start}/result_{exp_name}_{accumulated_iter} \
		--save_snapshot {result_dir}/frames/frame{frame_start}/result_{exp_name}_{accumulated_iter}/snapshot.msgpack \
		> {result_dir}/frames/frame{frame_start}/result_{exp_name}_{accumulated_iter}/log.txt ")

	for F in range(frame_start + 1, frame_end+1):  # frame_start+1 -> 241
		print(f"Training Frame{F} ...")
		accumulated_iter += transfer_n_iters
		os.system(f"sudo mkdir {result_dir}/frames/frame{F}/result_{exp_name}_{accumulated_iter};")
		os.system(f"sudo chmod 777 {result_dir}/frames/frame{F}/result_{exp_name}_{accumulated_iter};")
		os.system(f"touch {result_dir}/frames/frame{F}/result_{exp_name}_{accumulated_iter}/log.txt ")
		os.system(f"python {ingp_home_dir}/scripts/run.py \
			--scene {result_dir}/frames/frame{F}/transforms.json \
			--load_snapshot {result_dir}/frames/frame{F-1}/result_{exp_name}_{accumulated_iter-transfer_n_iters}/snapshot.msgpack \
			--n_steps {accumulated_iter} \
			--screenshot_transforms {result_dir}/frames/frame{F}/{exp_name}_poses.json \
			--screenshot_dir {result_dir}/frames/frame{F}/result_{exp_name}_{accumulated_iter} \
			--save_snapshot {result_dir}/frames/frame{F}/result_{exp_name}_{accumulated_iter}/snapshot.msgpack \
			> {result_dir}/frames/frame{F}/result_{exp_name}_{accumulated_iter}/log.txt ")


	os.system(f"mkdir {result_dir}/Output_{exp_name};")
	os.system(f"mkdir {result_dir}/Output_{exp_name}/temp")
	for i in range(frame_start-1, frame_end):
		os.system(f"sudo cp {result_dir}/frames/frame{i+1}/result_{exp_name}_{initial_n_iters+(i*transfer_n_iters)}/posestrace_{str(i).zfill(3)}.png \
	  {result_dir}/Output_{exp_name}/temp/posestrace_{str(i).zfill(3)}.png")
	os.system(f"sudo ffmpeg -framerate {input_frame_rate} -pattern_type glob \
		-i '{result_dir}/Output_{exp_name}/temp/*.png' -c:v libx264 -pix_fmt yuv420p \
		{result_dir}/Output_{exp_name}/poses_video.mp4 ")

else:
	test_views = config_data["experiment_config"]["test_set_view"]
	train_views = [i for i in range(view_start_idx, view_start_idx+num_of_views+1) if i not in test_views]
	for F in range(frame_start, frame_end+1):
		with open(f"{result_dir}/frames/frame{F}/transforms.json", "r") as f_orig:
			orig = json.load(f_orig)
			train = copy.deepcopy(orig)
			test = copy.deepcopy(orig)
			orig_frames = orig["frames"]
			for i in reversed(range(len(orig_frames))):
				if i not in train_views:
					del(train["frames"][i])
				if i not in test_views:
					del(test["frames"][i])
			f_train = open(f"{result_dir}/frames/frame{F}/{exp_name}_transforms_train.json", "w")
			f_test = open(f"{result_dir}/frames/frame{F}/{exp_name}_transforms_test.json", "w")
			json.dump(train, f_train, ensure_ascii=False, indent='\t')
			json.dump(test, f_test, ensure_ascii=False, indent='\t')
			f_train.close()
			f_test.close()
	
	
	print(f"Training Frame{frame_start} ...")
	accumulated_iter = initial_n_iters
	os.system(f"sudo mkdir {result_dir}/frames/frame{frame_start}/result_{exp_name}_{accumulated_iter};")
	os.system(f"sudo chmod 777 {result_dir}/frames/frame{frame_start}/result_{exp_name}_{accumulated_iter};")
	os.system(f"python {ingp_home_dir}/scripts/run.py \
		--scene {result_dir}/frames/frame{frame_start}/{exp_name}_transforms_train.json \
		--n_steps {accumulated_iter} \
		--screenshot_transforms {result_dir}/frames/frame{frame_start}/{exp_name}_transforms_test.json \
		--screenshot_dir {result_dir}/frames/frame{frame_start}/result_{exp_name}_{accumulated_iter} \
		--save_snapshot {result_dir}/frames/frame{frame_start}/result_{exp_name}_{accumulated_iter}/snapshot.msgpack \
		> {result_dir}/frames/frame{frame_start}/result_{exp_name}_{accumulated_iter}/log.txt ")

	for F in range (frame_start+1, frame_end+1):
		print(f"Training Frame{F} ...")
		accumulated_iter += transfer_n_iters
		os.system(f"sudo mkdir {result_dir}/frames/frame{F}/result_{exp_name}_{accumulated_iter};")
		os.system(f"sudo chmod 777 {result_dir}/frames/frame{F}/result_{exp_name}_{accumulated_iter};")
		os.system(f"python {ingp_home_dir}/scripts/run.py \
			--scene {result_dir}/frames/frame{F}/{exp_name}_transforms_train.json \
			--load_snapshot {result_dir}/frames/frame{F-1}/result_{exp_name}_{accumulated_iter-transfer_n_iters}/snapshot.msgpack \
			--n_steps {accumulated_iter} \
			--screenshot_transforms {result_dir}/frames/frame{F}/{exp_name}_transforms_test.json \
			--screenshot_dir {result_dir}/frames/frame{F}/result_{exp_name}_{accumulated_iter} \
			--save_snapshot {result_dir}/frames/frame{F}/result_{exp_name}_{accumulated_iter}/snapshot.msgpack \
			> {result_dir}/frames/frame{F}/result_{exp_name}_{accumulated_iter}/log.txt ")

	os.system(f"mkdir {result_dir}/Output_{exp_name};")
	for V in test_views:
		os.system(f"mkdir {result_dir}/Output_{exp_name}/temp_v{V}")
	for V in test_views:
		for i in range(frame_start, frame_end+1):
			os.system(f"sudo cp {result_dir}/frames/frame{i}/result_{exp_name}_{initial_n_iters+((i-1)*transfer_n_iters)}/image-v{V}-f{str(i).zfill(3)}.png \
				{result_dir}/Output_{exp_name}/temp_v{V}/image-v{V}-f{str(i).zfill(3)}.png")
		os.system(f"ffmpeg -framerate {input_frame_rate} -pattern_type glob \
		-i '{result_dir}/Output_{exp_name}/temp_v{V}/*.png' -c:v libx264 -pix_fmt yuv420p \
		{result_dir}/Output_{exp_name}/view{V}.mp4 ")
	for V in test_views:
		os.system(f"sudo rm {result_dir}/Output_{exp_name}/temp_v{V}/*.png")
		os.system(f"sudo rm -r {result_dir}/Output_{exp_name}/temp_v{V}")

# test!!