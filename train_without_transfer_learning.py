import json
import os
from os import path
import subprocess
import copy
import datetime

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
cuda_device_nums = config_data["cuda_device_nums"]
use_transforms_json = config_data["use_transforms_json"]
path_of_transforms_json = config_data["path_of_transforms_json"]
ingp_home_dir = '.'


view_start_idx = 0		 # will be modified below. don't touch.
num_of_views = 0			 # will be modified below. don't touch.

os.system(f'mkdir {result_dir}')
os.system(f'mkdir {result_dir}/train_{train_id}')

start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
with open(f"{result_dir}/train_{train_id}/log.txt", "w") as log_file:
	log_file.write("DO NOT DELETE THIS LOG FILE !!\n\n")
	log_file.write(f"[{start_time}] - Start!!\n")


with open(f"{result_dir}/train_{train_id}/log.txt", "a") as log_file:
	log_file.write("\n")
	for key, value in config_data.items():
		log_file.write(f"{key}: {value}\n")



if frame_start <= 0:
	print('frame_start should be 1 or bigger')
	exit()
if frame_start >= frame_end:
	print('frame_start should be smaller than frame_end')
	exit()



if path.exists(f'{image_base_dir}/images'):
	views_list = os.listdir(f'{image_base_dir}/images')	
	views_list.sort()
	if '_v0' in views_list[0]:
		view_start_idx = 0
	elif '_v1' in views_list[0]:
		view_start_idx = 1
	else:
		pass
	num_of_views = len(views_list)

	
	with open(f"{result_dir}/train_{train_id}/log.txt", "a") as log_file:
		log_file.write(f"view_start_idx: {view_start_idx}\n")
		log_file.write(f"num_of_views: {num_of_views}\n")
		log_file.write("\n")
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
	
	
	with open(f"{result_dir}/train_{train_id}/log.txt", "a") as log_file:
		log_file.write(f"view_start_idx: {view_start_idx}\n")
		log_file.write(f"num_of_views: {num_of_views}\n")
		log_file.write("\n")
	
	os.system(f'sudo mkdir {image_base_dir}')
	os.system(f'sudo mkdir {image_base_dir}/images')

	# 로그 파일에 동영상 변환 시작 시각 기록
	time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
	with open(f"{result_dir}/train_{train_id}/log.txt", "a") as log_file:
		log_file.write(f"[{time}] - Start converting yuv to png!!\n\n")

	for V in range(view_start_idx, num_of_views + view_start_idx):
		video_file = [x for x in video_file_list if f'v{V}_' in x or f'v0{V}_' in x]
		video_file = video_file[0]
		os.system(f'sudo mkdir {image_base_dir}/images/v{V}; \
			sudo ffmpeg -pixel_format yuv420p10le \
		-video_size {input_video_size} -framerate {input_frame_rate} \
		-i {yuv_dir}/{video_file} \
		-f image2 -pix_fmt rgba \
		{image_base_dir}/images/v{V}/image-v{V}-f%3d.png')

	
	time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
	with open(f"{result_dir}/train_{train_id}/log.txt", "a") as log_file:
		log_file.write(f"[{time}] - Success converting yuv to png. Saved at {image_base_dir}/images/\n\n")


if use_transforms_json == "true":
	os.system(f'cp {path_of_transforms_json} {result_dir}/train_{train_id}/transforms.json')

else:
	os.system(f'mkdir {result_dir}/train_{train_id}/frames')
	for F in range(frame_start, frame_end + 1):
		os.system(f'mkdir {result_dir}/train_{train_id}/frames/frame{F}')

	subprocess.call(f'python ./camorph/main.py 0 {miv_json_path} {result_dir}/train_{train_id}/transforms.json \
			{num_of_views} 0', shell=True)

	time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
	with open(f"{result_dir}/train_{train_id}/log.txt", "a") as log_file:
		log_file.write(f"[{time}] - Sucess converting camera parameter (OMAF->NeRF) {result_dir}/train_{train_id}/transforms.json\n\n")




f_tr_json = open(f'{result_dir}/train_{train_id}/transforms.json', 'r')
orig_tr_json = json.load(f_tr_json)
tr_json = orig_tr_json
f_tr_json.close()
# tr_json["aabb_scale"] = 64
# # tr_json["offset"] = [-1, -2, 0]
# translation_val_sample = \
# 	( abs(tr_json["frames"][0]["transform_matrix"][0][3]) \
# 		+ abs(tr_json["frames"][0]["transform_matrix"][1][3]) \
# 		+ abs(tr_json["frames"][1]["transform_matrix"][0][3]) \
# 		+ abs(tr_json["frames"][1]["transform_matrix"][1][3]) ) / 4			
# if translation_val_sample > 15:
# 	tr_json["scale"] = 0.01
f_tr_write = open(f'{result_dir}/train_{train_id}/transforms.json', 'w+')
json.dump(tr_json, f_tr_write, ensure_ascii=False, indent='\t')
f_tr_write.close()


f_fp_read = open(f'{result_dir}/train_{train_id}/transforms.json', 'r')
orig = json.load(f_fp_read)
f_fp_read.close()
for F in range(frame_start, frame_end + 1):
	new = orig
	orig_frames = orig["frames"]
	for j, data_frame in enumerate(orig_frames):
		new["frames"][j]["file_path"] = \
				f"{image_base_dir}/images/v{j+view_start_idx}/image-v{j+view_start_idx}-f{str(F).zfill(3)}.png"
	f_fp_write = open(f"{result_dir}/train_{train_id}/frames/frame{F}/transforms_train.json", 'w+')
	json.dump(new, f_fp_write, ensure_ascii=False, indent='\t')
	f_fp_write.close()



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



print(f"Training Frame{frame_start} ...")

time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
with open(f"{result_dir}/train_{train_id}/log.txt", "a") as log_file:
	log_file.write(f"[{time}] - Start training frame{frame_start}. Now accumulated iter: 0\n\n")

os.system(f"mkdir {result_dir}/train_{train_id}/models")
os.system(f"mkdir {result_dir}/train_{train_id}/models/frame{frame_start}")
os.system(f"CUDA_VISIBLE_DEVICES={cuda_device_nums[0]} \
	  python {ingp_home_dir}/scripts/run.py \
	  --network {ingp_home_dir}/configs/nerf/dyngp_initial.json \
		--scene {result_dir}/train_{train_id}/frames/frame{frame_start}/transforms_train.json \
		--n_steps {initial_n_iters} \
		--save_snapshot {result_dir}/train_{train_id}/models/frame{frame_start}/frame{frame_start}.msgpack \
		> {result_dir}/train_{train_id}/frames/frame{frame_start}/frame{frame_start}_log.txt ")

time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
with open(f"{result_dir}/train_{train_id}/log.txt", "a") as log_file:
	log_file.write(f"[{time}] - End training frame{frame_start}. Model saved at {result_dir}/train_{train_id}/models/frame{frame_start}/frame{frame_start}.msgpack\n\n")


for F in range(frame_start + 1, frame_end + 1):
	print(f"Training Frame{F} ...")
	time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
	with open(f"{result_dir}/train_{train_id}/log.txt", "a") as log_file:
		log_file.write(f"[{time}] - Start training frame{F}.\n\n")
	
	os.system(f"mkdir {result_dir}/train_{train_id}/models/frame{F}")
	os.system(f"CUDA_VISIBLE_DEVICES={cuda_device_nums[0]} \
	  python {ingp_home_dir}/scripts/run.py \
		--scene {result_dir}/train_{train_id}/frames/frame{F}/transforms_train.json \
		--n_steps {transfer_n_iters} \
		--save_snapshot {result_dir}/train_{train_id}/models/frame{F}/frame{F}.msgpack \
		> {result_dir}/train_{train_id}/frames/frame{F}/frame{F}_log.txt ")
	time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
	with open(f"{result_dir}/train_{train_id}/log.txt", "a") as log_file:
		log_file.write(f"[{time}] - End training frame{F}. Model saved at {result_dir}/train_{train_id}/models/frame{F}/frame{F}.msgpack\n\n")

time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
with open(f"{result_dir}/train_{train_id}/log.txt", "a") as log_file:
	log_file.write(f"[{time}] - Success on training frame{frame_start} to frame{frame_end}\n\n")
	log_file.write(f"You can now render 6DoF video by running render.py\n\n")
