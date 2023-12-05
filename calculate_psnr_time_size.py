import cv2
import json
import os
import re
import pandas as pd
import datetime
from datetime import datetime

train_log_file = "path_to_train_log_file.txt"
model_base_dir = 'path_to_results/models'
render_log_file = "path_to_render_log_file.txt"
result_base_dir = "path_to_results/results"


train_exp_id = None
images_base_dir = None
view_start_idx = None
num_of_views = None
render_exp_id = None
frame_start = None
frame_end = None
test_set_views = None

with open(train_log_file, 'r') as f:
	lines = f.readlines()
	for line in lines:
		if "train_exp_id" in line:
			train_exp_id = line.split(":")[1].strip()
		elif "images_base_dir" in line:
			images_base_dir = line.split(":")[1].strip()
		elif "view_start_idx" in line:
			view_start_idx = int(line.split(":")[1].strip())
		elif "num_of_views" in line:
			num_of_views = int(line.split(":")[1].strip())

with open(render_log_file, 'r') as f:
	lines = f.readlines()
	for line in lines:
		if "render_exp_id" in line:
			render_exp_id = line.split(":")[1].strip()
		if "frame_start" in line:
			frame_start = int(line.split(":")[1].strip())
		if "frame_end" in line:
			frame_end = int(line.split(":")[1].strip())
		if "test_set_view" in line:
			values_str = line.split(":")[1].strip()
			test_set_views = [int(val) for val in values_str[1:-1].split(",")]

# Part1. PSNR 
psnr_csv_file = f"/data/jychoi/results/PSNR_train_{train_exp_id}_render_{render_exp_id}.csv"

datas = dict()
datas["Frames"] = []
for V in test_set_views:
	datas[f"view{V} PSNR"] = []

for i, F in enumerate(range(frame_start, frame_end+1)):
	datas["Frames"].append(f"Frame {F}")
	for j, V in enumerate(test_set_views):
		orig_path = f"{images_base_dir}/images/v{V}/image-v{V}-f{str(F).zfill(3)}.png"
		target_path = f"{result_base_dir}/train_{train_exp_id}/render_{render_exp_id}/temp/frame{F}/testview_outputs/image-v{V}-f{str(F).zfill(3)}.png"
		os.system(f'sudo chmod 777 {orig_path}')
		orig = cv2.imread(orig_path)
		target = cv2.imread(target_path)
		psnr = cv2.PSNR(orig, target)
		datas[f"view{V} PSNR"].append(psnr)
		os.system(f'sudo chmod 774 {orig_path}')

datas["Frames"].append("Average")
for V in test_set_views:
	sum = 0
	for i, F in enumerate(range(frame_start, frame_end+1)):
		sum += datas[f"view{V} PSNR"][i]
	datas[f"view{V} PSNR"].append(float(sum) / (frame_end - frame_start + 1))

df = pd.DataFrame(datas)
df.set_index("Frames", inplace=True)
os.system(f"sudo chmod 777 {result_base_dir}")
df.to_csv(f"{psnr_csv_file}")
os.system(f"sudo chmod 774 {result_base_dir}")



# Part2. Time Info

time_csv_file = f"/data/jychoi/results/TIME_train_{train_exp_id}_render_{render_exp_id}.csv"

datas = []
# columns = ["Index", "Training Start", "Training End", "Training Duration",
# 	    "Rendering GPU", "Rendering Start", "Rendering End", "Rendering Duration"]
# df = pd.DataFrame(columns=columns)

pattern_start_training = re.compile(r'\[(.*?)\] - Start training frame(\d+)')
pattern_end_training = re.compile(r'\[(.*?)\] - End training frame(\d+)')
pattern_start_rendering = re.compile(r'\[(.*?)\] - Start rendering frame(\d+)')
pattern_end_rendering = re.compile(r'\[(.*?)\] - Finish rendering frame(\d+)')

with open(train_log_file, "r") as f:
	train_log = f.readlines() 
	for line in train_log:
		match_start_training = pattern_start_training.search(line)
		match_end_training = pattern_end_training.search(line)
		if match_start_training:
			time_start = datetime.strptime(match_start_training.group(1), '%Y-%m-%d %H:%M:%S.%f')
			frame_number = int(match_start_training.group(2))
			datas.append({'Index': frame_number, 'Training Start': time_start})
		if match_end_training:
			time_end = datetime.strptime(match_end_training.group(1), '%Y-%m-%d %H:%M:%S.%f')
			frame_number = int(match_end_training.group(2))
			for data in datas:
				if data['Index'] == frame_number:
					data['Training End'] = time_end
					data['Training Duration'] = str(data['Training End'] - data['Training Start'])


with open(render_log_file, 'r') as f:
	render_log = f.readlines()
	for line in render_log:
		match_start_rendering = pattern_start_rendering.search(line)
		match_end_rendering = pattern_end_rendering.search(line)
		if match_start_rendering:
			time_start = datetime.strptime(match_start_rendering.group(1), '%Y-%m-%d %H:%M:%S.%f')
			frame_number = int(match_start_rendering.group(2))
			for data in datas:
				if data['Index'] == frame_number:
					data['Rendering Start'] = time_start
		if match_end_rendering:
			time_end = datetime.strptime(match_end_rendering.group(1), '%Y-%m-%d %H:%M:%S.%f')
			frame_number = int(match_end_rendering.group(2))
			for data in datas:
				if data['Index'] == frame_number:
					data['Rendering End'] = time_end
					data['Rendering Duration'] = str(data['Rendering End'] - data['Rendering Start'])


df = pd.DataFrame(datas)
df.sort_values('Index', inplace=True)

df['Training Duration'] = pd.to_timedelta(df['Training Duration'])
df['Rendering Duration'] = pd.to_timedelta(df['Rendering Duration'])
avg_training_duration = df['Training Duration'].mean()
avg_rendering_duration = df['Rendering Duration'].mean()

total_train_duration = df.iloc[-1]['Training End'] - df.iloc[0]['Training Start']
total_render_duration = df.iloc[-1]['Rendering End'] - df.iloc[0]['Rendering Start']

df = df.append({
	'Index': 'Total',
	'Training Start': df.iloc[0]['Training Start'],
	'Training End': df.iloc[-1]['Training End'],
	'Training Duration': total_train_duration,
	'Rendering Start': df.iloc[0]['Rendering Start'],
	'Rendering End': df.iloc[-1]['Rendering End'],
	'Rendering Duration': total_render_duration
}, ignore_index=True)

df = df.append({
	'Index': 'Average',
	'Training Duration': avg_training_duration,
	'Rendering Duration': avg_rendering_duration,
	'Training Start': '',
	'Training End': '',
	'Rendering Start': '',
	'Rendering End': ''
}, ignore_index=True)

os.system(f"sudo chmod 777 {result_base_dir}")
df.to_csv(time_csv_file, index=False)
os.system(f"sudo chmod 774 {result_base_dir}")


# Part3. model size
size_csv_file = f"/data/jychoi/results/SIZE_train_{train_exp_id}_render_{render_exp_id}.csv"

frame_dirs = [f for f in os.listdir(model_base_dir) if os.path.isdir(os.path.join(model_base_dir, f))]
data = []
for dir in frame_dirs:
	file_path = os.path.join(model_base_dir, dir, dir + '.msgpack')
	if os.path.exists(file_path):
			size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 3)
			frame_number = int(dir.replace('frame', ''))
			data.append([frame_number, size_mb])
	else:
			print(f"File {file_path} not found.")

df = pd.DataFrame(data, columns=['Frame', 'Model Size (MB)'])
	
df = df.sort_values(by='Frame')

avg_size = df['Model Size (MB)'].mean()
df = df.append({'Frame': 'average', 'Model Size (MB)': avg_size}, ignore_index=True)

# save
df.to_csv(size_csv_file, index=False)
