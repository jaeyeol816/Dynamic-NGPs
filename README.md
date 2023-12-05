# Dynamic-NGPs

### Citation
```
 @article{Jaey2023DNGP,
    author    = {Choi, Jaeyeol and Jeong, Jong-Beom and Park, JunHyeong and Ryu, Eun-Seok},
    title     = {A Deep Learning-based 6DoF Video Synthesizing Method Using Instant-NGPs},
    journal   = {2023 IEEE Visual Communications and Image Processing (VCIP)},
    year      = {2023},
    }
```

## 1. Introduction
![overview](https://user-images.githubusercontent.com/80497842/237037849-3c183b5e-ba8f-46fb-a4a2-a0f7a9d8a525.gif)

Represent 3D dynamic data implicitly on Neural Network models, and render the video based on your own pose trace.<br>

This software is implemented based on [instant-NGP](https://github.com/NVlabs/instant-ngp) and [camorph](https://github.com/Fraunhofer-IIS/camorph).

By training  "[instant-NGP](https://github.com/NVlabs/instant-ngp)" model using transfer learning per frame, We provide temporal-consistent video with  relatively-high-speed.<br>
For camera parameter and pose trace format, we adopted MPEG-Immersive-Video(MIV) standard.<br>
- Input: Set of `.yuv` files, MIV-format camera parameter file, and (optionally) MIV-format pose trace file. (depth map is NOT required.)<br>
- Output: Rendered video of novel views containing dynamic scene.<br>

The overall time of data transforming, training, and rendering is about 1 hour for 90 frames. 

---
## 2. Architecture
![Screenshot 2023-06-14 at 4 18 32 PM](https://github.com/jaeyeol816/Dynamic-NGPs/assets/80497842/f66af57b-2509-4fdb-86e2-6754589ee9ea)

- All process shown above are implemented as two scripts, `train.py` and `render.py`. 
- You train the dynamic scene with `python train.py`. The conditions you need to give (e.g. location of video files, or iterations settings) should be entered on `train_config.json`
- After you train the models, you can render the video anytime you want by running `python render.py`. The conditions you need to give (e.g. which models you will use, or the pose trace you want to render) should be entered on `test_config.json`
> Some features are not implemented yet (e.g. using colmap). It will be updated soon.

- The concrete explaination is written below.
---

## 3. Usage
For now, this software is available only on Linux or WSL2. (not native Windows)

## 3-1. Building
**1. Pre requirements**
- CUDA 10.2 or higher 
	- [click here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) for CUDA installation for linux
	- [click here](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) for CUDA installation for WSL2.
	- in `~/.bashrc` or `~/.zshrc`, those line should be added. (e.g. if you have CUDA 11.7). Then run `source ~/.bashrc` or `source ~/.zshrc`.
	```bash
	export PATH="/usr/local/cuda-11.7/bin:$PATH"
	export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
	```
- Anaconda 
	- [click here](https://docs.anaconda.com/anaconda/install/linux/) for anaconda installation.
	- run `conda --version` to check whether it's available.
- CMake **v3.21 or higher**
	- [click here](https://cmake.org/download/) for download.

**2. Cloning the repository**
```bash
git clone --recursive https://github.com/jaeyeol816/Dynamic-NGP.git
cd Dynamic-NGP
```
- WARNING: Don't forget to include `--recursive` option when cloning.

**3. Create and activate anaconda environment**
```bash
conda env create -f dy_ngp.yml
```
- The command above helps you make the suitable anaconda environment named `miv_ngp` for this software.
```bash
conda activate dy_ngp
```
- Activate the `dy_ngp` environment.

**4. Installing the packages**
```bash
sudo apt-get install ffmpeg build-essential git python3-dev python3-pip libopenexr-dev libxi-dev libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev
```

**5. Compliation**
```bash
cmake . -B build
cmake --build build --config RelWithDebInfo -j
```
- If you are running out of memory, try running the command without `-j`.
- The command would take some time.
- If the building error occured, the [issues from instant-ngp](https://github.com/NVlabs/instant-ngp/issues?q=) will help.
	- Although, the building step for this sortware is not identical to instant-ngp. The `CMakeLists.txt` in this project set to do not build GUI part of instant-ngp.


## 3-2. Training

In constrast to building step, the running step is quite simple.<br>
To summerize, the only thing you have to do is (1) filling the `train_config.json` and (2) runnning python script as `python train.py`.

**1. Fill config file**

Fill out the `train_config.json` file to give information such as location of the yuv files and iteration time.<br>
The term `train_id` means a "training experiment". The multiple "rendering experiment" can be done with one training experiment
- `train_exp_id`: Any name can be set to represent specific traininig experiment.
- `path_of_dir_containing_only_texture_yuv`: The path of Folder that contains bunch of yuv files.
	- WARNING: depth file should NOT be in this directory.
	- WARNING: The file name should **contain string that show view info as `v2_` or `v02_`**. Then the software will automatically detect your view number.
- `path_of_MIV_json_file`: the location of camera path json file like `A.json`, `S.json`
- `image_base_dir`: The location where we will store the converted `png` files. **If they already exist, we will not convert.**
- `initial_n_iters`: The number of iterations of the first frame.
- `transfer_learning_n_iters`: The number of iterations of the second~last frame. For specific frame, the training is held using weights(parameter) of previous frame.
- `frame_start`, `frame_end`: You can set how much frame you will train.
- `exclude_specific_views`: if "true", then we will not train Instant-NGP with the views which you designate in `views_to_exclude`

**2. Run python file**
```bash
python train.py
```
- All the step will be done automatically.
- It will take some time (depending on `transfer_learning_n_iters` and num of frames). Take some coffee, or have dinner.

**3. Location of models and log files**
- The location of the models: `{result_dir}/train_{train_id}/render_{render_id}/models`
- The location of the log file: `{result_dir}/train_{train_id}/log.txt`
	- the time info and training info are written in log file
- you should not delete model and log file, because it will be used in rendering process.


## 3-3. Rendering (Creating a video)

The rendering experiment is subordinated to training experiemnt. 
To summerize, the only thing you have to do is (1) filling the `render_config.json` and (2) runnning python script as `python render.py`.

**1. Fill config file**
- `train_exp_id`: Specifying the trained model you want to render.
- `render_exp_id`: Any name that represents this rendering experiment. The many rendering experiemnt is available on one training experiment.
  
- `result_dir`: It should be same as `result_dir` of your training case (the training experiment that `train_exp_id` points)
- `frame_start`, `frame_end`: The frames you want to render. It should be subset of the frames you trained.

- IMPORTANT: `render_pose_trace`, `render_test_set`
	- There are two options for rendering.
	- First Option:  If you choose `"render_pose_trace": "true"`, then you will get the output video based on your pose trace file. (which should be located in `path_of_MIV_pose_trace_file`)
		- For now, the pose trace file should be MIV format.
	- Second Option: If you choose `"render_test_set": "true"`, then you will get the output video rendered in specific viewing position. (fill in `test_set_view`)

**2. Run python file**
```bash
python render.py
```

**3. See Results**
- Your output video will be generated in `{result_dir}/train_{train_id}/render_{render_id}/{poses_video or viewN}.mp4`
- The rendering log is located in `{result_dir}/train_{train_id}/render_{render_id}/log.txt`
