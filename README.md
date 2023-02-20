# MIV-dynamic-NGP


## 1. Introduction

This software is implemented based on [instant-NGP](https://github.com/NVlabs/instant-ngp) and [camorph](https://github.com/Fraunhofer-IIS/camorph).

Represent 3D dynamic data implicitly on Neural Network model, and render the video based on your own pose trace.<br>
By training  "[instant-NGP](https://github.com/NVlabs/instant-ngp)" model using transfer learning per frame, We provide temporal-consistent video with  relatively-high-speed.<br>
For camera parameter and pose trace format, we adopted MPEG-Immersive-Video(MIV) standard.<br>
- Input: Set of `.yuv` files, MIV-format camera parameter file, and (optionally) MIV-format pose trace file. (depth map is NOT required.)<br>
- Output: Rendered video of novel views containing dynamic scene.<br>

The overall time of data transforming, training, and rendering is about 1 hour for 90 frames. 

---

## 2. Usage
For now, this software is available only Linux or Windows WSL2. (not native Windows)

## 2-1. Building
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
git clone --recursive https://github.com/jaeyeol816/miv-dynamic-NGP.git
cd miv-dynamic-NGP
```
- WARNING: Don't forget to include `--recursive` option when cloning.

**3. Create and activate anaconda environment**
```bash
conda env create -f miv_ngp.yml
```
- The command above helps you make the suitable anaconda environment named `miv_ngp` for this software.
```bash
conda activate miv_ngp
```
- Activate the `miv_ngp` environment.

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


## 2-2. Executing

In constrast to building step, the running step is quite simple.<br>
To summrize, the only thing you have to do is (1) filling the `config.json` and (2) runnning python script as `python main.py`.

**1. Filling config files**

Fill out the `config.json` file to give information such as location of the yuv files and iteration time.<br>
The term "content" is related to your `yuv` content. The one execution of the program means one "experiment". The multiple "experiment" can be done with one "content".
- `content_name`: Any name can be set as content name.
- `path_of_dir_containing_only_texture_yuv`: The path of Folder that contains bunch of yuv files.
	- WARNING: depth file should **NOT** be in this directory.
	- WARNING: The file name should **contain string that show view info as `v2_` or `v02_`**. Then the software will automatically detect your view number.
- `path_of_MIV_json_file`: the location of camera path json file like `A.json`, `S.json`
- `experiment_config`: At each experiment, giving the suitable `experiment_name` is recommended.
- `initial_n_iters`: The number of iterations of the first frame.
- `transfer_learning_n_iters`: The number of iterations of the second~last frame. For specific frame, the training is held using weights(parameter) of previous frame.
- `frame_start`, `frame_end`: You can set how much frame you will train, render. 
	- WARNING: The index of frame start at 1
- **IMPORTANT:** `render_pose_trace`, `render_test_set`
	- There are two options for rendering.
	- First Option:  If you choose `"render_pose_trace": "true"`, then you will get the output video based on your pose trace file.
		- The pose trace file should be MIV format.
	- Second Option: If you choose `"render_test_set": "false"`, then you will get the output video rendered in specific viewing position.
		- The `"test_set_view"` you selected will be **excluded** in training step. And it will become the rendering view position.

**2. Run python file**
```bash
python main.py
```
- All the step will be done automatically.
	- Includes (1) converting yuv to png files, (2) converting camera parameter based on `MPEG OMAF` coordinate to `NeRF` coordinate, (3) making directories for each frame and change crucial information in json file, (4) training deep learning model, and (5) make video by concatenating test view files.
- It will take some time (depending on `transfer_learning_n_iters` and num of frames). Take some coffee, or have dinner.

**3. Seeing Results**
- your output video will be generated in `Data/{content_name}/Output_{exp_name}`.
	- If you selected 'pose trace render' mode, the name of the video will be `poses_video.mp4`.
	- If you selected 'test set render' mode, the name of the video will be `view{view_number}.mp4`.

---

## 3. Theory

## 4. Implementation
