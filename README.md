# miv-dynamic-NGP

## 1. Introduction

This software is implemented based on [instant-NGP](https://github.com/NVlabs/instant-ngp) and [camorph](https://github.com/Fraunhofer-IIS/camorph).

Reconstruct your dynamic 3D scene by Neural Network Model!!<br>
By training  "[instant-NGP](https://github.com/NVlabs/instant-ngp)" model per frame using "Transfer Learning", We provide relatively-high-speed and time-consistent video.<br>
For camera parameter and pose trace format, we adopted MPEG-Immersive-Video(MIV) standard.<br>
- Input: Set of yuv files, MIV-format camera parameter file, and (optionally) MIV-format pose trace file.<br>
- Output: Rendered video of novel views containing dynamic scene.<br>

The overall time of data transforming, training, and rendering is about 1 hour for 90 frames. 


## 2. Usage

### 2-1. Installation

### 2-2. Run 

## 3. Theory

## 4. Implementation