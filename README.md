# LBMNet - A Study on Road Area Detection Using Local Boundary Mask-based Deep Neural Network
A Road Area Detection Using Local Boundary Mask-based Deep Neural Network

Created by Seunghwan Byun from Soongsil University.

#### Introduction

This repository is code release for my graduate paper (paper report [here](http://oasis.dcollection.net/public_resource/pdf/200000361396_20220225135302.pdf)).

### Abstract
This paper solves the sparseness problem of LiDAR by adding color information to the LiDAR through sensor fusion of the camera and LiDAR and projecting it to the spherical coordinate system to compensate for the shortcomings caused by projecting the camera and LiDAR onto the camera plane. In addition, to increase the precision of road area detection, we propose a regional boundary mask module that trains the network feature map to reliably distinguish road area boundaries. Through this, a real-time road area detection algorithm is proposed.

### Network Architecture
<p align="center">
  <img src=https://user-images.githubusercontent.com/49049277/105051607-45ffc280-5ab2-11eb-9998-0b17936806c0.png width=100%>
</p>

### Setup
```
LBMNet
 |---- main.py
 |---- models
 |    |---- model_epoch_xxxx.pth
 |---- data_road
 |    |---- training
 |         |---- image_2
 |         |---- calib
 |         |---- gt_txt
 |    |---- testing
 |         |---- image_2
 |         |---- calib
 |---- data_road_velodyne
 |    |---- training
 |         |---- velodyne
 |    |---- testing
 |         |---- velodyne
```

### Usage
For training
```
python main.py
```

For inference
```
python inference_raw_data.py
```

### Result
#### Comparison with state-of-the-art
<img src=https://user-images.githubusercontent.com/49049277/105048920-75f99680-5aaf-11eb-8ec3-432822e0930a.png width="60%">

#### KITTI results
<img src=https://user-images.githubusercontent.com/49049277/105050153-d63d0800-5ab0-11eb-8f57-2bc27a126b23.png width="60%">
  
#### Real-time road detection in ROS
<img src=https://user-images.githubusercontent.com/49049277/105048599-18654a00-5aaf-11eb-9e6a-52ab64cf0098.gif>
