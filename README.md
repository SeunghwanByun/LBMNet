# LBMNet
A Road Area Detection Using Local Boundary Mask-based Deep Neural Network

### Abstract
본 논문은 카메라와 LiDAR를 카메라 평면에 투영함으로서 생기는 단점을 보완하기 위해서 카메라와 LiDAR의 센서 융합을 통해 LiDAR에 색상 정보를 추가하고 구면좌표계에 투영함으로서 LiDAR의 희소성 문제를 해결한다. 또한 도로 영역 검출의 정밀도를 높이기 위해 네트워크의 특징맵이 도로 영역 경계를 확실하게 구분하도록 학습하는 지역 경계 마스크 모듈을 제안한다. 이를 통해 실시간 도로 영역 검출 알고리즘을 제안한다.

### Network Architecture
<p align="center">
  <img src=https://user-images.githubusercontent.com/49049277/105051607-45ffc280-5ab2-11eb-9998-0b17936806c0.png width=100%>
</p>

### Result
#### Comparison with state-of-the-art
<img src=https://user-images.githubusercontent.com/49049277/105048920-75f99680-5aaf-11eb-8ec3-432822e0930a.png width="60%">

#### KITTI results
<img src=https://user-images.githubusercontent.com/49049277/105050153-d63d0800-5ab0-11eb-8f57-2bc27a126b23.png width="60%">
  
#### Real-time road detection in ROS
<img src=https://user-images.githubusercontent.com/49049277/105048599-18654a00-5aaf-11eb-9e6a-52ab64cf0098.gif>
