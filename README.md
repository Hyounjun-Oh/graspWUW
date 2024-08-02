# GraspUwant

 

 본 프로젝트는 AIRobotics LAB 환경에서 진행되었습니다.
 테이블위에 배치된 물체를 인식하고 사용자의 코멘트에 맞는 물건을 선택합니다.
 선택한 물건을 Pick & Place 하면 테스크는 마무리 됩니다.

LAB 환경은 다음과 같습니다.

 - 매니퓰레이터 : Doosan Robotics A0912 (6-DOF)
 - 카메라 : Realsense D455

패키지 개발 환경은 다음과 같습니다.

 - Ubuntu 20.04 LTS
 - ROS1 Noetic
 - CUDA 11.8/ driver version : 520.61.05


# Methods
사용한 방법은 다음과 같습니다.
 1. YOLOv8을 이용한 Semantic Segmentation (COCO dataset pretrained model)
 2. CLIP을 이용한 사용자 코멘트와 물체간 유사도 비교

## YOLOv8을 이용한 Semantic Segmentation

YOLOv8에서 기본으로 제공하는 yolov8x-seg.pt 사전훈련 모델을 이용해서 물체를 탐지했습니다. 만약 다른 물체 탐지를 원한다면 커스텀 데이터셋으로 학습 후 사용하면 됩니다.
![YOLO object detection result](https://drive.google.com/uc?export=view&id=1sTs89NnW1_lHh9LPAJAXqIfDcZ2lPweN)

## CLIP을 이용한 사용자 코멘트와 물체간 유사도 비교

유저로부터 받은 메세지를 text 인코딩 하고 검출한 객체의 개별 이미지를 image 인코딩 합니다. 두 벡터간 유사도를 비교하여 가장 유사한 워딩의 물체를 선별합니다.
![Input image of CLIP model](https://drive.google.com/uc?export=view&id=1icyAGEBMaWFt5L5eYijT6cBlnylDeEUR)
![Cosine similarity result](https://drive.google.com/uc?export=view&id=1eS-y3ipvmz2ub8_OnYJQCJPrs7eWaCPm)


## Reference

 1. CLIP, https://github.com/openai/CLIP
 2. YOLOv8, https://github.com/ultralytics/ultralytics
