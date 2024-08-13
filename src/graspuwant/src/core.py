#!/usr/bin/env python3
import signal
import clip
import sys
import threading
from ultralytics import YOLO
import torch
import rospy
import numpy as np
import cv2
import queue
from cv_bridge import CvBridge
from PIL import Image as PILImage

from std_msgs.msg import Int64
from sensor_msgs.msg import Image

YOLO_MODEDL_PATH = '/root/yolov8x-seg.pt' # COCO pretrained model for segmentation
CLIP_MODEL = 'ViT-B/32'

command_flag = False

# mode 0 : real-time
# mode 1 : rosbag
MODE = 1
CROPPED_IMG_SIZE = (320,320)
CLASS_INFO = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 
    8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 
    29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 
    71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 
    78: 'hair drier', 79: 'toothbrush'
}

# 1280 * 720 pinhole
CAMERA_INTRINSIC = [635.5997366807397,635.1073315378363,633.9239491636653,369.02950972096005] # fx, fx, cx, cy
CAMERA_DISTORTION = [-0.055731027995673074, 0.040781967334706, -0.00014895584848947406, 0.0003826882589985031] # k1, k2, p1, p2
K = np.array([CAMERA_INTRINSIC[0],0,CAMERA_INTRINSIC[2]],
             [0,CAMERA_INTRINSIC[1],CAMERA_INTRINSIC[3]],
             [0,                  0,                  0])

rgb_queue = queue.Queue()
depth_queue = queue.Queue()
command_queue = queue.Queue()

class Core:
    def __init__(self):
        # ros setting 
        rospy.init_node('core', anonymous=True)
        self.command_sub = rospy.Subscriber('/command', Int64, self.command_callback)
        self.rgb_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        
        # Yolo model setting
        self.yolo_model = YOLO(YOLO_MODEDL_PATH)
        self.bridge = CvBridge()
        
        # CLIP model setting
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load(CLIP_MODEL, self.device)
        
        # process threading
        self.process_thread = threading.Thread(target=self.run_process)
        self.process_thread.daemon = True
        self.process_thread.start()
        rospy.loginfo("CORE node initialization finished !")
    
    # callback function        
    def rgb_callback(self, msg):
        # rospy.loginfo("RGB callback called")
        rgb_queue.put(msg)
        
    def depth_callback(self, msg):
        # rospy.loginfo("Depth callback called")
        depth_queue.put(msg)
        
    def command_callback(self, msg):
        global command_flag
        # rospy.loginfo("Command callback called")
        command_queue.put(msg)
        command_flag = True
        
    def getTime(self):
        return rospy.get_rostime()
    
    # WHEN REALTIME CAPTURE
    def syncImage(self):
        time = self.getTime().to_sec()
        if rgb_queue.qsize() * depth_queue.qsize():
            while(1):
                rgb_msg = rgb_queue.get()
                depth_msg = depth_queue.get()
                
                rgb_time = rgb_msg.header.stamp.to_sec()
                depth_time = depth_msg.header.stamp.to_sec()

                if abs(rgb_time - time) < 0.1 and abs(depth_time - time) < 0.1:
                    break
            return [rgb_msg, depth_msg], True
        else:
            # rospy.loginfo("Image queue EMPTY")
            return [], False
  
    # WHEN ROSBAG
    def getImage(self):
        if rgb_queue.qsize() * depth_queue.qsize():
            while(rgb_queue.qsize() > 1 and depth_queue.qsize() > 1):
                rgb_queue.get()
                depth_queue.get()
            return [rgb_queue.get(), depth_queue.get()], True
        else:
            # rospy.loginfo("Image queue EMPTY")
            return [], False
    
    def run_process(self):
        global command_flag
        while not rospy.is_shutdown():
            if command_flag:
                command_flag = False
                self.process()
    
    def getCameraPose(self):
        pass
    
    def process(self):
        def objectVisualization():
                # Concatenate images horizontally (Visualization DEBUG)
                if object_imgs.shape[0] > 0:
                    concatenated_img = cv2.hconcat([img for img in object_imgs])
                    cv2.imshow("Concatenated Image", concatenated_img)
                cv2.waitKey(1)
        def OpenCV2PIL(opencv_image):
            color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(color_coverted)
            return pil_image
        try:
            status = False
            # object info
            object_bb = np.array([])
            object_cls = np.array([])
            object_seg = np.array([])
            object_imgs = np.empty((0, CROPPED_IMG_SIZE[0], CROPPED_IMG_SIZE[1], 3), dtype=np.uint8)
            user_input = input("Enter a message: ")
            while(not status):
                if MODE == 0:
                    data, status = self.syncImage()
                else:
                    data, status = self.getImage()

            if not status:
                return
            rospy.loginfo("YOLOv8 Semantic Segmentation Processing . . .")
            rgb_image = self.bridge.imgmsg_to_cv2(data[0], 'bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(data[1], '16UC1').astype('float32')
            depth_image /= 65535.0
            # print(np.min(depth_image), np.max(depth_image), np.mean(depth_image))
            
            result = self.yolo_model.predict(source=rgb_image, show=True, show_conf=True, show_labels=True, show_boxes=True)

            # Save the object detection results
            for odr in result:
                object_cls = odr.boxes.cls.numpy().astype(np.int64)
                object_bb = odr.boxes.xyxy.numpy().astype(np.int64)
                object_seg = odr.masks.xy

            obj_num = len(object_cls)
            
            for i in range(obj_num):
                object_img = rgb_image[object_bb[i][1]:object_bb[i][3],object_bb[i][0]:object_bb[i][2]]
                object_img = cv2.resize(object_img, dsize=CROPPED_IMG_SIZE, interpolation=cv2.INTER_LINEAR)
                object_imgs = np.concatenate((object_imgs, [object_img]), axis=0)
                
            objectVisualization()

            rospy.loginfo("CLIP Inference Processing . . .")
            # CLIP inference
            user_text_input = clip.tokenize(user_input).to(self.device)
            with torch.no_grad():
                user_text_features = self.clip_model.encode_text(user_text_input)
                user_text_features /= user_text_features.norm(dim=-1, keepdim=True)

            highest_similarity = 0
            obj_index = 0
            for i in range(obj_num):
                image = OpenCV2PIL(object_imgs[i])
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_input)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    similarity = (image_features @ user_text_features.T).item()
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        obj_index = i

            print(f"Most similar image index: {obj_index}, similarity: {highest_similarity:.4f}")
            
            rospy.loginfo("Object Pose Processing. . .")
            # Camera coordinate to robot base coordinate
            # to determine the end-effector position
            
            # Object center point
            center_x = int(object_bb[obj_index][0] + (object_bb[obj_index][2] - object_bb[obj_index][0]) / 2)
            center_y = int(object_bb[obj_index][1] + (object_bb[obj_index][3] - object_bb[obj_index][1]) / 2)

            # Object center depth
            obj_center_area = depth_image[center_y-5:center_y+5, center_x-5:center_x+5]
            obj_center_depth = np.max(obj_center_area)

            print(f"object center depth is : {obj_center_depth*100:.4f} m")
            
            # Object pose in the Camera coordinate
            camera_pose_w = self.getCameraPose()
            obj_pose_c = np.linalg.inv(K) * np.array([center_x, center_y, 1]).T * obj_center_depth
            
            # Object pose in the Base coordinate
            obj_pose_w
            
            # Send Approach command to manipulator(Moveit)
            obj_pose_w service call
            
            # Decision grasping pose
            
            # Final Approach & Grasp -> Home pose
            
                                
        except Exception as e:
            rospy.logerr(f"Process STOP: {e}")   
            
def signal_handler(sig, frame):
    print('Exiting...')
    cv2.destroyAllWindows()
    torch.cuda.empty_cache()
    rospy.signal_shutdown('Signal received')
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    core = Core()
    rospy.spin()
