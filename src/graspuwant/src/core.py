#!/usr/bin/env python3
import signal
import sys
import threading
from ultralytics import YOLO
import torch
import rospy
import numpy as np
import cv2
import queue
from cv_bridge import CvBridge

from std_msgs.msg import Int64
from sensor_msgs.msg import Image

YOLO_MODEDL_PATH = '/root/yolov8x-seg.pt' # COCO pretrained model for segmentation

command_flag = False

# mode 0 : real-time
# mode 1 : rosbag
MODE = 1
CROPPED_IMG_SIZE = (320,320)

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
        self.model = YOLO(YOLO_MODEDL_PATH)
        self.bridge = CvBridge()
        
        # process threading
        self.process_thread = threading.Thread(target=self.run_process)
        self.process_thread.daemon = True
        self.process_thread.start()
    
    # callback function        
    def rgb_callback(self, msg):
        # rospy.loginfo("RGB callback called")
        rgb_queue.put(msg)
        
    def depth_callback(self, msg):
        # rospy.loginfo("Depth callback called")
        depth_queue.put(msg)
        
    def command_callback(self, msg):
        global command_flag
        rospy.loginfo("Command callback called")
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
            
    def process(self):
        try:
            status = False
            # object info
            object_bb = np.array([])
            object_cls = np.array([])
            object_seg = np.array([])
            object_imgs = np.empty((0, CROPPED_IMG_SIZE[0], CROPPED_IMG_SIZE[1], 3), dtype=np.uint8)
            while(not status):
                if MODE == 0:
                    data, status = self.syncImage()
                else:
                    data, status = self.getImage()

            if not status:
                return
            rospy.loginfo("process enter")
            rgb_image = self.bridge.imgmsg_to_cv2(data[0], 'bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(data[1], '32FC1')

            result = self.model.predict(source=rgb_image, show=True, show_conf=True, show_labels=True, show_boxes=True)

            # Save the object detection results
            for odr in result:
                object_cls = odr.boxes.cls.numpy().astype(np.int64)
                object_bb = odr.boxes.xyxy.numpy().astype(np.int64)
                object_seg = odr.masks.xy

            # Crop the image for CLIP input
            # right : +x, down : +y
            for i in range(len(object_cls)):
                object_img = rgb_image[object_bb[i][1]:object_bb[i][3],object_bb[i][0]:object_bb[i][2]]
                object_img = cv2.resize(object_img, dsize=CROPPED_IMG_SIZE, interpolation=cv2.INTER_LINEAR)
                object_imgs = np.concatenate((object_imgs, [object_img]), axis=0)

            # Concatenate images horizontally (Visualization DEBUG)
            if object_imgs.shape[0] > 0:
                concatenated_img = cv2.hconcat([img for img in object_imgs])
                cv2.imshow("Concatenated Image", concatenated_img)
            
            cv2.waitKey(1)
            
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
