#!/usr/bin/env python3
import signal
import clip
import threading
from ultralytics import YOLO
import torch
import rospy
import numpy as np
import cv2
import queue
from cv_bridge import CvBridge
from PIL import Image as PILImage
import open3d as o3d
from std_msgs.msg import Int64
from sensor_msgs.msg import Image
from moveit_msgs.srv import GetPositionFK
import sys
import os
sys.dont_write_bytecode = True
sys.path.append("/root/caktin_ws/src/doosan-robot/common/imp") # get import path : DSR_ROBOT.py 
# for single robot 
ROBOT_ID     = "dsr01"
ROBOT_MODEL  = "a0912"
import DR_init
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
from DSR_ROBOT import *

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
CAMERA_INTRINSIC = [635.5997366807397,635.1073315378363,633.9239491636653,369.02950972096005] # fx, fy, cx, cy
CAMERA_DISTORTION = [-0.055731027995673074, 0.040781967334706, -0.00014895584848947406, 0.0003826882589985031] # k1, k2, p1, p2
K = np.array([[CAMERA_INTRINSIC[0],0,CAMERA_INTRINSIC[2]],
             [0,CAMERA_INTRINSIC[1],CAMERA_INTRINSIC[3]],
             [0,                  0,                  1]])

GRIPPER_OFFSET = 100 #mm

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
        
        self.t1 = threading.Thread(target=self.thread_subscriber)
        self.t1.daemon = True 
        # self.t1.start()
        
        self.msgRobotState_cb_count = 0
        tool_name = get_tool()
        tcp_name = get_tcp()
        set_tcp(tcp_name)
        print("tool name is {0}\ntcp name is : {1}".format(tool_name, tcp_name))
        set_tool(tool_name)
        set_tool_shape("Tool Shape")
        
        self.home_pose = [0.0,-17.5,117.88,0.94,80.98,-92.71]
        # home_pose = [0,0,0,0,0,0]
        status = movej(self.home_pose,time=5)
        print("go to home pose")
        print("robot initialization status : {0}".format(status))
        
        # flange_serial_open(56700)

        rospy.loginfo("CORE node initialization finished !")
    
    def thread_subscriber(self):
        rospy.Subscriber('/'+ROBOT_ID +ROBOT_MODEL+'/state', RobotState, self.msgRobotState_cb)
        # rospy.spin()
        #rospy.spinner(2)    
        
    def msgRobotState_cb(self, msg):
        self.msgRobotState_cb_count += 1

        if (0==(self.msgRobotState_cb_count % 100)): 
            rospy.loginfo("________ ROBOT STATUS ________")
            print("  robot_state           : %d" % (msg.robot_state))
            print("  robot_state_str       : %s" % (msg.robot_state_str))
            print("  actual_mode           : %d" % (msg.actual_mode))
            print("  actual_space          : %d" % (msg.actual_space))
            print("  current_posj          : %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % (msg.current_posj[0],msg.current_posj[1],msg.current_posj[2],msg.current_posj[3],msg.current_posj[4],msg.current_posj[5]))
            print("  current_velj          : %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % (msg.current_velj[0],msg.current_velj[1],msg.current_velj[2],msg.current_velj[3],msg.current_velj[4],msg.current_velj[5]))
            print("  joint_abs             : %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % (msg.joint_abs[0],msg.joint_abs[1],msg.joint_abs[2],msg.joint_abs[3],msg.joint_abs[4],msg.joint_abs[5]))
            print("  joint_err             : %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % (msg.joint_err[0],msg.joint_err[1],msg.joint_err[2],msg.joint_err[3],msg.joint_err[4],msg.joint_err[5]))
            print("  target_posj           : %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % (msg.target_posj[0],msg.target_posj[1],msg.target_posj[2],msg.target_posj[3],msg.target_posj[4],msg.target_posj[5]))
            print("  target_velj           : %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % (msg.target_velj[0],msg.target_velj[1],msg.target_velj[2],msg.target_velj[3],msg.target_velj[4],msg.target_velj[5]))    
            print("  current_posx          : %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % (msg.current_posx[0],msg.current_posx[1],msg.current_posx[2],msg.current_posx[3],msg.current_posx[4],msg.current_posx[5]))
            print("  current_velx          : %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % (msg.current_velx[0],msg.current_velx[1],msg.current_velx[2],msg.current_velx[3],msg.current_velx[4],msg.current_velx[5]))
            print("  task_err              : %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % (msg.task_err[0],msg.task_err[1],msg.task_err[2],msg.task_err[3],msg.task_err[4],msg.task_err[5]))
            print("  solution_space        : %d" % (msg.solution_space))
            sys.stdout.write("  rotation_matrix       : ")
            for i in range(0 , 3):
                sys.stdout.write(  "dim : [%d]"% i)
                sys.stdout.write("  [ ")
                for j in range(0 , 3):
                    sys.stdout.write("%d " % msg.rotation_matrix[i].data[j])
                sys.stdout.write("] ")
            print ##end line
            print("  dynamic_tor           : %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % (msg.dynamic_tor[0],msg.dynamic_tor[1],msg.dynamic_tor[2],msg.dynamic_tor[3],msg.dynamic_tor[4],msg.dynamic_tor[5]))
            print("  actual_jts            : %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % (msg.actual_jts[0],msg.actual_jts[1],msg.actual_jts[2],msg.actual_jts[3],msg.actual_jts[4],msg.actual_jts[5]))
            print("  actual_ejt            : %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % (msg.actual_ejt[0],msg.actual_ejt[1],msg.actual_ejt[2],msg.actual_ejt[3],msg.actual_ejt[4],msg.actual_ejt[5]))
            print("  actual_ett            : %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % (msg.actual_ett[0],msg.actual_ett[1],msg.actual_ett[2],msg.actual_ett[3],msg.actual_ett[4],msg.actual_ett[5]))
            print("  sync_time             : %7.3f" % (msg.sync_time))
            print("  actual_bk             : %d %d %d %d %d %d" % (msg.actual_bk[0],msg.actual_bk[1],msg.actual_bk[2],msg.actual_bk[3],msg.actual_bk[4],msg.actual_bk[5]))
            print("  actual_bt             : %d %d %d %d %d " % (msg.actual_bt[0],msg.actual_bt[1],msg.actual_bt[2],msg.actual_bt[3],msg.actual_bt[4]))
            print("  actual_mc             : %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % (msg.actual_mc[0],msg.actual_mc[1],msg.actual_mc[2],msg.actual_mc[3],msg.actual_mc[4],msg.actual_mc[5]))
            print("  actual_mt             : %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % (msg.actual_mt[0],msg.actual_mt[1],msg.actual_mt[2],msg.actual_mt[3],msg.actual_mt[4],msg.actual_mt[5]))

            #print digital i/o
            sys.stdout.write("  ctrlbox_digital_input : ")
            for i in range(0 , 16):
                sys.stdout.write("%d " % msg.ctrlbox_digital_input[i])
            print ##end line
            sys.stdout.write("  ctrlbox_digital_output: ")
            for i in range(0 , 16):
                sys.stdout.write("%d " % msg.ctrlbox_digital_output[i])
            print
            sys.stdout.write("  flange_digital_input  : ")
            for i in range(0 , 6):
                sys.stdout.write("%d " % msg.flange_digital_input[i])
            print
            sys.stdout.write("  flange_digital_output : ")
            for i in range(0 , 6):
                sys.stdout.write("%d " % msg.flange_digital_output[i])
            print
            #print modbus i/o
            sys.stdout.write("  modbus_state          : " )
            if len(msg.modbus_state) > 0:
                for i in range(0 , len(msg.modbus_state)):
                    sys.stdout.write("[" + msg.modbus_state[i].modbus_symbol)
                    sys.stdout.write(", %d] " % msg.modbus_state[i].modbus_value)
            print

            print("  access_control        : %d" % (msg.access_control))
            print("  homming_completed     : %d" % (msg.homming_completed))
            print("  tp_initialized        : %d" % (msg.tp_initialized))
            print("  mastering_need        : %d" % (msg.mastering_need))
            print("  drl_stopped           : %d" % (msg.drl_stopped))
            print("  disconnected          : %d" % (msg.disconnected))
            
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
            rospy.loginfo("Image queue EMPTY")
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
    
    # change depth image to pointcloud
    def depth2pc(self, depth, K, rgb=None):
        mask = np.where(depth > 0)
        x, y = mask[1], mask[0]
        
        normalized_x = (x.astype(np.float32)-K[2])
        normalized_y = (y.astype(np.float32)-K[3])
        
        world_x = normalized_x * depth[y, x] / K[0]
        world_y = normalized_y * depth[y, x] / K[1]
        world_z = depth[y, x]
        
        if rgb is not None:
            rgb = rgb[y, x]
        
        pc = np.vstack([world_x, world_y, world_z]).T
        return (pc, rgb)
    
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
            set_velx(30,20)  # set global task speed: 30(mm/sec), 20(deg/sec)
            set_accx(60,40)  # set global task accel: 60(mm/sec2), 40(deg/sec2)
            
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
                print("No status")

            if not status:
                return
            rospy.loginfo("YOLOv8 Semantic Segmentation Processing . . .")
            rgb_image = self.bridge.imgmsg_to_cv2(data[0], 'bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(data[1], '16UC1')#.astype('float32')
            # depth_image /= 65535.0
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
            # obj_center_area = depth_image[center_y-5:center_y+5, center_x-5:center_x+5]
            obj_center_area = depth_image[object_bb[obj_index][1]:object_bb[obj_index][3], object_bb[obj_index][0]:object_bb[obj_index][2]]
            obj_rgb_image = rgb_image[object_bb[obj_index][1]:object_bb[obj_index][3], object_bb[obj_index][0]:object_bb[obj_index][2]]
            filtered_area = obj_center_area[obj_center_area > 0]
            filtered_area = mask[obj_index]
            pc, _ = self.depth2pc(obj_center_area, CAMERA_INTRINSIC, obj_rgb_image)
            if pc.size == 0:
                print("Point Cloud is empty!")
            if np.any(np.isnan(pc)) or np.any(np.isinf(pc)):
                print("Point Cloud contains invalid values!")

            pc_o3d = o3d.geometry.PointCloud()
            pc_o3d.points = o3d.utility.Vector3dVector(pc)
            o3d.visualization.draw_geometries([pc_o3d])
            o3d.io.write_point_cloud("output.pcd",pc_o3d)
            print("PC : {0}".format(pc))
            obj_center_depth = np.min(filtered_area)#*0.001
            # print(obj_center_area)
            if obj_center_depth > 1000:
                print("Object depth is not valid.")
                return
            print(f"object center depth is : {obj_center_depth:.4f} mm")
            
            # Object pose in the Camera coordinate
            obj_pix = np.array([center_x * obj_center_depth, center_y * obj_center_depth, obj_center_depth])
            
            # Object pose in the Base coordinate
            t67 = [0, -78.4, 71.458]
    
            T06 = get_current_tool_flange_posx(DR_BASE)
            T67 = t67 + [0,0,0] # joint6 to camera
            
            T07 = htrans(T06, T67)
            
            R07 = eul2rotm(T07[3:])
            T07 = T07[:3]
            
            obj_cam = np.linalg.inv(K) @ obj_pix
            obj_base = R07 @ obj_cam + T07
            # limit of grasp position
            # if obj_base[2] < 100:
            #     obj_base[2] = 100
            
            # 
            pose = get_current_posj()
            # get_pose = [0,0,0,0,0,0]
            for i in range(len(pose)):
                print("joint num : {0}, pose : {1}".format(i+1,pose[i]))
            #     get_pose[i] = pose[i]
            
            # res = fkin(obj_base,DR_BASE)
            # print(res)
            solution_space = 2
            jTime = 5
            target= [obj_base[0], obj_base[1], obj_base[2]+ GRIPPER_OFFSET, 0, 180, 0]
            # print(target)
            # state = movejx(target, sol=solution_space,time=jTime)
            # print(state)
            
            # DEBUGING CODE for real scale 
            # print("Pixel u : {0}, v : {1}".format(center_x,center_y))
            # cv2.circle(rgb_image, (center_x, center_y), 3, (255,0,0))
            # cv2.imshow("pixel position",rgb_image)
            # cv2.waitKey(1)
            # print(obj_cam)
            # print(obj_base)
            status = movej(self.home_pose,time=5)
            
                                
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
