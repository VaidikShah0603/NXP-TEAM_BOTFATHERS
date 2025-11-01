# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
import math
import numpy as np
import cv2
from typing import Optional, Tuple
import threading
import tkinter as tk
from tkinter import ttk
from pyzbar.pyzbar import decode
from sensor_msgs.msg import Joy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.msg import BehaviorTreeLog
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from synapse_msgs.msg import Status
from synapse_msgs.msg import WarehouseShelf
from scipy.ndimage import label, center_of_mass

QOS_PROFILE_DEFAULT = 10
SERVER_WAIT_TIMEOUT_SEC = 5.0
PROGRESS_TABLE_GUI = True

class WindowProgressTable:
    def __init__(self, root, shelf_count):
        self.root = root
        self.root.title("Shelf Objects & QR Link")
        self.root.attributes("-topmost", True)
        self.row_count = 2
        self.col_count = shelf_count
        self.boxes = []
        for row in range(self.row_count):
            row_boxes = []
            for col in range(self.col_count):
                box = tk.Text(root, width=30, height=3, wrap=tk.WORD, borderwidth=0,
                              relief="solid", font=("Arial", 14))
                box.insert(tk.END, "NULL")
                box.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
                row_boxes.append(box)
            self.boxes.append(row_boxes)
        for row in range(self.row_count):
            self.root.grid_rowconfigure(row, weight=1)
        for col in range(self.col_count):
            self.root.grid_columnconfigure(col, weight=1)

    def change_box_color(self, row, col, color):
        self.boxes[row][col].config(bg=color)

    def change_box_text(self, row, col, text):
        self.boxes[row][col].delete(1.0, tk.END)
        self.boxes[row][col].insert(tk.END, text)

box_app = None
def run_gui(shelf_count):
    global box_app
    root = tk.Tk()
    box_app = WindowProgressTable(root, shelf_count)
    root.mainloop()

class WarehouseExplore(Node):
    def __init__(self):
        super().__init__('warehouse_explore')
        self.action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.subscription_pose = self.create_subscription(
            PoseWithCovarianceStamped, '/pose', self.pose_callback, QOS_PROFILE_DEFAULT)
        self.subscription_global_map = self.create_subscription(
            OccupancyGrid, '/global_costmap/costmap', self.global_map_callback, QOS_PROFILE_DEFAULT)
        self.subscription_simple_map = self.create_subscription(
            OccupancyGrid, '/map', self.simple_map_callback, QOS_PROFILE_DEFAULT)
        self.subscription_status = self.create_subscription(
            Status, '/cerebri/out/status', self.cerebri_status_callback, QOS_PROFILE_DEFAULT)
        self.subscription_behavior = self.create_subscription(
            BehaviorTreeLog, '/behavior_tree_log', self.behavior_tree_log_callback, QOS_PROFILE_DEFAULT)
        self.subscription_shelf_objects = self.create_subscription(
            WarehouseShelf, '/shelf_objects', self.shelf_objects_callback, QOS_PROFILE_DEFAULT)
        self.subscription_camera = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.camera_image_callback, QOS_PROFILE_DEFAULT)
        self.publisher_joy = self.create_publisher(Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)
        self.publisher_qr_decode = self.create_publisher(
            CompressedImage, "/debug_images/qr_code", QOS_PROFILE_DEFAULT)
        self.publisher_shelf_data = self.create_publisher(WarehouseShelf, '/shelf_data', QOS_PROFILE_DEFAULT)

        self.declare_parameter('shelf_count', 1)
        self.declare_parameter('initial_angle', 0.0)
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('yaw', 0.0)

        self.shelf_count = self.get_parameter('shelf_count').get_parameter_value().integer_value
        self.initial_angle = math.radians(self.get_parameter('initial_angle').get_parameter_value().double_value)
        self.initial_x = self.get_parameter('x').get_parameter_value().double_value
        self.initial_y = self.get_parameter('y').get_parameter_value().double_value
        self.initial_yaw = self.get_parameter('yaw').get_parameter_value().double_value

        # Robot State
        self.armed = False
        self.logger = self.get_logger()
        self.pose_curr = PoseWithCovarianceStamped()
        self.buggy_pose_x = self.initial_x
        self.buggy_pose_y = self.initial_y
        self.buggy_center = (self.buggy_pose_x, self.buggy_pose_y)
        self.world_center = (0.0, 0.0)
        self.simple_map_curr = None
        self.global_map_curr = None
        self.goal_completed = True
        self.goal_handle_curr = None
        self.cancelling_goal = False
        self.recovery_threshold = 10
        self.xy_goal_tolerance = 0.5
        self._frame_id = "map"
        self.shelf_objects_curr = WarehouseShelf()
        self.qr_code_str = "Empty"
        self.table_row_count = 0
        self.table_col_count = 0
        self.angular_tolerance = math.radians(1.0)
        self.max_step_dist_world_meters = 10.0
        self.min_step_dist_world_meters = 0.5
        self.full_map_explored_count = 0
        self.state = 0  # State machine for sequences
        self.shelf_detected = False
        self.current_shelf = 1
        self.next_angle = self.initial_angle  # Start with initial angle
        self.current_shelf_center = None  # To store current shelf center
        self.shelf_center_x = None  # To store shelf center X coordinate
        self.stored_object_names = []  # To store object names when object_count >= 5
        self.stored_object_counts = []  # To store object counts when object_count >= 5
        self.qr_detected = False  # Flag to ensure single publish per shelf
        self.heuristic_angle = 0.0  # Store the extracted heuristic angle
        self.stored_qr_code = ""  # To store QR code when detected
        self.wait_start_time = None  # To track wait period in state 8
        self.object_detected_in_wait = False  # Flag for object_count >= 3 during wait

    def pose_callback(self, message):
        self.pose_curr = message
        self.buggy_pose_x = message.pose.pose.position.x
        self.buggy_pose_y = message.pose.pose.position.y
        self.buggy_center = (self.buggy_pose_x, self.buggy_pose_y)

    def simple_map_callback(self, message):
        self.simple_map_curr = message
        map_info = self.simple_map_curr.info
        self.world_center = self.get_world_coord_from_map_coord(
            map_info.width / 2, map_info.height / 2, map_info)

    def global_map_callback(self, message):
        self.global_map_curr = message
        if not self.goal_completed:
            return

        if not self.shelf_detected and self.state == 0:
            height, width = message.info.height, message.info.width
            map_array = np.array(message.data).reshape((height, width))
            frontiers = self.get_frontiers_for_space_exploration(map_array)

            map_info = message.info
            if frontiers:
                best_frontier = None
                min_distance_curr = float('inf')
                for fy, fx in frontiers:
                    fx_world, fy_world = self.get_world_coord_from_map_coord(fx, fy, map_info)
                    distance = math.sqrt((fx_world - self.buggy_center[0])**2 + (fy_world - self.buggy_center[1])**2)
                    angle_to_frontier = math.atan2(fy_world - self.buggy_center[1], fx_world - self.buggy_center[0])
                    angle_diff = abs((self.next_angle - angle_to_frontier + math.pi) % (2 * math.pi) - math.pi)
                    if (angle_diff <= self.angular_tolerance and
                        self.min_step_dist_world_meters <= distance <= self.max_step_dist_world_meters):
                        if distance < min_distance_curr:
                            min_distance_curr = distance
                            best_frontier = (fy, fx)

                if best_frontier:
                    fy, fx = best_frontier
                    goal = self.create_goal_from_map_coord(fx, fy, map_info, yaw=self.next_angle)
                    self.send_goal_from_world_pose(goal)
                    self.logger.info(f"Sending goal to frontier at map coords ({fx}, {fy}) aligned with next_angle")
                    self.angular_tolerance = math.radians(15.0)
                    return
                else:
                    self.angular_tolerance += math.radians(7.0)
                    self.max_step_dist_world_meters += 1.0
                    self.min_step_dist_world_meters = max(0.25, self.min_step_dist_world_meters - 0.5)
                    self.logger.info(f"No frontier found in direction, relaxing tolerance to {math.degrees(self.angular_tolerance)}°")
            else:
                self.full_map_explored_count += 1
                self.logger.info(f"No frontiers found; count = {self.full_map_explored_count}")
                if self.full_map_explored_count >= 5:
                    self.logger.warn("Exploration stalled, relaxing constraints further")
                    self.angular_tolerance += math.radians(10.0)
                    self.max_step_dist_world_meters += 2.0
        elif self.state == 1 and self.goal_completed:
            # Turn 100 degrees clockwise
            current_yaw = self.get_current_yaw()
            turn_angle = math.radians(100)  # 100 degrees clockwise
            target_yaw = current_yaw + turn_angle
            goal = self.create_goal_from_world_coord(self.buggy_pose_x, self.buggy_pose_y, target_yaw)
            self.logger.info(f"Executing state 1: Turning 100° clockwise to yaw {math.degrees(target_yaw):.2f}°")
            self.state = 2
            self.send_goal_from_world_pose(goal)
        elif self.state == 2 and self.goal_completed:
            # Move forward 2.5 meters
            current_yaw = self.get_current_yaw()
            target_x = self.buggy_pose_x + 3.0 * math.cos(current_yaw)
            target_y = self.buggy_pose_y + 3.0 * math.sin(current_yaw)
            goal = self.create_goal_from_world_coord(target_x, target_y, current_yaw)
            self.logger.info("Executing state 2: Moving forward 2.5 meters")
            self.state = 3
            self.send_goal_from_world_pose(goal)
        elif self.state == 3 and self.goal_completed:
            # Rotate 150 degrees counterclockwise
            current_yaw = self.get_current_yaw()
            turn_angle = -math.radians(150)  # 150 degrees counterclockwise
            target_yaw = current_yaw + turn_angle
            goal = self.create_goal_from_world_coord(self.buggy_pose_x, self.buggy_pose_y, target_yaw)
            self.logger.info(f"Executing state 3: Rotating 150° counterclockwise to yaw {math.degrees(target_yaw):.2f}°")
            self.state = 4  # Move to QR scanning state
            self.send_goal_from_world_pose(goal)
        elif self.state == 4 and self.goal_completed and self.qr_code_str == "Empty":
            # QR code not detected, stop and log
            self.logger.warn("QR code not detected")
            self.stop_robot()
        # States for QR code sequence (unchanged)
        elif self.state == 5 and self.goal_completed:
            # Turn 85 degrees counterclockwise
            current_yaw = self.get_current_yaw()
            turn_angle = -math.radians(85)  # 85 degrees counterclockwise
            target_yaw = current_yaw + turn_angle
            goal = self.create_goal_from_world_coord(self.buggy_pose_x, self.buggy_pose_y, target_yaw)
            self.logger.info(f"Executing state 5: Turning 85° counterclockwise to yaw {math.degrees(target_yaw):.2f}°")
            self.state = 6
            self.send_goal_from_world_pose(goal)
        elif self.state == 6 and self.goal_completed:
            # Move forward 2.75 meters
            current_yaw = self.get_current_yaw()
            target_x = self.buggy_pose_x + 2.65 * math.cos(current_yaw)
            target_y = self.buggy_pose_y + 2.65 * math.sin(current_yaw)
            goal = self.create_goal_from_world_coord(target_x, target_y, current_yaw)
            self.logger.info("Executing state 6: Moving forward 2.75 meters")
            self.state = 7
            self.send_goal_from_world_pose(goal)
        elif self.state == 7 and self.goal_completed:
            # Turn 150 degrees clockwise
            current_yaw = self.get_current_yaw()
            turn_angle = math.radians(150)  # 150 degrees clockwise
            target_yaw = current_yaw + turn_angle
            goal = self.create_goal_from_world_coord(self.buggy_pose_x, self.buggy_pose_y, target_yaw)
            self.logger.info(f"Executing state 7: Turning 150° clockwise to yaw {math.degrees(target_yaw):.2f}°")
            self.state = 8
            self.wait_start_time = self.get_clock().now().nanoseconds / 1e9  # Start 10-second timer
            self.object_detected_in_wait = False
            self.send_goal_from_world_pose(goal)
        elif self.state == 8 and self.goal_completed:
            # Wait for 10 seconds to check object_count >= 2
            current_time = self.get_clock().now().nanoseconds / 1e9
            if current_time - self.wait_start_time >= 10.0:
                if not self.object_detected_in_wait:
                    self.logger.info("10 seconds elapsed without detecting object_count >= 2, aligning to heuristic angle")
                    self.align_to_heuristic_angle()
                # Reset state regardless of object detection
                self.state = 0
                self.wait_start_time = None
        # New state for backward movement
        elif self.state == 9 and self.goal_completed:
            # Move backward 0.7 meters
            current_yaw = self.get_current_yaw()
            target_x = self.buggy_pose_x - 0.7 * math.cos(current_yaw)
            target_y = self.buggy_pose_y - 0.7 * math.sin(current_yaw)
            goal = self.create_goal_from_world_coord(target_x, target_y, current_yaw)
            self.logger.info("Executing state 9: Moving backward 0.7 meters")
            self.state = 1
            self.send_goal_from_world_pose(goal)

    def align_to_heuristic_angle(self):
        if self.heuristic_angle != 0.0:
            target_angle = self.calculate_angle_to_next_shelf(self.current_shelf_center, self.heuristic_angle)
            goal = self.create_goal_from_world_coord(self.buggy_pose_x, self.buggy_pose_y, target_angle)
            self.logger.info(f"Aligning to heuristic angle {math.degrees(target_angle):.2f}° for next shelf")
            self.next_angle = target_angle
            self.qr_detected = False  # Reset for future QR detections
            self.shelf_detected = False
            self.shelf_center_x = None  # Reset shelf center X
            self.send_goal_from_world_pose(goal)
        else:
            self.stop_robot()

    def get_frontiers_for_space_exploration(self, map_array):
        frontiers = []
        for y in range(1, map_array.shape[0] - 1):
            for x in range(1, map_array.shape[1] - 1):
                if map_array[y, x] == -1:  # Unknown space
                    neighbors_complete = [
                        (y, x - 1), (y, x + 1), (y - 1, x), (y + 1, x),
                        (y - 1, x - 1), (y + 1, x - 1), (y - 1, x + 1), (y + 1, x + 1)
                    ]
                    near_obstacle = any(map_array[ny, nx] > 0 for ny, nx in neighbors_complete)
                    if near_obstacle:
                        continue
                    neighbors_cardinal = [(y, x - 1), (y, x + 1), (y - 1, x), (y + 1, x)]
                    for ny, nx in neighbors_cardinal:
                        if map_array[ny, nx] == 0:  # Free space
                            frontiers.append((ny, nx))
                            break
        return frontiers

    def publish_debug_image(self, publisher, image):
        if image.size:
            message = CompressedImage()
            _, encoded_data = cv2.imencode('.jpg', image)
            message.format = "jpeg"
            message.data = encoded_data.tobytes()
            publisher.publish(message)

    def camera_image_callback(self, message):
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        resized_image = cv2.resize(image, (640, 480))
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        qr_result = decode(gray)
        if qr_result and (self.state == 0 or self.state == 4):
            qr_data = qr_result[0].data.decode('utf-8').strip()
            self.qr_code_str = qr_data
            self.get_logger().info(f"[QR DETECTED] {self.qr_code_str}")
            (x, y, w, h) = qr_result[0].rect
            cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(resized_image, self.qr_code_str, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Extract heuristic angle and shelf ID
            shelf_id_str = self.qr_code_str.split('_')[0]
            heuristic_angle_str = self.qr_code_str[2:7]
            random_string = self.qr_code_str[7:]
            self.current_shelf = int(shelf_id_str)
            self.heuristic_angle = math.radians(float(heuristic_angle_str))

            if self.state == 0 and not self.qr_detected:
                total_objects = sum(self.shelf_objects_curr.object_count)
                if total_objects < 5:
                    # Case: QR detected in state 0 with object_count < 5
                    self.cancel_current_goal()  # Stop the robot
                    self.shelf_objects_curr.qr_decoded = self.qr_code_str
                    self.shelf_objects_curr.object_name = []
                    self.shelf_objects_curr.object_count = []
                    self.publisher_shelf_data.publish(self.shelf_objects_curr)
                    self.logger.info(f"Published to shelf_data (QR only): Shelf {self.current_shelf}, QR {self.qr_code_str}")
                    self.qr_detected = True
                    self.stored_qr_code = self.qr_code_str
                    # Start new sequence: Turn 85° counterclockwise
                    current_yaw = self.get_current_yaw()
                    turn_angle = -math.radians(85)  # 85 degrees counterclockwise
                    target_yaw = current_yaw + turn_angle
                    goal = self.create_goal_from_world_coord(self.buggy_pose_x, self.buggy_pose_y, target_yaw)
                    self.logger.info(f"Starting QR sequence: Turning 85° counterclockwise to yaw {math.degrees(target_yaw):.2f}°")
                    self.state = 5
                    self.send_goal_from_world_pose(goal)
                else:
                    # If object_count >= 5, let shelf_objects_callback handle it
                    pass
            elif self.state == 4 and not self.qr_detected:
                # Existing case: QR detected in state 4
                self.shelf_objects_curr.object_name = self.stored_object_names
                self.shelf_objects_curr.object_count = self.stored_object_counts
                self.shelf_objects_curr.qr_decoded = self.qr_code_str
                self.publisher_shelf_data.publish(self.shelf_objects_curr)
                self.logger.info(f"Published to shelf_data (with QR): Shelf {self.current_shelf}, Objects {self.stored_object_names}, Counts {self.stored_object_counts}, QR {self.qr_code_str}")
                self.qr_detected = True

                if self.heuristic_angle == 0.0:
                    self.stop_robot()
                else:
                    if self.current_shelf_center:
                        target_angle = self.calculate_angle_to_next_shelf(self.current_shelf_center, self.heuristic_angle)
                        goal = self.create_goal_from_world_coord(self.buggy_pose_x, self.buggy_pose_y, target_angle)
                        self.logger.info(f"Aligning to heuristic angle {math.degrees(target_angle):.2f}° for next shelf")
                        self.next_angle = target_angle
                        self.state = 0
                        self.shelf_detected = False
                        self.shelf_center_x = None  # Reset shelf center X
                        self.send_goal_from_world_pose(goal)
        self.publish_debug_image(self.publisher_qr_decode, resized_image)

    def cerebri_status_callback(self, message):
        if message.mode == 3 and message.arming == 2:
            self.armed = True
        else:
            msg = Joy()
            msg.buttons = [0, 1, 0, 0, 0, 0, 0, 1]
            msg.axes = [0.0, 0.0, 0.0, 0.0]
            self.publisher_joy.publish(msg)

    def behavior_tree_log_callback(self, message):
        for event in message.event_log:
            if (event.node_name == "FollowPath" and
                event.previous_status == "SUCCESS" and
                event.current_status == "IDLE"):
                pass

    def shelf_objects_callback(self, message):
        self.shelf_objects_curr = message
        total_objects = sum(message.object_count)
        
        # Existing case: object_count >= 5 in state 0, only if QR not detected
        if total_objects >= 5 and not self.shelf_detected and self.state == 0 and not self.qr_detected:
            self.shelf_detected = True
            self.qr_detected = False  # Reset QR detection flag for new shelf
            self.logger.info(f"Object count >= 5 ({total_objects}), stopping at shelf {self.current_shelf}")
            self.cancel_current_goal()
            # Store robot's current coordinates
            robot_x = self.buggy_pose_x
            robot_y = self.buggy_pose_y
            self.logger.info(f"Robot stopped at coordinates: ({robot_x:.2f}, {robot_y:.2f})")
            # Store and publish object data
            self.stored_object_names = message.object_name[:]
            self.stored_object_counts = message.object_count[:]
            self.shelf_objects_curr.object_name = self.stored_object_names
            self.shelf_objects_curr.object_count = self.stored_object_counts
            self.shelf_objects_curr.qr_decoded = ""  # No QR data yet
            self.publisher_shelf_data.publish(self.shelf_objects_curr)
            self.logger.info(f"Published to shelf_data (initial): Shelf {self.current_shelf}, Objects {self.stored_object_names}, Counts {self.stored_object_counts}")
            if PROGRESS_TABLE_GUI and self.table_col_count < self.shelf_count:
                obj_str = ""
                for name, count in zip(message.object_name, message.object_count):
                    obj_str += f"{name}: {count}\n"
                box_app.change_box_text(self.table_row_count, self.table_col_count, obj_str)
                box_app.change_box_color(self.table_row_count, self.table_col_count, "cyan")
                self.table_row_count += 1
                box_app.change_box_text(self.table_row_count, self.table_col_count, message.qr_decoded or self.qr_code_str)
                box_app.change_box_color(self.table_row_count, self.table_col_count, "yellow")
                self.table_row_count = 0
                self.table_col_count += 1
            elif self.table_col_count >= self.shelf_count:
                self.logger.warn("GUI update skipped: table_col_count exceeds shelf_count")
            # Calculate current shelf center using SLAM map
            if self.global_map_curr:
                map_array = np.array(self.global_map_curr.data).reshape(
                    (self.global_map_curr.info.height, self.global_map_curr.info.width))
                labeled_array, num_features = label(map_array > 0)  # Label obstacles
                if num_features > 0:
                    sizes = np.bincount(labeled_array.ravel())[1:]  # Exclude background
                    largest_label = np.argmax(sizes) + 1
                    shelf_mask = (labeled_array == largest_label)
                    coords = np.column_stack(np.where(shelf_mask))
                    if len(coords) > 0:
                        self.current_shelf_center = self.get_world_coord_from_map_coord(
                            int(coords[:, 1].mean()), int(coords[:, 0].mean()), self.global_map_curr.info)
                        self.shelf_center_x = self.current_shelf_center[0]
                        self.logger.info(f"Shelf {self.current_shelf} center at: {self.current_shelf_center}")
                        # Compare robot's X coordinate with shelf's center X
                        if robot_x > self.shelf_center_x:
                            self.logger.info("Robot X > Shelf Center X, starting sequence: 100° clockwise")
                            self.state = 1  # Start with 100° clockwise
                        else:
                            self.logger.info("Robot X < Shelf Center X, starting sequence: Move backward 0.7m")
                            self.state = 9  # Start with backward 0.7m
        # New case: object_count >= 2 in state 8 after QR detection
        elif total_objects >= 2 and self.state == 8 and not self.object_detected_in_wait and not self.shelf_detected:
            self.shelf_detected = True
            self.object_detected_in_wait = True
            self.qr_detected = True  # Ensure QR flag remains set to block case 1
            self.logger.info(f"Object count >= 2 ({total_objects}) detected in state 8, publishing complete shelf data")
            # Publish complete message with object_count, object_name, and stored QR data
            self.shelf_objects_curr.object_name = message.object_name[:]
            self.shelf_objects_curr.object_count = message.object_count[:]
            self.shelf_objects_curr.qr_decoded = self.stored_qr_code
            self.publisher_shelf_data.publish(self.shelf_objects_curr)
            self.logger.info(f"Published to shelf_data (complete): Shelf {self.current_shelf}, Objects {message.object_name}, Counts {message.object_count}, QR {self.stored_qr_code}")
            if PROGRESS_TABLE_GUI and self.table_col_count < self.shelf_count:
                obj_str = ""
                for name, count in zip(message.object_name, message.object_count):
                    obj_str += f"{name}: {count}\n"
                box_app.change_box_text(self.table_row_count, self.table_col_count, obj_str)
                box_app.change_box_color(self.table_row_count, self.table_col_count, "cyan")
                self.table_row_count += 1
                box_app.change_box_text(self.table_row_count, self.table_col_count, self.stored_qr_code)
                box_app.change_box_color(self.table_row_count, self.table_col_count, "yellow")
                self.table_row_count = 0
                self.table_col_count += 1
            self.align_to_heuristic_angle()
        elif total_objects < 5 and not self.shelf_detected and self.state == 0:
            # Do not publish or store if object_count < 5 in state 0
            pass
        elif total_objects < 2 and self.state == 8:
            # Wait for more objects during 10-second period
            pass

    def get_current_yaw(self):
        if self.pose_curr:
            q = self.pose_curr.pose.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            return math.atan2(siny_cosp, cosy_cosp)
        return self.initial_yaw

    def rover_move_manual_mode(self, speed, turn):
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        msg.axes = [0.0, speed, 0.0, turn]
        self.publisher_joy.publish(msg)

    def stop_robot(self):
        msg = Joy()
        msg.buttons = [0, 1, 0, 0, 0, 0, 0, 1]
        msg.axes = [0.0, 0.0, 0.0, 0.0]
        self.publisher_joy.publish(msg)
        self.cancel_current_goal()
        self.logger.info("Robot stopped")

    def cancel_goal_callback(self, future):
        cancel_result = future.result()
        if cancel_result:
            self.logger.info("Goal cancellation successful.")
            self.cancelling_goal = False
        else:
            self.logger.error("Goal cancellation failed.")
            self.cancelling_goal = False

    def cancel_current_goal(self):
        if self.goal_handle_curr is not None and not self.cancelling_goal:
            self.cancelling_goal = True
            self.logger.info("Requesting cancellation of current goal...")
            cancel_future = self.action_client._cancel_goal_async(self.goal_handle_curr)
            cancel_future.add_done_callback(self.cancel_goal_callback)

    def goal_result_callback(self, future):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.logger.info("Goal completed successfully!")
        elif status == GoalStatus.STATUS_CANCELED:
            self.logger.info("Goal was canceled, likely due to object count >= 5 or QR detection")
        else:
            self.logger.warn(f"Goal failed with status: {status}")
        self.goal_completed = True
        self.goal_handle_curr = None

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.logger.warn('Goal rejected :(')
            self.goal_completed = True
            self.goal_handle_curr = None
        else:
            self.logger.info('Goal accepted :)')
            self.goal_completed = False
            self.goal_handle_curr = goal_handle
            get_result_future = goal_handle.get_result_async()
            get_result_future.add_done_callback(self.goal_result_callback)

    def goal_feedback_callback(self, msg):
        distance_remaining = msg.feedback.distance_remaining
        number_of_recoveries = msg.feedback.number_of_recoveries
        navigation_time = msg.feedback.navigation_time.sec
        estimated_time_remaining = msg.feedback.estimated_time_remaining.sec
        self.logger.debug(f"Recoveries: {number_of_recoveries}, "
                         f"Navigation time: {navigation_time}s, "
                         f"Distance remaining: {distance_remaining:.2f}, "
                         f"Estimated time remaining: {estimated_time_remaining}s")
        if number_of_recoveries > self.recovery_threshold and not self.cancelling_goal:
            self.logger.warn(f"Cancelling. Recoveries = {number_of_recoveries}.")
            self.cancel_current_goal()

    def send_goal_from_world_pose(self, goal_pose):
        if not self.goal_completed or self.goal_handle_curr is not None:
            return False
        self.goal_completed = False
        goal = NavigateToPose.Goal()
        goal.pose = goal_pose
        if not self.action_client.wait_for_server(timeout_sec=SERVER_WAIT_TIMEOUT_SEC):
            self.logger.error('NavigateToPose action server not available!')
            return False
        goal_future = self.action_client.send_goal_async(goal, self.goal_feedback_callback)
        goal_future.add_done_callback(self.goal_response_callback)
        return True

    def _get_map_conversion_info(self, map_info) -> Optional[Tuple[float, float, float]]:
        if map_info:
            origin = map_info.origin
            resolution = map_info.resolution
            return resolution, origin.position.x, origin.position.y
        else:
            return None

    def get_world_coord_from_map_coord(self, map_x: int, map_y: int, map_info) -> Tuple[float, float]:
        if map_info:
            resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
            world_x = (map_x + 0.5) * resolution + origin_x
            world_y = (map_y + 0.5) * resolution + origin_y
            return (world_x, world_y)
        else:
            return (0.0, 0.0)

    def get_map_coord_from_world_coord(self, world_x: float, world_y: float, map_info) -> Tuple[int, int]:
        if map_info:
            resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
            map_x = int((world_x - origin_x) / resolution)
            map_y = int((world_y - origin_y) / resolution)
            return (map_x, map_y)
        else:
            return (0, 0)

    def _create_quaternion_from_yaw(self, yaw: float) -> Quaternion:
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = sy
        q.w = cy
        return q

    def create_yaw_from_vector(self, dest_x: float, dest_y: float, source_x: float, source_y: float) -> float:
        delta_x = dest_x - source_x
        delta_y = dest_y - source_y
        yaw = math.atan2(delta_y, delta_x)
        return yaw

    def create_goal_from_world_coord(self, world_x: float, world_y: float, yaw: Optional[float] = None) -> PoseStamped:
        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = self._frame_id
        goal_pose.pose.position.x = world_x
        goal_pose.pose.position.y = world_y
        goal_pose.pose.position.z = 0.0
        if yaw is None and self.pose_curr is not None:
            source_x = self.pose_curr.pose.pose.position.x
            source_y = self.pose_curr.pose.pose.position.y
            yaw = self.create_yaw_from_vector(world_x, world_y, source_x, source_y)
        elif yaw is None:
            yaw = 0.0
        goal_pose.pose.orientation = self._create_quaternion_from_yaw(yaw)
        self.logger.info(f"Goal created: ({world_x:.2f}, {world_y:.2f}, yaw={math.degrees(yaw):.2f}°)")
        return goal_pose

    def create_goal_from_map_coord(self, map_x: int, map_y: int, map_info, yaw: Optional[float] = None) -> PoseStamped:
        world_x, world_y = self.get_world_coord_from_map_coord(map_x, map_y, map_info)
        return self.create_goal_from_world_coord(world_x, world_y, yaw)

    def calculate_angle_to_next_shelf(self, current_center, heuristic_angle):
        # Calculate angle from current shelf center to next shelf center
        return heuristic_angle

def main(args=None):
    rclpy.init(args=args)
    warehouse_explore = WarehouseExplore()
    if PROGRESS_TABLE_GUI:
        gui_thread = threading.Thread(target=run_gui, args=(warehouse_explore.shelf_count,))
        gui_thread.start()
    try:
        rclpy.spin(warehouse_explore)
    except Exception as e:
        warehouse_explore.get_logger().error(f"Node crashed with exception: {str(e)}")
    finally:
        warehouse_explore.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

