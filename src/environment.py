#!/usr/bin/env python

import rospy
import math
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import numpy as np
from typing import List, Tuple

COLLISION_DISTANCE = 0.06
TARGET_DISTANCE = 0.05

class Env():
    """
    Represents the environment for robot navigation.

    This class manages the state of the environment, including robot position,
    goal position, and action space. It also provides methods for resetting the
    environment and obtaining sensor data.

    Attributes:
        posex (float): The x-coordinate of the robot's position.
        posey (float): The y-coordinate of the robot's position.
        goal_x (float): The x-coordinate of the goal position.
        goal_y (float): The y-coordinate of the goal position.
        initial_goal_distance (float): The initial distance to the goal.
        prev_distance (float): The previous distance to the goal.
        heading (float): The angle between robot and goal.
        get_goalbox (bool): Indicates if the goal has been reached.
        init_goal (bool): Indicates if the goal has been initialized.
        action_space (list): The available actions for the robot.
        reset_robot: A service proxy for resetting the robot's position.
        odom: A subscriber for obtaining odometry data.
        move_pub: A publisher for sending movement commands.
        respawn_goal: An instance of the Respawn class for managing goal respawn.
    """
    def __init__(self):
        self.posex = 0
        self.posey = 0
        self.goal_x = 0
        self.goal_y = 0
        self.initial_goal_distance = 0
        self.prev_distance = 0
        self.prev_heading = 0
        self.get_goalbox = False
        self.init_goal = True
        self.action_space = [-1.5, -0.75, 0, 0.75, 1.5]
        self.reset_robot = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.odom = rospy.Subscriber('/odom', Odometry, self.get_pose)
        self.move_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)



    def get_state(self, data: List[float]) -> Tuple[List[float], bool]:
        """
        Calculate the current state of the environment based on sensor data.

        This method computes the current state of the environment using sensor data 
        and internal state variables. It calculates the heading of the robot, checks
        for collisions with obstacles, and determines if the goal has been reached.

        Parameters:
            data: Sensor data containing information about the environment.

        Returns:
            tuple: A tuple containing the current state of the environment and a flag
            indicating whether the episode is done.

        """
        
        heading = self.heading / (2 * math.pi) + 0.5  # normalised
        done = False
        laser_state = self.moving_average_with_zeros(data.ranges)
        laser_state = [1.0 if laser_state[i] > 4 or laser_state[i]== 0.0 else laser_state[i] / 4 for i in range(0, 360, 36)]
        print(f"{laser_state = }")

        if min(laser_state) < COLLISION_DISTANCE:
            done = True

        current_distance = self.get_goal_distance()
        print(f"{current_distance = }")
        print(f"{self.posex =} {self.posey = }")
        if current_distance <  TARGET_DISTANCE:
            self.get_goalbox = True
        
        return (laser_state + [heading, current_distance], done)


    def get_pose(self, data_pos: Odometry) -> None:
        """
        Update the robot's position and calculate its heading based on the given
        odometry data.

        Parameters:
            data_pos: Odometry data containing information about the
            robot's position and orientation.
        """
        self.posex = data_pos.pose.pose.position.x
        self.posey = data_pos.pose.pose.position.y

        orientation = data_pos.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.posey, self.goal_x - self.posex)
        heading = goal_angle - yaw
        if heading > math.pi:
            heading -= 2 * math.pi
        elif heading < -math.pi:
            heading += 2 * math.pi

        self.heading = heading

    def get_goal_distance(self) -> float:
        """
        Calculate the distance between the robot's current position and the goal.

        Returns:
            float: The distance between the robot's current position and the goal.
        """
        return math.hypot(self.goal_x - self.posex, self.goal_y - self.posey)

    def set_reward(self, state: List[float], done: bool, action: int) -> float:
        """
        Calculate the reward based on the current state, action, and episode status.

        Parameters:
            state (List[float]): The current state of the environment.
            done (bool): A flag indicating whether the episode is done.
            action (int): The action taken by the agent.

        Returns:
            float: The reward for the current step.
        """
        current_distance = state[-1]
        current_heading = state[-2]

        if done:
            rospy.loginfo("Collision!!")
            reward = -200
            self.move_pub.publish(Twist())
        elif self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 250
            self.move_pub.publish(Twist())
        else:
            # ## Reward function with only distance reward
            # reward = 30 * (self.prev_distance - current_distance) if current_distance < self.prev_distance else \
            #     60 * (self.prev_distance - current_distance) if current_distance > self.prev_distance else -1
            # self.prev_distance = current_distance
            
            # Reward function with angle ratio + distance ratio + angle penalty
            angle_ratio = current_heading / math.pi
            angle_reward = 2*(0.5-(angle_ratio**2))

            distance_ratio = current_distance / self.initial_goal_distance
            distance_reward = 2*(1 - math.sqrt(2 * distance_ratio)) if distance_ratio>=0 else (1 - math.sqrt(2 * distance_ratio))

            angle_penalty = math.pi * abs(self.prev_heading - current_heading)
            reward = angle_reward + distance_reward - angle_penalty

            #print(f"angle penalty = {angle_penalty}\n {self.prev_heading, current_heading}")
            self.prev_heading = current_heading 
    
        return reward
    

    def step(self, action) -> Tuple[np.array, float, bool, bool]:
        """
        Perform one step in the environment.

        This method executes one step in the environment based on the given action.
        It publishes the action to control the robot's movement, waits for sensor
        data, calculates the current state, computes the reward, and normalizes
        the state before returning it along with the reward, episode completion flag,
        and the status of the goal.

        Parameters:
            action: An integer representing the action to take.

        Returns:
            Tuple: A tuple containing the current state, reward, episode completion flag,
            and the status of the goal.
        """
        data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
        move = Twist()
        move.linear.x =  self.linear_velocity(min(data.ranges))
        #move.angular.z = self.action_space[action] # DQN
        move.angular.z = action # DDPG
        self.move_pub.publish(move)
        state, done = self.get_state(data)
        reward = self.set_reward(state, done, action)


        state[-1] /= 4  # normalising distance
        return np.array(state, dtype=np.float32), reward, done, self.get_goalbox

    def reset(self) -> np.array:
        """
        Reset the environment to its initial state.

        This method resets the environment to its initial state by resetting
        the robot's position, obtaining initial sensor data, setting the initial
        goal position, calculating the initial goal distance, and normalizing
        the distance in the state before returning it.        

        Returns:
            numpy.ndarray: The initial state of the environment.
        """
        self.get_goalbox = False
        data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
        goal  = rospy.wait_for_message("/move_base_simple/goal", PoseStamped, timeout=100)
        if self.init_goal:
            self.goal_x, self.goal_y = goal.pose.position.x, goal.pose.position.y
            rospy.loginfo(f"Goal Position: x={self.goal_x}, y={self.goal_y}")  
            self.init_goal = False

        state, done = self.get_state(data)
        self.initial_goal_distance = self.get_goal_distance()
        #self.prev_distance = self.goal_distance
        self.prev_heading = state[-2]
     


        state[-1] /= 4  # normalising distance
        return np.array(state, dtype=np.float32)
    
    def linear_velocity(self, distance_to_obstacle: float) -> float:
        max_velocity = 0.22  # Maximum velocity when far from obstacles
        min_velocity = 0.05  # Minimum velocity when near obstacles
        max_distance = 0.4  # Distance at which velocity is max_velocity

        velocity = np.sqrt(distance_to_obstacle) * (max_velocity - min_velocity) + min_velocity
        # Clip velocity to ensure it's within the specified range
        velocity = np.clip(velocity, min_velocity, max_velocity)

        return velocity

    def extract_equidistant_elements(self,list, num_elements):
        if num_elements <= 0:
            return []
        
        laser_state_no_zero = [i for i in list if 0 != i]

        # Generate equidistant indices using linspace
        indices = np.linspace(0, len(laser_state_no_zero) - 1, num_elements, dtype=int)

        # Extract elements at the generated indices
        extracted_elements = [laser_state_no_zero[i] for i in indices]

        return extracted_elements
            
    def interpolate_zeros(self, scan_values):
        """
        Perform linear interpolation to replace zero values in a list of scan values.

        Parameters:
            scan_values (list): List of LiDAR scan values.

        Returns:
            interpolated_values (list): List of scan values with zeros replaced by interpolated values.
        """
        interpolated_values = list(scan_values)  # Make a copy of the original list

        # Iterate over the scan values to identify and interpolate zeros
        for i in range(len(interpolated_values)):
            if interpolated_values[i] == 0:
                # Find neighboring non-zero values for interpolation
                left_index = i - 1
                right_index = i + 1

                # Find the closest non-zero value on the left
                while left_index >= 0 and interpolated_values[left_index] == 0:
                    left_index -= 1

                # Find the closest non-zero value on the right
                while right_index < len(interpolated_values) and interpolated_values[right_index] == 0:
                    right_index += 1

                # Perform linear interpolation if valid neighboring values are found
                if left_index >= 0 and right_index < len(interpolated_values):
                    left_value = interpolated_values[left_index]
                    right_value = interpolated_values[right_index]

                    # Linear interpolation formula
                    interpolated_value = left_value + (right_value - left_value) * ((i - left_index) / (right_index - left_index))
                    interpolated_values[i] = interpolated_value
                else:
                    # Edge case: No valid neighboring values found for interpolation
                    # Retain zero value if unable to interpolate
                    interpolated_values[i] = 0

        return interpolated_values

    def moving_average_with_zeros(self, data, window_size=10):
        """
        Compute the moving average of a list of numeric data, replacing zero values with smoothed estimates
        based on neighboring non-zero values within a specified window size.

        Parameters:
            data (list): List of numeric data values to be smoothed.
            window_size (int): Size of the moving average window (number of neighboring elements to consider).

        Returns:
            smoothed_data (list): List of smoothed data values after applying moving average with zero handling.
        """
        smoothed_data = []

        # Iterate over the data to compute moving average with zero handling
        for i in range(len(data)):
            if data[i] == 0:
                # Find neighboring non-zero values within the window
                valid_values = []
                for j in range(max(0, i - window_size // 2), min(len(data), i + window_size // 2 + 1)):
                    if data[j] != 0:
                        valid_values.append(data[j])

                if valid_values:
                    # Calculate moving average using neighboring non-zero values
                    smoothed_value = sum(valid_values) / len(valid_values)
                    smoothed_data.append(smoothed_value)
                else:
                    # If no valid neighbors found, retain the original zero value
                    smoothed_data.append(data[i])
            else:
                # If current value is non-zero, retain the original value
                smoothed_data.append(data[i])

        return smoothed_data


            
                


