#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped

def goal_callback(msg):
    goal_position = msg.pose.position
    goal_orientation = msg.pose.orientation
    rospy.loginfo(f"Goal Position: x={goal_position.x}, y={goal_position.y}, z={goal_position.z}")
    rospy.loginfo("Goal Orientation: x={}, y={}, z={}, w={}".format(goal_orientation.x, goal_orientation.y, goal_orientation.z, goal_orientation.w))

def listener():
    rospy.init_node('goal_listener', anonymous=True)
    rospy.Subscriber("/move_base_simple/goal", PoseStamped, goal_callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
