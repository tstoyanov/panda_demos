#!/usr/bin/env python

from trajectory_generator.srv import *
import rospy

def handle_trajectory_generator_gripper(req):
    print "Returning [%s + %s = %s]"%(req.a, req.b, (req.a + req.b))
    return trajectory_generator_gripperResponse(req.a + req.b)

def trajectory_generator_gripper_server():
    rospy.init_node('trajectory_generator_gripper_server')
    s = rospy.Service('trajectory_generator_gripper_server', trajectory_generator_gripper, handle_trajectory_generator_gripper)
    print "Ready to add two ints."
    rospy.spin()

if __name__ == "__main__":
    trajectory_generator_gripper_server()
