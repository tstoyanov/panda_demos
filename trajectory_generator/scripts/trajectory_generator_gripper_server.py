#!/usr/bin/env python

from trajectory_generator.srv import *
import rospy

def handle_trajectory_generetor_gripper(req):
    print "Returning [%s + %s = %s]"%(req.a, req.b, (req.a + req.b))
    return trajectory_generator_gripperResponse(req.a + req.b)

def trajectory_generetor_gripper_server():
    rospy.init_node('trajectory_generetor_gripper_server')
    s = rospy.Service('trajectory_generetor_gripper', trajectory_generator_gripper, handle_trajectory_generetor_gripper)
    print "Ready to add two ints."
    rospy.spin()

if __name__ == "__main__":
    trajectory_generetor_gripper_server()
