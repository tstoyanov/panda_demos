#!/usr/bin/env python

import sys
import rospy
from trajectory_generator.srv import *

def trajectory_generator_gripper_client(x, y):
    rospy.wait_for_service('trajectory_generator_gripper_server')
    try:
        trajectory_generator_gripper = rospy.ServiceProxy('trajectory_generator_gripper', trajectory_generator_gripper)
        resp1 = trajectory_generator_gripper(x, y)
        return resp1.sum
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def usage():
    return "%s [x y]"%sys.argv[0]

if __name__ == "__main__":
    if len(sys.argv) == 3:
        x = int(sys.argv[1])
        y = int(sys.argv[2])
    else:
        print usage()
        sys.exit(1)
    print "Requesting %s+%s"%(x, y)
    print "%s + %s = %s"%(x, y, trajectory_generator_gripper_client(x, y))