import rospy

from PyKDL import Tree
from PyKDL import Chain
from PyKDL import Joint
from PyKDL import Jacobian
from PyKDL import ChainJntToJacSolver

from urdf_parser_py.urdf import URDF

import argparse

parser = argparse.ArgumentParser(description='resolved_rate_motion_control')
parser.add_argument('-s', '--sim', nargs='?', const=True, default=False,
                    help='Says whether we are in simulation or not')

args = parser.parse_args()

def main():
    if args.sim:
        param_name = "/robot_description"
    else:
        param_name = "/panda/robot_description"

    print ("param_name = ", param_name)
    # robot = URDF.from_parameter_server(param_name).get_chain("world", "panda_hand")
    robot_description = rospy.get_param(param_name)
    robot = URDF.from_xml_string(robot_description)

    print ("robot: ")
    # print (robot)
    print ("robot_description: ")
    print (robot_description)

if __name__ == '__main__':
    main()