#!/usr/bin/env python
import rospy
import rospkg
import kdl_parser_py.urdf as urdf

import argparse

from PyKDL import Tree
from PyKDL import Chain
from PyKDL import Joint
from PyKDL import Jacobian
from PyKDL import JntArray
from PyKDL import ChainJntToJacSolver

from urdf_parser_py.urdf import URDF

import xml.etree.ElementTree


parser = argparse.ArgumentParser(description='resolved_rate_motion_control')
parser.add_argument('-s', '--sim', nargs='?', const=True, default=False,
                    help='Says whether we are in simulation or not')

args = parser.parse_args()

rospack = rospkg.RosPack()

def main():
    package_path = rospack.get_path("trajectory_generator")
    if args.sim:
        param_name = "/robot_description"
    else:
        param_name = "/panda/robot_description"

    # robot_description = rospy.get_param(param_name)

    # et = xml.etree.ElementTree.fromstring(robot_description)
    # xml_file = open(package_path+"/../../../robot_description.xml", "a+")
    # xml_str = xml.etree.ElementTree.tostring(et).decode()
    # xml_file.write(xml_str)

    # robot = URDF.from_parameter_server(param_name).get_chain("world", "panda_hand")
    # robot = URDF.from_xml_string(robot_description)
    # robot = URDF.from_parameter_server(param_name)
    # robot = URDF.from_xml_file(package_path+"/robot_description.xml")
    # for i in range(len(robot.joints)):
    #     print (robot.joints[i].name)
    #     print ("")
    # print ("len(robot.joints)")
    # print(len(robot.joints))
    # for i in range(len(robot.links)):
    #     print (robot.links[i].name)
    #     print ("")
    # print ("robot.links")
    # print(len(robot.links))
    # for i in range(len(list(robot.child_map))):
    #     print (list(robot.child_map)[i])
    # my_chain = robot.get_chain("world", "panda_hand")
    
    # ret, my_tree = urdf.treeFromUrdfModel(URDF.from_parameter_server(param_name))
    ret, my_tree = urdf.treeFromUrdfModel(URDF.from_xml_file(package_path+"/robot_description.xml"))
    my_chain = my_tree.getChain("world", "panda_hand")
    nr_of_joints = my_chain.getNrOfJoints()

    jnt_array = JntArray(nr_of_joints)
    jacobian = Jacobian(nr_of_joints)
    # ChainJntToJacSolver.JntToJac

    # print ("robot: ")
    # print (robot)
    print ("my_chain: ")
    print (my_chain)
    print ("jnt_array: ")
    print (jnt_array)
    print ("jacobian: ")
    print (jacobian)
    # print ("robot_description: ")
    # print (robot_description)

if __name__ == '__main__':
    main()