#!/usr/bin/env python

import rospy
from std_msgs.msg import String

class robo_env:
	def __init__(self, pub):
		self.pub = pub
		self.storage = ""
		self.storage_shit = ""
		self.rate = rospy.Rate(10)

	def save_shit(self, data):
		#rospy.loginfo(data.data)
		self.storage = data.data
		self.rate.sleep()

	def send_action(self):
		#while not rospy.is_shutdown():
		action = self.storage_shit
		rospy.loginfo(action)
		self.pub.publish(action)
		self.rate.sleep()

	def mainfan(self):
		#while not rospy.is_shutdown():
		rate = rospy.Rate(1)
		action = self.storage
		action = "eh" + action
		#rospy.loginfo(action)
		self.storage_shit = action
		self.send_action()
		rate.sleep()

def main():
	rospy.init_node('DRL_traffic')
	pub = rospy.Publisher('something', String, queue_size=10)
	robo = robo_env(pub)
	
	rospy.Subscriber("chatter", String, robo.save_shit)
	while not rospy.is_shutdown():
		robo.mainfan()
		#robo.send_action()	

	rospy.spin()
	

if __name__ == '__main__':
	main()
