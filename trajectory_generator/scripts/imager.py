#!/usr/bin/env python
from __future__ import print_function

import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:

    def __init__(self):
        self.image_pub = rospy.Publisher("my_image_topic",Image)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.callback)

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # =============================================
        grey_img = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)

        grey_img = cv2.medianBlur(grey_img,5)
        # cv_image = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

        stones = cv2.HoughCircles(grey_img,cv2.HOUGH_GRADIENT,1,
                                    param1=50,param2=25,minRadius=5,maxRadius=15,minDist=9)
        if isinstance(stones, np.ndarray):
            stones_nr = len(stones[0])
            stones = np.uint16(np.around(stones))
            for stone in stones[0,:]:
                # draw the outer circle
                center_x = stone[0]
                center_y = stone[1]
                radius = stone[2]
                cv2.circle(cv_image,(center_x,center_y),radius,(0,0,255),2)
                # draw the center of the circle
                cv2.circle(cv_image,(center_x,center_y),2,(255,0,0),3)
                # rospy.loginfo("radius: " + str(radius))


        else:
            stones_nr = 0
        rospy.loginfo("Number of stone(s) found: " + str(stones_nr))
        # =============================================

        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)

def main(args):

    def find_nearest_stone_idx(stones_array, x, y):
        # array = np.asarray(array)
        idx = (np.sqrt(np.square(stones_array[0] - x) + np.square(stones_array[1] - y))).argmin()
        return idx

    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)