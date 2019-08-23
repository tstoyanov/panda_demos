#!/usr/bin/env python
from __future__ import print_function

import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

min_radius = 8
max_radius = 14
min_dist = min_radius*2

class stone_class:
    def __init__(self, x=0, y=0, r=0, color=0):
        self.sample_counter = 0
        self.x = x
        self.y = y
        self.r = r
        self.color = color #BGR
        self.color_name = None

    def set_color_name(self, color_name):
        self.color_name = color_name
    
    def get_color(self):
        return self.color
    
    def get_color_name(self):
        if self.color_name is None:
            return "None"
        else:
            return self.color_name

    
    def get_center(self):
        return (self.x, self.y)
    
    def refine_stone(self, x, y, r, color):
        self.x = (self.x + x)/2
        self.y = (self.y + y)/2
        # self.x = (self.sample_counter*self.x + x)/(self.sample_counter+1)
        # self.y = (self.sample_counter*self.y + y)/(self.sample_counter+1)
        self.r = (self.sample_counter*self.r+r)/(self.sample_counter+1)
        self.color = (self.sample_counter*self.color+color)/(self.sample_counter+1)
        if self.sample_counter < 10:
            self.sample_counter += 1
        print("refined stone:")
        print("\tx = ", str(self.x))
        print("\ty = ", str(self.y))
        print("\tr = ", str(self.r))
        print("\tcolor = ", str(self.color))
        print("\tsample_counter = ", str(self.sample_counter))


class stone_organizer:
    def __init__(self):
        self.stones = []
        self.color_boundaries = np.array([
                # ([min], [max])
                ("orange", [0, 0, 150], [49, 255, 250]),  # orange
                ("blue", [50, 0, 0], [150, 255, 149])   # blue
            ],
            dtype=[('name', 'S10'),('lower', '<f8', (3)), ('upper', '<f8', (3))])

    def clear(self):
        self.stones = []

    def add_sample(self, x, y, r, color):
        if len(self.stones) != 0:
            x_array = np.asarray(list((s.x for s in self.stones)))
            y_array = np.asarray(list((s.y for s in self.stones)))
            distances = np.sqrt(np.square(x_array - x) + np.square(y_array - y))
            if np.amin(distances) < min_dist:
                self.stones[distances.argmin()].refine_stone(x, y, r, color)
                return True

        self.stones.append(stone_class(x, y, r, color))
        return True
    
    def end_search(self):
        for stone in self.stones:
            color = stone.get_color()
            color_index = np.argwhere(np.all((color >= self.color_boundaries["lower"]) & (color <= self.color_boundaries["upper"]), axis=1))
            if len(color_index) != 0:
                stone.set_color_name(self.color_boundaries["name"][color_index.reshape(1)[0]])

class image_converter:

    def __init__(self, iteration_treshold=10):
        self.image_pub = rospy.Publisher("my_image_topic",Image)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.callback)
        self.so = stone_organizer()
        self.do_search = False
        self.iteration = 0
        self.iteration_treshold = iteration_treshold

    def clear(self):
        self.so.clear()
        self.iteration = 0
    
    def search(self):
        self.clear()
        self.do_search = True

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # =============================================
        grey_img = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)

        grey_img = cv2.medianBlur(grey_img,5)
        drawable_image = np.copy(cv_image)

        if self.do_search:
            if self.iteration < self.iteration_treshold:
                self.iteration += 1
                stones = cv2.HoughCircles(grey_img,cv2.HOUGH_GRADIENT,1,
                                            param1=50,param2=25,minRadius=min_radius,maxRadius=max_radius,minDist=min_dist)
                if isinstance(stones, np.ndarray):
                    stones_nr = len(stones[0])
                    stones = np.uint16(np.around(stones))
                    print("The number of stones found in iteration " + str(self.iteration) + " is: " + str(stones_nr))
                    for stone in stones[0,:]:
                        # draw the outer circle
                        center_x = stone[0]
                        center_y = stone[1]
                        radius = stone[2]

                        color_matrix = cv_image[center_y-(int)(radius/2):center_y+(int)(radius/2),center_x-(int)(radius/2):center_x+(int)(radius/2)]
                        color = np.mean(color_matrix, axis=(0, 1))
                        self.so.add_sample(center_x, center_y, radius, color)
                        
                        cv2.circle(drawable_image,(center_x,center_y),radius,(0,0,255),2)
                        # draw the center of the circle
                        # cv2.circle(colored_img,(center_x,center_y),2,(255,0,0),3)


                else:
                    stones_nr = 0
            else:
                self.do_search = False
                self.so.end_search()                    
                rospy.loginfo("Number of stone(s) found: " + str(len(self.so.stones)))
        # =============================================
        for s in self.so.stones:
            cv2.circle(drawable_image,s.get_center(),2,(0,0,0),2)
            cv2.putText(drawable_image, s.get_color_name(), s.get_center(), cv2.FONT_HERSHEY_SIMPLEX, 1.0, s.get_color(), lineType=cv2.LINE_AA)
        cv2.imshow("drawable_image window", drawable_image)
        # cv2.imshow("cv_image window", cv_image)
        cv2.waitKey(3)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)

def main(args):
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        # rospy.spin()
        while not rospy.core.is_shutdown():
            rospy.rostime.wallsleep(0.5)
            print("Press enter to search for stones")
            raw_input()
            ic.search()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)