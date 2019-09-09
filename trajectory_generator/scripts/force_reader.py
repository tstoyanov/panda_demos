#!/usr/bin/env python
import sys
import json
import copy
import rospy
import argparse

from geometry_msgs.msg import WrenchStamped
from rospy_message_converter import json_message_converter

import pprint
pp = pprint.PrettyPrinter(indent=2)

g = 9.807

import matplotlib.pyplot as plt
import numpy as np
x = []
y = []

# You probably won't need this if you're embedding things in a tkinter plot...
plt.ion()

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.set_ylabel('z force')
line11, = ax1.plot(x, y, 'r-', label="low pass filter (a = 0.99)", alpha=0.7, linewidth=2) # Returns a tuple of line objects, thus the comma
line12, = ax1.plot(x, y, 'b-', label="sample mean", alpha=0.7, linewidth=2)
# line13, = ax1.plot(x, y, 'k-', label="new data", marker="o", alpha=0.7)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=2, fancybox=True, shadow=True)
ax2 = fig.add_subplot(212)
ax2.set_ylabel('last sample')
# line21, = ax2.plot(x, y, 'r-', label="dumb mean", alpha=0.7) 
# line22, = ax2.plot(x, y, 'b-', label="sample mean", alpha=0.7) 
line23, = ax2.plot(x, y, 'k-', label="new data", marker="o", alpha=0.7)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=2, fancybox=True, shadow=True)

# calulates the mean, population variance and sample variance every "batch_size" data points
class statistics_calculator:
    def __init__(self, batch_size=100, topic="/panda/franka_state_controller/F_ext", message_type=WrenchStamped):

        self.batch_count = 0
        self.dumb_count = 0
        self.dumb_mean = {
            "force": {
                "x": None,
                "y": None,
                "z": None
            },
            "torque": {
                "x": None,
                "y": None,
                "z": None
            }
        }

        self.topic_sub = rospy.Subscriber(topic, message_type, self.callback)
        self.batch_size = batch_size
        self.count = 0
        self.mean = {
            "force": {
                "x": None,
                "y": None,
                "z": None
            },
            "torque": {
                "x": None,
                "y": None,
                "z": None
            }
        }
        self.M2 = {
            "force": {
                "x": 0,
                "y": 0,
                "z": 0
            },
            "torque": {
                "x": 0,
                "y": 0,
                "z": 0
            }
        }
        self.variance = {
            "force": {
                "x": 0,
                "y": 0,
                "z": 0
            },
            "torque": {
                "x": 0,
                "y": 0,
                "z": 0
            }
        }
        self.sample_variance = {
            "force": {
                "x": 0,
                "y": 0,
                "z": 0
            },
            "torque": {
                "x": 0,
                "y": 0,
                "z": 0
            }
        }
        self.last_sample = {
            "force": {
                "x": 0,
                "y": 0,
                "z": 0
            },
            "torque": {
                "x": 0,
                "y": 0,
                "z": 0
            }
        }
        

    # for a new value new_value, compute the new count, new mean, the new M2.
    # mean accumulates the mean of the entire dataset
    # M2 aggregates the squared distance from the mean
    # count aggregates the number of samples seen so far
    # after updating "batch_size" amount of times, calculates
    # the population and sample variance and resets the batch_count variable
    def update(self, new_value):
        self.last_sample = copy.deepcopy(new_value)
        if self.count == 0:
            self.mean = copy.deepcopy(new_value)
        
        # (count, mean, M2) = existing_aggregate
        self.count += 1
        self.batch_count += 1
        
        if self.dumb_count == 0:
            self.dumb_mean = copy.deepcopy(new_value)
        self.dumb_count += 1
        for key, coordinates in new_value.iteritems():
            for coordinate in coordinates.keys():
                coordinate = str(coordinate)
                # self.dumb_mean[key][coordinate] = ((self.dumb_count-1)*(self.dumb_mean[key][coordinate]) + new_value[key][coordinate]) / self.dumb_count
                self.dumb_mean[key][coordinate] = 0.99*(self.dumb_mean[key][coordinate]) + 0.01*(new_value[key][coordinate])
                # self.dumb_mean[key][coordinate] = new_value[key][coordinate]
        # print ("dumb mean:")
        # pp.pprint(self.dumb_mean)

        for key, coordinates in new_value.iteritems():
            for coordinate in coordinates.keys():
                coordinate = str(coordinate)
                delta = new_value[key][coordinate] - self.mean[key][coordinate]
                self.mean[key][coordinate] += delta / self.count
                delta2 = new_value[key][coordinate] - self.mean[key][coordinate]
                if self.count == 1:
                    self.M2[key][coordinate] = 0
                else:
                    self.M2[key][coordinate] += delta * delta2
        
        # delta = new_value - self.mean
        # self.mean += delta / self.count
        # delta2 = new_value - self.mean
        # self.M2 += delta * delta2

        if self.batch_count >= self.batch_size:
            for key, coordinates in self.mean.iteritems():
                for coordinate in coordinates.keys():
                    self.variance[key][coordinate] = self.M2[key][coordinate]/self.count
                    self.sample_variance[key][coordinate] = self.M2[key][coordinate]/(self.count - 1)
            
            print ("mean:")
            pp.pprint(self.mean)
            print ("population variance:")
            pp.pprint(self.variance)
            print ("sample variance:")
            pp.pprint(self.sample_variance)
            print ("----------------------------------------------")
            # print ("mean = ", self.mean)
            # print ("polulation variance = ", self.variance)
            # print ("sample variance = ", self.sample_variance)
            self.batch_count = 0
        

        # return (count, mean, M2)

    # retrieve the mean, variance and sample variance from an aggregate
    def get_statistics(self):
        # (count, mean, M2) = existing_aggregate
        # (mean, variance, sample_variance) = (self.mean, self.M2/self.count, self.M2/(self.count - 1))
        
        if self.count < 2:
            return float('nan')
        else:
            #       mean,      variance,           sample_variance
            # return (self.mean, self.M2/self.count, self.M2/(self.count - 1))
            return (self.mean, self.variance, self.sample_variance)
    
    def callback(self, data):
        json_str = json_message_converter.convert_ros_message_to_json(data.wrench)
        json_obj = json.loads(json_str)
        self.update(json_obj)
    
    

def main(args):
    rospy.init_node('force_reader', anonymous=True)
    stats = statistics_calculator(topic="/panda/franka_state_controller/F_ext", message_type=WrenchStamped)
    try:
        rospy.rostime.wallsleep(1)
        start = rospy.get_rostime()
        while (start.secs == 0 | start.nsecs == 0):
            start = rospy.get_rostime()
            # start = start.to_sec()
        while not rospy.core.is_shutdown():
            rospy.rostime.wallsleep(0.1)
            
            line11.set_xdata(np.append(line11.get_xdata(), stats.dumb_count))
            line11.set_ydata(np.append(line11.get_ydata(), stats.dumb_mean["force"]["z"]))
            line12.set_xdata(np.append(line12.get_xdata(), stats.dumb_count))
            line12.set_ydata(np.append(line12.get_ydata(), stats.mean["force"]["z"]))
            # line13.set_xdata(np.append(line13.get_xdata(), stats.dumb_count))
            # line13.set_ydata(np.append(line13.get_ydata(), stats.last_sample["force"]["z"]))
            # line21.set_xdata(np.append(line21.get_xdata(), stats.dumb_count))
            # line21.set_ydata(np.append(line21.get_ydata(), stats.dumb_mean["force"]["x"]))
            # line22.set_xdata(np.append(line22.get_xdata(), stats.dumb_count))
            # line22.set_ydata(np.append(line22.get_ydata(), stats.mean["force"]["x"]))
            # line23.set_xdata(np.append(line23.get_xdata(), stats.dumb_count))
            # line23.set_ydata(np.append(line23.get_ydata(), stats.last_sample["force"]["x"]))
            line23.set_xdata(np.append(line23.get_xdata(), stats.dumb_count))
            line23.set_ydata(np.append(line23.get_ydata(), stats.last_sample["force"]["z"]))

            now = rospy.get_rostime()

            # ax1.text(1, 0.65, "estimated lpf mass = " + str(stats.dumb_mean["force"]["z"]/g))
            # ax1.text(2, 0.65, "estimated sample mass = " + str(stats.mean["force"]["z"]/g))
            print ("Elapsed time: " + str(now.to_sec() - start.to_sec()))
            print ("Estimated lpf mass = " + str(stats.dumb_mean["force"]["z"]/g*1000) + " [g]")
            print ("Estimated sample mass = " + str(stats.mean["force"]["z"]/g*1000) + " [g]")
            print ("Delta = " + str(abs((stats.dumb_mean["force"]["z"]/g*1000)-(stats.mean["force"]["z"]/g*1000))) + " [g]\n")

            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
        # rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)