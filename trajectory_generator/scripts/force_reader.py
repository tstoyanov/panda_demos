import sys
import rospy
import argparse

from geometry_msgs.msg import WrenchStamped
from rospy_message_converter import json_message_converter


# calulates the mean, population variance and sample variance every "batch_size" data points
class statistics_calculator:
    def __init__(self, batch_size=30, topic="/panda/franka_state_controller/F_ext", message_type=WrenchStamped):
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

    # for a new value new_value, compute the new count, new mean, the new M2.
    # mean accumulates the mean of the entire dataset
    # M2 aggregates the squared distance from the mean
    # count aggregates the number of samples seen so far
    # after updating "batch_size" amount of times calculates
    # the population and sample variance and resets the accumulators
    def update(self, new_value):
        if self.count == 0:
            self.mean = new_value
        # (count, mean, M2) = existing_aggregate
        self.count += 1
        for key, coordinates in new_value.iteritems():
            for coordinate in coordinate.keys():
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

        if self.count >= self.batch_size:
            for key, coordinates in self.mean.iteritems():
                for coordinate in coordinate.keys():
                    self.variance[key][coordinate] = self.M2[key][coordinate]/self.count
                    self.sample_variance[key][coordinate] = self.M2[key][coordinate]/(self.count - 1)
            print ("mean = ", self.mean)
            print ("polulation variance = ", self.variance)
            print ("sample variance = ", self.sample_variance)
            self.count = 0
        

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
        json_str = json_message_converter.convert_ros_message_to_json(data)
        json_obj = json.loads(json_str)
        self.update(json_obj)
    
    

def main(args):
    rospy.init_node('force_reader', anonymous=True)
    stats = statistics_calculator(topic="/panda/franka_state_controller/F_ext", message_type=WrenchStamped)

if __name__ == '__main__':
    main(sys.argv)