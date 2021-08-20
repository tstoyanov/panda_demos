#ifndef TRAJECTORY_HANDLER_H
#define TRAJECTORY_HANDLER_H

#include "ros/ros.h"

#include "panda_insertion/Panda.hpp"
#include <boost/thread/mutex.hpp>
#include "geometry_msgs/Twist.h"
#include "vector"
#include "string"

//namespace 

typedef struct Point
{
    double x, y, z;
}Point;

typedef std::vector<Point> Trajectory;

class TrajectoryHandler
{
private:
    ros::NodeHandle* nodeHandler;
    Panda* panda;
    boost::mutex mutex;

public:
    TrajectoryHandler();
    TrajectoryHandler(ros::NodeHandle* nodeHandler, Panda* panda);

    Trajectory generateArchimedeanSpiral(double a, double b, int nrOfPoints);
    Trajectory generateInitialPositionTrajectory(int nrOfPoints);
    Trajectory generateExternalDownTrajectory(int nrOfPoints);
    Trajectory generateInternalUpTrajectory(int nrOfPoints);

    void writeTrajectoryToFile(Trajectory trajectory,
                               geometry_msgs::Twist twist,
                               const std::string& fileName,
                               bool appendToFile = false,
                               bool lastTraj = false);

    void writeStateToFile(const std::string& fileName, bool appendToFile = false);

    void writeDataset(const std::string& fileName, Point& point, geometry_msgs::Twist& twist, 
                    float reward, bool terminal, bool appendToFile = false);
};

#endif
