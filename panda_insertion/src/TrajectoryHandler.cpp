#include "panda_insertion/TrajectoryHandler.hpp"

#include "ros/package.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>

using namespace std;

TrajectoryHandler::TrajectoryHandler() {}

TrajectoryHandler::TrajectoryHandler(ros::NodeHandle* nodeHandler, Panda* panda)
{
    this->nodeHandler = nodeHandler;
    this->panda = panda;
}

Trajectory TrajectoryHandler::generateArchimedeanSpiral(double a, double b,
                                                        int nrOfPoints)
{
    Trajectory spiral;
    geometry_msgs::Transform transform;

    // Get parameter from server
    vector<double> goal;
    const string goalParameter = "/spiral/goal";
    if (!nodeHandler->getParam(goalParameter  , goal))
    {
        throw runtime_error("Could not get parameter from server");
    }

    mutex.lock();
    transform = panda->transformStamped.transform;
    mutex.unlock();

    Point startPoint;
    startPoint.x = transform.translation.x;
    startPoint.y = transform.translation.y;
    startPoint.z = goal.at(2);
    

    const double ROTATIONS = 5;
    double x = startPoint.x;
    double y = startPoint.y;
    double z = startPoint.z;

    for (double n = 1.0; n <= nrOfPoints; n += 1.0)
    {
        Point point;

        double theta = sqrt(n / nrOfPoints) * (ROTATIONS * 2 * M_PI);
        double r = (a + b * theta);

        x = startPoint.x + r * cos(theta);
        y = startPoint.y + r * sin(theta);

        point.x = x;
        point.y = y;
        point.z = z;

        spiral.push_back(point);
    }

    //reverse(spiral.begin(), spiral.end());

    return spiral;
}

Trajectory TrajectoryHandler::generateInitialPositionTrajectory(int nrOfPoints)
{
    Trajectory trajectory;
    geometry_msgs::Transform transform;

    // Get parameter from server
    vector<double> goal;
    const string goalParameter = "/move_to_initial_position/goal";
    if (!nodeHandler->getParam(goalParameter, goal))
    {
        throw runtime_error("Could not get parameter from server");
    }

    mutex.lock();
    transform = panda->transformStamped.transform;
    mutex.unlock();

    Point startPoint;
    startPoint.x = transform.translation.x;
    startPoint.y = transform.translation.y;
    startPoint.z = transform.translation.z;

    ROS_DEBUG_STREAM_ONCE("Startpoint xyz: " << startPoint.x << ", " << startPoint.y << ", " << startPoint.z);
    Point goalPoint;
    goalPoint.x = goal.at(0);
    goalPoint.y = goal.at(1);
    goalPoint.z = goal.at(2);

    Point pointSteps;
    pointSteps.x = (goalPoint.x - startPoint.x) / nrOfPoints;
    pointSteps.y = (goalPoint.y - startPoint.y) / nrOfPoints;
    pointSteps.z = (goalPoint.z - startPoint.z) / nrOfPoints;

    for (auto i = 1; i <= nrOfPoints; i++)
    {
        Point point;

        point.x = startPoint.x + (pointSteps.x * i);
        point.y = startPoint.y + (pointSteps.y * i);
        point.z = startPoint.z + (pointSteps.z * i);

        trajectory.push_back(point);
    }

    return trajectory;
}

Trajectory TrajectoryHandler::generateExternalDownTrajectory(int nrOfPoints)
{
    Trajectory trajectory;
    geometry_msgs::Transform transform;

    // Get parameter from server
    vector<double> goal;
    const string goalParameter = "/external_down_movement/goal";
    if (!nodeHandler->getParam(goalParameter  , goal))
    {
        throw runtime_error("Could not get parameter from server");
    }

    mutex.lock();
    transform = panda->transformStamped.transform;
    mutex.unlock();

    Point startPoint;
    startPoint.x = transform.translation.x;
    startPoint.y = transform.translation.y;
    startPoint.z = transform.translation.z;

    ROS_DEBUG_STREAM_ONCE("Startpoint xyz: " << startPoint.x << ", " << startPoint.y << ", " << startPoint.z);
    Point goalPoint;
    goalPoint.x = transform.translation.x;
    goalPoint.y = transform.translation.y;
    goalPoint.z = goal.at(2);

    Point pointSteps;
    pointSteps.x = (goalPoint.x - startPoint.x) / nrOfPoints;
    pointSteps.y = (goalPoint.y - startPoint.y) / nrOfPoints;
    pointSteps.z = (goalPoint.z - startPoint.z) / nrOfPoints;

    for (auto i = 1; i <= nrOfPoints; i++)
    {
        Point point;

        point.x = startPoint.x + (pointSteps.x * i);
        point.y = startPoint.y + (pointSteps.y * i);
        point.z = startPoint.z + (pointSteps.z * i);

        trajectory.push_back(point);
    }

    return trajectory;
}

Trajectory TrajectoryHandler::generateInternalUpTrajectory(int nrOfPoints)
{
    Trajectory trajectory;
    geometry_msgs::Transform transform;

    //Get parameter from server
    vector<double> goal;
    const string goalParameter = "/internal_up_movement/goal";
    if (!nodeHandler->getParam(goalParameter, goal))
    {
        throw runtime_error("Could not get parameter from server");
    }

    mutex.lock();
    transform = panda->transformStamped.transform;
    mutex.unlock();


    Point startPoint;
    startPoint.x = transform.translation.x;
    startPoint.y = transform.translation.y;
    startPoint.z = transform.translation.z;



    ROS_DEBUG_STREAM_ONCE("Startpoint xyz: " << startPoint.x << ", " << startPoint.y << ", " << startPoint.z);
    Point goalPoint;
    goalPoint.x = transform.translation.x;
    goalPoint.y = transform.translation.y;
    goalPoint.z = goal.at(2);
    
    Point pointSteps;
    pointSteps.x = (goalPoint.x - startPoint.x) / nrOfPoints;
    pointSteps.y = (goalPoint.y - startPoint.y) / nrOfPoints;
    pointSteps.z = (goalPoint.z - startPoint.z) / nrOfPoints;


    for (auto i = 1; i <= nrOfPoints; i++)
    {
        Point point;

        point.x = startPoint.x + (pointSteps.x * i);
        point.y = startPoint.y + (pointSteps.y * i);
        point.z = startPoint.z + (pointSteps.z * i);

        trajectory.push_back(point);
    }

    return trajectory;
} 



void TrajectoryHandler::writeTrajectoryToFile(Trajectory trajectory,
                                              geometry_msgs::Twist twist,
                                              const string& fileName,
                                              bool appendToFile,
                                              bool lastTraj)
{
    ofstream outfile;

    std::stringstream filePath;
    filePath << ros::package::getPath("panda_insertion")
             << "/trajectories/" << fileName;

    if (appendToFile)
    {
        outfile.open(filePath.str(), ios_base::app);
    }
    else
    {
        outfile.open(filePath.str());
    }

    if (!outfile.is_open())
    {
        ROS_WARN_STREAM("Unable to open file " << filePath.str());
        return;
    }

    for (auto& point : trajectory)
    {
        // actions
        outfile << point.x << "," << point.y << "," << point.z << ","
                << twist.linear.x << "," << twist.linear.y << "," << twist.linear.z << ","
                << twist.angular.x << "," << twist.angular.y << "," << twist.angular.z; 
         
        // terminals
        if (!lastTraj){
            outfile << "," << "0" << "\n";
        }
        else{
            if (&point != &trajectory.back()){
                outfile << "," << "0" << "\n";
            }
            else{
                outfile << "," << "1" << "\n";
            }
        }
    }

    ROS_DEBUG_STREAM("Wrote trajectory to file " << filePath.str());

    outfile.close();
}

void TrajectoryHandler::writeStateToFile(const string& fileName,
                                        bool appendToFile)
{
    ofstream outfile;

    std::stringstream filePath;
    filePath << ros::package::getPath("panda_insertion")
             << "/trajectories/" << fileName;

    if (appendToFile)
    {
        outfile.open(filePath.str(), ios_base::app);
    }
    else
    {
        outfile.open(filePath.str());
    }

    if (!outfile.is_open())
    {
        ROS_WARN_STREAM("Unable to open file " << filePath.str());
        return;
    }

    // get robot transform
    mutex.lock();
    geometry_msgs::Transform transform = panda->transformStamped.transform;
    mutex.unlock();

    outfile << transform.translation.x << "," 
            << transform.translation.y << "," 
            << transform.translation.z << ","
            << transform.rotation.x << ","
            << transform.rotation.y << ","
            << transform.rotation.z << ","
            << transform.rotation.w << ","
            << "\n";
}

