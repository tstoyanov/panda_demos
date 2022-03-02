#include "panda_insertion/TrajectoryHandler.hpp"

#include "ros/package.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <tf/tf.h>


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

void TrajectoryHandler::writeSpiralTrajectories(Trajectory trajectory,
                                            Trajectory trajectory1,
                                            Trajectory trajectory2,
                                            Trajectory trajectory3,
                                            const std::string& fileName,
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

    for (auto& point : trajectory)
    {
        // actions
        outfile << point.x << "," << point.y << "," << point.z << "\n";
         
    }
    
    for (auto& point : trajectory1)
    {
        // actions
        outfile << point.x << "," << point.y << "," << point.z << "\n";
    }
    
    for (auto& point : trajectory2)
    {
        // actions
        outfile << point.x << "," << point.y << "," << point.z << "\n";
    }

    for (auto& point : trajectory3)
    {
        // actions
        outfile << point.x << "," << point.y << "," << point.z << "\n";
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

void TrajectoryHandler::writeDataset(const string& fileName,
                                        Point& point,
                                        geometry_msgs::Twist& twist,
                                        float reward,
                                        bool terminal,
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

    //state rotation
    tf::Quaternion q(transform.rotation.x, 
                     transform.rotation.y,
                     transform.rotation.z,
                     transform.rotation.w);
    tf::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    //action orientation
    tf::Quaternion q_action(panda->straightOrientation.x,
                            panda->straightOrientation.y,
                            panda->straightOrientation.z,
                            panda->straightOrientation.w);
    tf::Matrix3x3 m_action(q_action);
    double theta_x, theta_y, theta_z;
    m_action.getRPY(theta_x, theta_y, theta_z);
    
    // state, action, reward, terminal
    outfile // state(19 dimensional)
            << panda->q[0] << "," << panda->q[1] << "," << panda->q[2] << "," << panda->q[3] << ","
            << panda->q[4] << "," << panda->q[5] << "," << panda->q[6] << ","
            << panda->dq[0] << "," << panda->dq[1] << "," << panda->dq[2] << "," << panda->dq[3] << ","
            << panda->dq[4] << "," << panda->dq[5] << "," << panda->dq[6] << ","
            << transform.translation.x << "," 
            << transform.translation.y << ","
            << transform.translation.z << ","
            << yaw << ","
            << panda->getWrench().force.z << ","
            // action(8 dimensional)
            << point.x << ","
            << point.y << ","
            << point.z << ","
            << theta_z << ","
            << twist.linear.x << ","
            << twist.linear.y << ","
            << twist.linear.z << ","
            << twist.angular.z << ","
            // reward
            << reward << ","
            // terminal flag
            << terminal << "\n";

}

void TrajectoryHandler::writeSpiralDataset(const string& fileName,
                                        geometry_msgs::PoseStamped& message,
                                        double theta_z,
                                        geometry_msgs::Twist& twist,
                                        float reward,
                                        bool terminal,
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

    //state rotation
    tf::Quaternion q(transform.rotation.x, 
                     transform.rotation.y,
                     transform.rotation.z,
                     transform.rotation.w);
    tf::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
   
    //std::cout << "+++writing action:" << message.pose.position.x << ","
    //        << message.pose.position.y << ","
    //        << message.pose.position.z << ","
    //        << theta_z << std::endl;;

    // state, action, reward, terminal
    outfile // state(19 dimensional)
            << panda->q[0] << "," << panda->q[1] << "," << panda->q[2] << "," << panda->q[3] << ","
            << panda->q[4] << "," << panda->q[5] << "," << panda->q[6] << ","
            << panda->dq[0] << "," << panda->dq[1] << "," << panda->dq[2] << "," << panda->dq[3] << ","
            << panda->dq[4] << "," << panda->dq[5] << "," << panda->dq[6] << ","
            << transform.translation.x << "," 
            << transform.translation.y << ","
            << transform.translation.z << ","
            << yaw << ","
            << panda->getWrench().force.z << ","
            // action(8 dimensional)
            << message.pose.position.x << ","
            << message.pose.position.y << ","
            << message.pose.position.z << ","
            << theta_z << ","
            << twist.linear.x << ","
            << twist.linear.y << ","
            << twist.linear.z << ","
            << twist.angular.z << ","
            // reward
            << reward << ","
            // terminal flag
            << terminal << "\n";

}
