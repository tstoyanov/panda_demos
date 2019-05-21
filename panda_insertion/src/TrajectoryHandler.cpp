#include "panda_insertion/TrajectoryHandler.hpp"

#include "ros/package.h"

#include <iostream>
#include <fstream>
#include <cmath>

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

    double initX = double(panda->position.x);
    double initY = double(panda->position.y);
    double initZ = double(panda->position.z);

    const double RANGE = (12 * M_PI);
    double x = initX, y = initY, z = initZ;

    for (auto i = 0; i <= nrOfPoints; i++)
    {
        Point point;

        double theta = i * (RANGE / nrOfPoints);
        double r = (a - b * theta);

        x = initX + r * cos(theta);
        y = initY + r * sin(theta);

        point.x = x;
        point.y = y;
        point.z = z;

        spiral.push_back(point);
    }
    return spiral;
}

Trajectory TrajectoryHandler::generateInitialPositionTrajectory(int nrOfPoints)
{
    Trajectory trajectory;
    geometry_msgs::Transform transform;

    mutex.lock();
    transform = panda->transformStamped.transform;
    mutex.unlock();

    Point startPoint;
    startPoint.x = transform.translation.x;
    startPoint.y = transform.translation.y;
    startPoint.z = transform.translation.z;

    ROS_DEBUG_STREAM_ONCE("Startpoint xyz: " << startPoint.x << ", " << startPoint.y << ", " << startPoint.z);
    Point goalPoint;
    goalPoint.x = 0.153;
    goalPoint.y = 0.345;
    goalPoint.z = 0.050;

    Point pointSteps;
    pointSteps.x = (goalPoint.x - startPoint.x) / nrOfPoints;
    pointSteps.y = (goalPoint.y - startPoint.y) / nrOfPoints;
    pointSteps.z = (goalPoint.z - startPoint.z) / nrOfPoints;

    for (auto i = 0; i <= nrOfPoints; i++)
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
    goalPoint.z = -0.009;

    Point pointSteps;
    pointSteps.x = (goalPoint.x - startPoint.x) / nrOfPoints;
    pointSteps.y = (goalPoint.y - startPoint.y) / nrOfPoints;
    pointSteps.z = (goalPoint.z - startPoint.z) / nrOfPoints;

    for (auto i = 0; i <= nrOfPoints; i++)
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
                                              const string& fileName,
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

    for (auto point : trajectory)
    {
        outfile << point.x << "," << point.y << "," << point.z << "\n";
    }

    ROS_DEBUG_STREAM("Wrote trajectory to file " << filePath.str());

    outfile.close();
}

