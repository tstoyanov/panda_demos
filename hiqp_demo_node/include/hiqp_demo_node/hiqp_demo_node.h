#ifndef HIQP_DEMO_NODE_HH
#define HIQP_DEMO_NODE_HH

#include <hiqp_ros/hiqp_client.h>
#include <hiqp_msgs/Primitive.h>
#include <hiqp_msgs/Task.h>

#include <std_srvs/Empty.h>

#include <Eigen/Core>

namespace hiqp_panda_demo {

///** Abstract interface for a set of tasks and corresponding primitives
/// must implement methods to get the lists of primitives and tasks that
/// should be loaded to the client
class TaskPrimitiveWrapper {
  public:
    virtual void getPrimitiveList(std::vector<hiqp_msgs::Primitive> &primitives) = 0;
    virtual void getTasksList(std::vector<hiqp_msgs::Task> &tasks) = 0;
    
    virtual void getPrimitiveNames(std::vector<std::string> &primitives) = 0;
    virtual void getTaskNames(std::vector<std::string> &tasks) = 0;
};

///**To simplify, a grasp intervall is given as two concentric cylinders,
/// described by axis v and a point p on the axis (referenced in a static
/// obj_frame), and two planes. The controller will try to bring endeffector
/// point e, expressed in frame e_frame, inside the intervall described by the
/// two cylinders and the planes (i.e., inside the shell formed by the cylinders
/// and in between the planes described by n^Tx - d = 0)
class GraspInterval : public TaskPrimitiveWrapper {
  private:
    hiqp_msgs::Primitive upper;
    hiqp_msgs::Primitive lower;
    hiqp_msgs::Primitive left;
    hiqp_msgs::Primitive right;
    hiqp_msgs::Primitive inner;
    hiqp_msgs::Primitive outer;

    std::string obj_frame_;  // object frame
    std::string e_frame_;    // endeffector frame
    Eigen::Vector3d e_;      // endeffector point expressed in e_frame_
    float opening_angle;

    bool initialized;

    std::vector<hiqp_msgs::Task> tasks_;

  public:    

    ///sets up the grasp interval
    void setInterval();

    //getters
    virtual void getPrimitiveList(std::vector<hiqp_msgs::Primitive> &primitives);
    virtual void getTasksList(std::vector<hiqp_msgs::Task> &tasks);

    virtual void getPrimitiveNames(std::vector<std::string> &primitives);
    virtual void getTaskNames(std::vector<std::string> &tasks);

};

class DemoNode {

  public: 
    DemoNode();
  private:
    unsigned int n_jnts;
    std::vector<std::string> link_frame_names;

    ros::NodeHandle nh_;
    ros::NodeHandle n_;

    hiqp_ros::HiQPClient hiqp_client_;

    /// Servers
    ros::ServiceServer start_demo_srv_;
    bool startDemo(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res);
};

}
#endif
