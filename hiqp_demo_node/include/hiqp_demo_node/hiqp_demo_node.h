#ifndef HIQP_DEMO_NODE_HH
#define HIQP_DEMO_NODE_HH

#include <hiqp_ros/hiqp_client.h>
#include <hiqp_msgs/Primitive.h>
#include <hiqp_msgs/Task.h>

#include <std_srvs/Empty.h>

#include <Eigen/Core>
#include <Eigen/Geometry>


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
    //related to grasp interval
    hiqp_msgs::Primitive upper;
    hiqp_msgs::Primitive lower;
    hiqp_msgs::Primitive left;
    hiqp_msgs::Primitive right;
    hiqp_msgs::Primitive inner;
    hiqp_msgs::Primitive outer;
    hiqp_msgs::Primitive obj_z_axis;
    
    //related to the robot itself
    hiqp_msgs::Primitive point_eef; 		//point on end effector
    hiqp_msgs::Primitive eef_approach_axis;	//axis pointing forward from gripper
    hiqp_msgs::Primitive eef_orthogonal_axis;	//axis pointing orthogonal to grasping plane

    std::string obj_frame_;  // object frame
    std::string e_frame_;    // endeffector frame
    Eigen::Vector3d e_;      // endeffector point expressed in e_frame_
    //float opening_angle;

    double DYNAMICS_GAIN=1.0;
    bool initialized;

    std::vector<hiqp_msgs::Task> tasks_;

  public:    

    ///sets up the grasp interval
    void setInterval(std::string obj_frame, std::string e_frame, Eigen::Vector3d &eef,
		    Eigen::Affine3d &obj_pose);

    //getters
    virtual void getPrimitiveList(std::vector<hiqp_msgs::Primitive> &primitives);
    virtual void getTasksList(std::vector<hiqp_msgs::Task> &tasks);

    virtual void getPrimitiveNames(std::vector<std::string> &primitives);
    virtual void getTaskNames(std::vector<std::string> &tasks);

};

///** Alternative formulation for top grasps. 
class TopGraspInterval : public TaskPrimitiveWrapper {
  private:
    hiqp_msgs::Primitive cylinder;
    hiqp_msgs::Primitive vertical_axis;
    hiqp_msgs::Primitive horizontal_axis;
    hiqp_msgs::Primitive plane_above;
    hiqp_msgs::Primitive plane_below;
    hiqp_msgs::Primitive point_eef;

    std::string obj_frame_;  // object frame
    std::string e_frame_;    // endeffector frame
    Eigen::Vector3d e_;      // endeffector point expressed in e_frame_
    //float opening_angle;

    bool initialized;

    std::vector<hiqp_msgs::Task> tasks_;

  public:    

    ///sets up the grasp interval
    void setInterval(std::string obj_frame, std::string e_frame, Eigen::Vector3d &eef,
		    Eigen::Affine3d &obj_pose);

    //getters
    virtual void getPrimitiveList(std::vector<hiqp_msgs::Primitive> &primitives);
    virtual void getTasksList(std::vector<hiqp_msgs::Task> &tasks);

    virtual void getPrimitiveNames(std::vector<std::string> &primitives);
    virtual void getTaskNames(std::vector<std::string> &tasks);

};

class DemoGrasping {

  public: 
    DemoGrasping();
  private:
    unsigned int n_jnts;
    std::vector<std::string> link_frame_names;

    ros::NodeHandle nh_;
    ros::NodeHandle n_;

    hiqp_ros::HiQPClient hiqp_client_;

    Eigen::Vector3d eef_offset_;
    Eigen::Affine3d object_pose_;

    /// Servers
    ros::ServiceServer start_demo_srv_;
    bool startDemo(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res);

    /// that's the equilibrium pose of the manipulator. used for resolving redundancies. 
    std::vector<double> start_config_;

};

}
#endif
