# Panda Insertion
Panda insertion is a ROS package for performing a peg-in-hole insertion with a [Franka Emika Panda](https://www.franka.de/panda/) robot arm.

## About The Project
This project is an exam work done for the Center for Applied Autonomous Sensor Systems (AASS) which is a research environment on autonomous systems, robotics and artificial intelligence at Örebro University. AASS was contacted by SAAB to investigate a solution for automation of drill assembly using a robotic arm with lower precision than the clearance of the hole would demand.

## Installation

### ROS
This project is build with Kinetic Kame distribution. Follow the [ROS Kinetic installation instructions](http://wiki.ros.org/kinetic/Installation) to install the environemnt needed to run the packages.

### Clone
After ROS has been fully installed, you can clone the branch into your machine inside the `src` directory of your ROS workspace. Clone the branch with the following command:
```bash
$ git clone --single-branch --branch insertion https://github.com/tstoyanov/panda_demos
```

### Packages Dependencies
There are several packages included in this repository which all have dependencies that needs to be installed. To do this, use the tool [rosdep](http://wiki.ros.org/rosdep). To install all dependencies needed, run the following command:
```
$ rosdep install --from-paths src --ignore-src -r -y
```
If you run in to any trouble, check out the [rosdep tutorial](http://wiki.ros.org/ROS/Tutorials/rosdep).

### Build
Once the repository has been cloned the workspace should look like:
```
ros_workspace/
├── build/
├── devel/
└── src/
    └── panda_demos/
```
You can now [build the packages using catkin](http://wiki.ros.org/ROS/Tutorials/BuildingPackages). To build the packages, navigate to the root of your ROS workspace and execute the following command:

```
$ catkin_make
```

## Code Structure


## Usage
### Lanch Files
There are two (launch files)[http://wiki.ros.org/roslaunch/XML] to run the `panda_insertion` package, one for running in simulation and one for running live on the robot. To execute the launch files, use the tool [roslaunch](http://wiki.ros.org/roslaunch). To see what nodes are being launch you can look at the content of the `.launch` files.

Run the package in **simulation** with the following command:
```bash
$ roslaunch panda_insertion panda_insertion_sim.launch
```

Run the package on the **robot** with the following command:
```bash
$ roslaunch panda_insertion panda_insertion_live.launch
```

### Available Services
There are some ROS [services](http://wiki.ros.org/Services) to help debugging and testing the state machine:
- **change_state**, used for manually changing to a state in the state machine. Example usage of changing the state to the *spiralMotion* state:
    ```bash
    $ rosservice call /change_state "Spiralmotion"
    ```
    Notice that the state name is not case sensative.

- **swap_controller**, used for swapping the active controller. Example usage of swapping from the *impedance controller* to the *position joint trajectory* controller:
    ```bash
    $ rosservice call /swap_controller "impedance_controller\
    position_joint_trajectory_controller"
    ```

### Parameters
Some of the data used in the panda_insertion controller are parameterized and are available to modify in the `config/spring_parameters.yaml` file. The parameters are:
- Goal x, y and z-coordinate.
- Translational stiffness
- Rotational stiffness

## Authors
- [Tobias Lans](https://github.com/lanstobias) - [lanstobias@gmail.com](mailto:lanstobias@gmail.com)
- [Bobo Lillqvist](https://github.com/BoboLillqvist) - [bobo.lillqvist@gmail.com](mailto:bobo.lillqvist@gmail.com)

## License
[BSD](https://github.com/tstoyanov/panda_demos/blob/insertion/LICENSE.md)

## Acknowledgement 
- Our supervisor [Todor Stoyanov](https://github.com/tstoyanov)
