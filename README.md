# Peg-In-Hole Insertion of Industrial Workpieces


## About The Project

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

## Usage
### Lanch files
There are two (launch files)[http://wiki.ros.org/roslaunch/XML] to run the `panda_insertion` package, one for running in simulation and one for running live on the robot. To execute the launch files, use the tool [roslaunch](http://wiki.ros.org/roslaunch). To see what nodes are being launch you can look at the content of the `.launch` files.

Run the package in **simulation** with the following command:
```bash
$ roslaunch panda_insertion panda_insertion_sim.launch
```

Run the package on the **robot** with the following command:
```bash
$ roslaunch panda_insertion panda_insertion_live.launch
```



## License
[BSD](https://opensource.org/licenses/BSD-3-Clause)

## Authors
- [Tobias Lans](https://github.com/lanstobias)
- [Bobo Lillqvist](https://github.com/BoboLillqvist)

## Contributing
