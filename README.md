
## Gesture Controlled Robot Using ROS 2 and Gazebo

This project demonstrates a gesture-controlled differential drive robot, simulated in Gazebo, that responds to hand gestures detected using a MediaPipe-based recognition system. The project comprises two ROS 2 packages:

1. Gesture Recognition Package: 
This package detects hand gestures and publishes movement commands to control the robot.
2. Robot Control and Simulation Package:
This package simulates a differential drive robot in Gazebo using the ROS 2 control framework, enabling it to respond to commands received from the gesture recognition system.

### Project Overview

1. Gesture Recognition: The system utilizes MediaPipe to recognize specific hand gestures. Upon detecting a gesture, it publishes movement commands (such as forward, backward, left, and right) to a ROS 2 topic. These commands are subscribed to by the robot control package to drive the robot in Gazebo.

2. Robot Control & Simulation: The robot is a simulated differential drive robot, meaning that it is equipped with two independently driven wheels on either side. By varying the speed and direction of each wheel, the robot can achieve both linear and angular motion. The ROS 2 control framework is used for simulating and controlling the robot’s movements.

### Differential Drive Robot

A differential drive robot operates using two wheels, one on the left and one on the right. The robot moves forward or backward by rotating both wheels in the same direction at the same speed. To turn, the wheels are rotated at different speeds or in opposite directions. This type of motion is ideal for flat surfaces and offers a simple yet effective control system for robot locomotion.
####Key Components:

1. Wheel Separation: The distance between the two wheels. It affects how sharply the robot can turn.
2. Wheel Radius: The size of the wheels, which influences the speed and distance traveled per wheel rotation.
3. Velocity and Acceleration Limits: Control parameters that limit how fast the robot can move and accelerate, ensuring smoother operation.
### ROS 2 Packages

1. Gesture Recognition Package

This package is responsible for:

##### Gesture Detection: 
- Uses MediaPipe for hand tracking and gesture recognition.
- Command Publisher: Based on the detected gesture, it publishes velocity commands (cmd_vel) to a topic, which are then used to control the robot’s motion.

2. Robot Control and Simulation Package

This package contains the configuration and simulation files for the robot, including:

- URDF/XACRO Files: Define the robot's physical model and its components (wheels, chassis, sensors, etc.).
- ROS 2 Control Configuration (controller_manager.yaml): Defines how the robot interacts with the ROS 2 control framework, including the differential drive controller and the joint state broadcaster.
- Launch Files: Automate the startup of the robot simulation, including loading the robot’s description, control parameters, and Gazebo plugins.
- Gazebo Plugins: Integrate ROS 2 control with Gazebo to simulate real-world physics and movement.

