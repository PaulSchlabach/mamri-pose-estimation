# MAMRI Robot Control Module

The **MAMRI** module for 3D Slicer is a suite for pose-estimation and controlling of the MAMRI robotic arm. It features a hybrid feedback system that combines MRI-based pose estimation with real-time hardware feedback from optical encoders to ensure accurate and reliable robot positioning.

---

## Key Features

- **Automatic Pose Estimation**: Automatically segments petroleum jelly-filled fiducial markers from an MRI scan to calculate the robot's initial joint angles using inverse kinematics.
- **Trajectory Planning**: Generates a safe, collision-free path from the robot's current pose to a user-defined target, using an "up, over, and down" heuristic strategy.
- **Hardware Integration**: Provides direct control over the MAMRI motor controller and receives real-time position data from MR-safe fiber optic encoders via separate serial connections.
- **Live Status Monitoring**: A dedicated panel displays the real-time status of the robot, including encoder steps, motor controller steps, target positions, and the Tool-Tip-Coordinate (TCP) in the world frame.
- **Full Workflow Control**: Manages the entire biopsy procedure from within a single, intuitive user interface.

---

## User Interface & Workflow

### Workflow Steps

1.  **Pose Estimation**:

    - Load a MRI-scan of the MAMRI containing markers via the **Add DICOM DATA** tab in 3D Slicer.
    - Select the scan data in the drop down menu and Click **Start Robot Pose Estimation**. The module will detect the fiducials and solve the inverse kinematics. The solver needs at least the baseplate and the end effector fiducials to be contained in the scan.
    - As the baseplate is not moving across scans, the baseplate fiducials need to be determined just once and that transform can be saved through **Save Baseplate**.
    - If the baseplate is not in the field of view, check **Use Saved Baseplate** to load a previously saved position.

2.  **Trajectory Planning**:

    - Via the **Markups** tab, add the **Target** marker node and specify the target in the scan. The **Entry** marker node can also be manually picked.
    - Select the **Target** and **Entry** marker nodes that you have placed in the scene through the dropdown menus. Alternatively, use the **Find Closest Suitable Entry Point** button to have the module calculate an optimal entry point on the skin surface.
    - Adjust the **Safety distance** to specify how far the needle tip should stop from the entry point.
    - Click **Plan Safe Path**. The module calculates a collision-free trajectory. The start and end poses, along with collision status, are displayed in the **Planned Trajectory Information** table.
    - Preview the planned path using the **Trajectory Simulation** slider and play button.

3.  **Connection & Execution**:
    - In the **Connection** panel, refresh the COM ports and connect to the **MAMRI Controller** and the **Encoder** hardware.
    - Live robot status from the hardware will appear in the **Live Status** panel.
    - Use the buttons in the **Execution** panel to command the MAMRI:
      - **Move to Estimated Pose**: Moves the robot to the pose calculated in Step 1.
      - **Execute Trajectory on Robot**: Runs the full, planned trajectory.
      - **Return to Zero**: Sends the robot to its home position.
      - **STOP**: Immediately halts robot motion.
      - **Zero Hardware Pose**: Resets the step counters on the motor controller and encoder, used for calibration.

---

## Setup

To use this module, you need to connect both components:

- Connect the MAMRI Controller via USB.
- Connect the Encoder electronics via USB.

---

## Installation

To install this extension and the MAMRI module into 3D Slicer:

1.  Download this repository.
2.  Unpack the zip file. It should result in the folder **mamri-pose-estimation-main**.
3.  Open 3D Slicer.
4.  Open the Python terminal.
5.  Type in `slicer.util.pip_install('pyserial')`.
6.  In the Module drop down menu, under **Developer Tools** select **Extension Wizard**.
7.  In the **Extension Wizard** click **Select Extension**.
8.  Browse for the folder from step 2.
9.  The module is now installed and can be found in the Module drop down menu, under **Robotics** as **Mamri Robot Arm**.

---

## About

This module was developed by **Paul Schlabach** as part of a Master's Thesis at the research group **Robotics and Mechatronics** at the **University of Twente**. If you need to get in touch with me: **paul@schlabach.biz**. The project was supervised by **Vincent Groenhuis**. Thank you **Vincent**!!!
