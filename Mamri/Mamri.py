import logging
import os
import itertools
import math
from typing import Annotated, Optional, Dict, List, Tuple
import time

import serial
import serial.tools.list_ports

import vtk
import SimpleITK as sitk
import sitkUtils
import scipy.optimize
import numpy as np
import qt

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin, loadModel
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLModelNode
from slicer import vtkMRMLLinearTransformNode
from slicer import vtkMRMLSegmentationNode
from slicer import vtkMRMLMarkupsFiducialNode

#
# Mamri
#
class Mamri(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Mamri Robot Arm")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Robotics")]
        self.parent.dependencies = []
        self.parent.contributors = ["Paul Schlabach (University of Twente)"]
        self.parent.helpText = _("""This module automatically detects fiducial markers in MRI scans
and renders the MAMRI robot model and estimates joint angles.""")
        self.parent.acknowledgementText = _("""
        This module was developed as part of a Master's Thesis at the University of Twente.
        """)

#
# MamriParameterNode
#
@parameterNodeWrapper
class MamriParameterNode:
    inputVolume: vtkMRMLScalarVolumeNode
    useSavedBaseplate: bool = False
    applyEndEffectorCorrection: bool = False

    segmentationNode: vtkMRMLSegmentationNode
    targetFiducialNode: vtkMRMLMarkupsFiducialNode
    entryPointFiducialNode: vtkMRMLMarkupsFiducialNode
    safetyDistance: Annotated[float, WithinRange(0.0, 50.0)] = 5.0

#
# MamriWidget
#
class MamriWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        
        self._animationTimer = None
        self._isPlaying = False
        
        self.trajectoryPath = None
        self.trajectoryKeyframes = None
        self._isExecuting = False 
        self.lastEstimatedPoseSteps = None

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/Mamri.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        
        if hasattr(self.ui, "applyButton"):
            self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        if hasattr(self.ui, "planTrajectoryButton"):
            self.ui.planTrajectoryButton.clicked.connect(self.onPlanHeuristicPathButton)
        if hasattr(self.ui, "drawFiducialsCheckBox"):
            self.ui.drawFiducialsCheckBox.connect("toggled(bool)", self.onDrawFiducialsCheckBoxToggled)
        if hasattr(self.ui, "drawModelsCheckBox"):
            self.ui.drawModelsCheckBox.connect("toggled(bool)", self.onDrawModelsCheckBoxToggled)
        if hasattr(self.ui, "saveBaseplateButton"):
            self.ui.saveBaseplateButton.connect("clicked(bool)", self.onSaveBaseplateButton)
        if hasattr(self.ui, "findEntryPointButton"):
            self.ui.findEntryPointButton.connect("clicked(bool)", self.onFindEntryPointButton)
        if hasattr(self.ui, "zeroRobotButton"):
            self.ui.zeroRobotButton.connect("clicked(bool)", self.onZeroRobotButton)
        if hasattr(self.ui, "drawDebugMarkersCheckBox"):
            self.ui.drawDebugMarkersCheckBox.connect("toggled(bool)", self.onDrawDebugMarkersCheckBoxToggled)
        if hasattr(self.ui, "trajectorySlider"):
            self.ui.trajectorySlider.valueChanged.connect(self.onTrajectorySliderChanged)
        if hasattr(self.ui, "playPauseButton"):
            self.ui.playPauseButton.clicked.connect(self.onPlayPauseButton)
        if hasattr(self.ui, "moveToPoseButton"):
            self.ui.moveToPoseButton.clicked.connect(self.onMoveToPoseButton)
        if hasattr(self.ui, "refreshPortsButton"):
            self.ui.refreshPortsButton.clicked.connect(self.onRefreshPortsButton)
        if hasattr(self.ui, "connectButton"):
            self.ui.connectButton.toggled.connect(self.onConnectButtonToggled)
        if hasattr(self.ui, "executeTrajectoryButton"):
            self.ui.executeTrajectoryButton.clicked.connect(self.onExecuteTrajectoryButton)
        if hasattr(self.ui, "stopTrajectoryButton"):
            self.ui.stopTrajectoryButton.clicked.connect(self.onStopTrajectoryButton)
        if hasattr(self.ui, "returnToZeroButton"):
            self.ui.returnToZeroButton.clicked.connect(self.onReturnToZeroButton)

        self._animationTimer = qt.QTimer()
        self._animationTimer.setInterval(50)
        self._animationTimer.timeout.connect(self.doAnimationStep)

        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = MamriLogic()
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.initializeParameterNode()

    def cleanup(self) -> None: 
        self.removeObservers()
        if self._animationTimer:
            self._animationTimer.stop()
        if self.logic and self.logic.is_robot_connected():
            self.logic.stop_trajectory_execution() 
            self.logic.disconnect_from_robot()

    def enter(self) -> None: 
        self.initializeParameterNode()
        self.onRefreshPortsButton()
        self._checkAllButtons()

    def exit(self) -> None:
        self.remove_parameter_node_observers()

    def onSceneStartClose(self, caller, event) -> None: self.setParameterNode(None)
    
    def onSceneEndClose(self, caller, event) -> None:
        if self.parent.isEntered: self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        self.logic = self.logic or MamriLogic()
        self.setParameterNode(self.logic.getParameterNode())
        
        if not self._parameterNode:
            return

        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode
        
        targetNode = slicer.mrmlScene.GetFirstNodeByName("Target")
        if targetNode and isinstance(targetNode, slicer.vtkMRMLMarkupsFiducialNode):
            if self._parameterNode.targetFiducialNode != targetNode:
                self._parameterNode.targetFiducialNode = targetNode
                logging.info("Automatically selected 'Target' fiducial node.")

    def remove_parameter_node_observers(self):
        if self._parameterNode:
            if self._parameterNodeGuiTag:
                self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
                self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkAllButtons)

    def setParameterNode(self, inputParameterNode: Optional[MamriParameterNode]) -> None:
        self.remove_parameter_node_observers()
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkAllButtons)
        self._checkAllButtons()
        
    def _checkAllButtons(self, caller=None, event=None) -> None:
        can_apply = self._parameterNode and self._parameterNode.inputVolume is not None
        if hasattr(self.ui, "applyButton"):
            self.ui.applyButton.enabled = can_apply
            self.ui.applyButton.toolTip = _("Run fiducial detection and robot model rendering.") if can_apply else _("Select an input volume node.")

        model_is_built = self.logic and self.logic.jointTransformNodes.get("Baseplate") is not None
        
        # The check for segmentationNode has been removed from this logic
        can_plan_base = (self._parameterNode and
                         self._parameterNode.targetFiducialNode and self._parameterNode.targetFiducialNode.GetNumberOfControlPoints() > 0 and
                         self._parameterNode.entryPointFiducialNode and self._parameterNode.entryPointFiducialNode.GetNumberOfControlPoints() > 0)
        
        if hasattr(self.ui, "planTrajectoryButton"):
            self.ui.planTrajectoryButton.enabled = can_plan_base and model_is_built
        
        if hasattr(self.ui, "zeroRobotButton"):
            self.ui.zeroRobotButton.enabled = model_is_built
            self.ui.zeroRobotButton.toolTip = _("Sets all robot joint angles to zero in the simulation only.") if model_is_built else _("Run 'Start robot pose estimation' first to build the model.")
        
        trajectory_is_planned = self.trajectoryPath is not None
        if hasattr(self.ui, "trajectorySlider"):
            self.ui.trajectorySlider.enabled = trajectory_is_planned
        if hasattr(self.ui, "playPauseButton"):
            self.ui.playPauseButton.enabled = trajectory_is_planned
        if hasattr(self.ui, "trajectoryStatusLabel"):
             if not self._isExecuting and self.trajectoryPath is None:
                self.ui.trajectoryStatusLabel.text = "(No trajectory planned)"

        is_connected = self.logic and self.logic.is_robot_connected()
        is_executing = self._isExecuting

        if hasattr(self.ui, "connectButton"):
            self.ui.connectButton.enabled = not is_executing
        if hasattr(self.ui, "refreshPortsButton"):
            self.ui.refreshPortsButton.enabled = not is_executing
        if hasattr(self.ui, "executeTrajectoryButton"):
            self.ui.executeTrajectoryButton.enabled = is_connected and (self.trajectoryKeyframes is not None) and not is_executing
        if hasattr(self.ui, "stopTrajectoryButton"):
            self.ui.stopTrajectoryButton.enabled = is_executing
        if hasattr(self.ui, "returnToZeroButton"):
            self.ui.returnToZeroButton.enabled = is_connected and not is_executing
        if hasattr(self.ui, "moveToPoseButton"):
            self.ui.moveToPoseButton.enabled = is_connected and not is_executing and (self.lastEstimatedPoseSteps is not None)

    def onApplyButton(self) -> None:
        if not self._parameterNode:
            slicer.util.errorDisplay("Parameter node is not initialized.")
            return
        
        # Reset previous pose when running a new estimation
        self.lastEstimatedPoseSteps = None
        self.ui.moveToPoseButton.enabled = False
        self.ui.poseStatusLabel.text = "Estimating..."
        slicer.app.processEvents()

        models_visible = self.ui.drawModelsCheckBox.isChecked() if hasattr(self.ui, "drawModelsCheckBox") else True
        markers_visible = self.ui.drawFiducialsCheckBox.isChecked() if hasattr(self.ui, "drawFiducialsCheckBox") else True
        
        # The process function now returns the calculated steps
        estimated_steps = self.logic.process(self._parameterNode, models_visible=models_visible, markers_visible=markers_visible)
        
        if estimated_steps is not None:
            self.lastEstimatedPoseSteps = estimated_steps
            self.ui.poseStatusLabel.text = f"Estimated Pose Steps: {estimated_steps.tolist()}"
        else:
            self.ui.poseStatusLabel.text = "Estimation failed. See logs for details."

        self._checkAllButtons()

    def onPlanHeuristicPathButton(self) -> None:
        if not self._parameterNode:
            slicer.util.errorDisplay("Parameter node is not initialized.")
            return

        if self._isPlaying:
            self.onPlayPauseButton()
        
        with slicer.util.MessageDialog("Planning...", "Generating heuristic path...") as dialog:
            slicer.app.processEvents()
            path, keyframes = self.logic.planHeuristicPath(self._parameterNode)

        # Store both the dense path (for visualization) and keyframes (for execution)
        self.trajectoryPath = path
        self.trajectoryKeyframes = keyframes

        if self.trajectoryPath:
            slicer.util.infoDisplay(f"Generated heuristic path with {len(self.trajectoryPath)} steps.")
            if hasattr(self.ui, "trajectorySlider"):
                self.ui.trajectorySlider.blockSignals(True)
                self.ui.trajectorySlider.maximum = len(self.trajectoryPath) - 1
                self.ui.trajectorySlider.value = 0
                self.ui.trajectorySlider.blockSignals(False)
            self.onTrajectorySliderChanged(0) # Manually trigger the update for the starting pose
        else:
            slicer.util.errorDisplay("Failed to generate heuristic path.")

        self._checkAllButtons()

    def onPlayPauseButton(self):
        if self._isPlaying:
            self._animationTimer.stop()
            self._isPlaying = False
            self.ui.playPauseButton.text = "Play"
        else:
            if self.ui.trajectorySlider.value == self.ui.trajectorySlider.maximum:
                self.ui.trajectorySlider.value = 0
            
            self._animationTimer.start()
            self._isPlaying = True
            self.ui.playPauseButton.text = "Pause"
            
    def onTrajectorySliderChanged(self, value):
        if self.trajectoryPath is None or value >= len(self.trajectoryPath):
            return
            
        self.logic.setRobotPose(self.trajectoryPath[value])
        if hasattr(self.ui, "trajectoryStatusLabel"):
            percent = (value / self.ui.trajectorySlider.maximum) * 100 if self.ui.trajectorySlider.maximum > 0 else 0
            self.ui.trajectoryStatusLabel.text = f"Path: {percent:.0f}%"

    def doAnimationStep(self):
        current_value = self.ui.trajectorySlider.value
        if current_value < self.ui.trajectorySlider.maximum:
            self.ui.trajectorySlider.value = current_value + 1
        else:
            self._animationTimer.stop()
            self._isPlaying = False
            self.ui.playPauseButton.text = "Play"

    def onFindEntryPointButton(self) -> None:
        if not self._parameterNode:
            slicer.util.errorDisplay("Parameter node is not initialized.")
            return
        
        if not (self._parameterNode.targetFiducialNode and self._parameterNode.targetFiducialNode.GetNumberOfControlPoints() > 0 and self._parameterNode.segmentationNode):
            slicer.util.errorDisplay("Please select a body segmentation and place a target marker first.")
            return
            
        self.logic.findAndSetEntryPoint(self._parameterNode)

    def onSaveBaseplateButton(self) -> None:
        if not self.logic:
            slicer.util.errorDisplay("Logic module is not initialized."); return
        baseplate_tf_node = self.logic.jointTransformNodes.get("Baseplate")
        if not baseplate_tf_node:
            slicer.util.errorDisplay("Baseplate has not been processed yet. Run fiducial detection first to establish its transform."); return
        self.logic.saveBaseplateTransform(baseplate_tf_node)
        slicer.util.infoDisplay(f"Baseplate transform saved successfully to node: '{self.logic.SAVED_BASEPLATE_TRANSFORM_NODE_NAME}'.")

    def onDrawFiducialsCheckBoxToggled(self, checked: bool) -> None:
        self.logic._toggle_mri_fiducials(checked)

    def onDrawDebugMarkersCheckBoxToggled(self, checked: bool):
        self.logic._toggle_debug_markers(checked)

    def onDrawModelsCheckBoxToggled(self, checked: bool) -> None:
        self.logic._toggle_robot_models(checked)

    def onZeroRobotButton(self) -> None:
        if self.logic:
            self.logic.zeroRobot()
    
    def onMoveToPoseButton(self):
        if self._isExecuting:
            slicer.util.warningDisplay("An action is already being executed.")
            return
        if self.lastEstimatedPoseSteps is None:
            slicer.util.errorDisplay("No pose has been estimated yet.")
            return

        self._isExecuting = True
        self._checkAllButtons()
        try:
            self.logic.move_to_specific_pose(
                target_pose_steps=self.lastEstimatedPoseSteps,
                status_label=self.ui.trajectoryStatusLabel
            )
        except Exception as e:
            slicer.util.errorDisplay(f"An unexpected error occurred during movement: {e}")
            logging.error(f"An unexpected error occurred during movement: {e}", exc_info=True)
        finally:
            self._isExecuting = False
            self.logic.stop_execution_flag = False
            self._checkAllButtons()
            if hasattr(self.ui, "trajectoryStatusLabel"):
                self.ui.trajectoryStatusLabel.text = "Movement to pose finished."

    def onRefreshPortsButton(self):
        if hasattr(self.ui, "serialPortComboBox"):
            self.ui.serialPortComboBox.clear()
            ports = self.logic.get_available_serial_ports()
            self.ui.serialPortComboBox.addItems(ports)
            if not ports:
                self.ui.serialPortComboBox.addItem("No ports found")

    def onConnectButtonToggled(self, checked):
        if not hasattr(self.ui, "serialPortComboBox"):
            return
            
        port = self.ui.serialPortComboBox.currentText
        
        if checked:
            if self.logic.connect_to_robot(port):
                self.ui.connectionStatusLabel.text = f"Connected to {port}"
                self.ui.connectButton.text = "Disconnect"
            else:
                self.ui.connectionStatusLabel.text = f"Failed to connect"
                self.ui.connectButton.setChecked(False)
        else:
            self.logic.disconnect_from_robot()
            self.ui.connectionStatusLabel.text = "Not Connected"
            self.ui.connectButton.text = "Connect"
        self._checkAllButtons()
        
    def onExecuteTrajectoryButton(self):
        if self._isExecuting:
            slicer.util.warningDisplay("A trajectory is already being executed.")
            return

        if not self.trajectoryKeyframes:
            slicer.util.errorDisplay("Please plan a path first.")
            return

        self._isExecuting = True
        self._checkAllButtons()  # Update button states (e.g., disable Execute, enable Stop)

        try:
            # Directly call the blocking logic function, passing the UI label for status updates
            self.logic.execute_trajectory_on_robot(
                self.trajectoryKeyframes,
                self.ui.trajectoryStatusLabel
            )
        except Exception as e:
            slicer.util.errorDisplay(f"An unexpected error occurred during execution: {e}")
            logging.error(f"An unexpected error occurred during execution: {e}", exc_info=True)
        finally:
            # Ensure UI state is reset regardless of how execution finished
            self._isExecuting = False
            self.logic.stop_execution_flag = False  # Reset the stop flag
            self._checkAllButtons()  # Revert button states
            if hasattr(self.ui, "trajectoryStatusLabel"):
                self.ui.trajectoryStatusLabel.text = "Execution Finished"
            
    def onReturnToZeroButton(self):
        if self._isExecuting:
            slicer.util.warningDisplay("A trajectory is already being executed.")
            return

        self._isExecuting = True
        self._checkAllButtons()

        try:
            # Call the new logic function
            self.logic.return_to_zero_position(
                status_label=self.ui.trajectoryStatusLabel
            )
        except Exception as e:
            slicer.util.errorDisplay(f"An unexpected error occurred during homing: {e}")
            logging.error(f"An unexpected error occurred during homing: {e}", exc_info=True)
        finally:
            self._isExecuting = False
            self.logic.stop_execution_flag = False
            self._checkAllButtons()
            if hasattr(self.ui, "trajectoryStatusLabel"):
                self.ui.trajectoryStatusLabel.text = "Homing Finished"

    def onStopTrajectoryButton(self):
        self.logic.stop_trajectory_execution()


#
# MamriLogic
#
class MamriLogic(ScriptedLoadableModuleLogic):

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)
        self.jointModelNodes: Dict[str, vtkMRMLModelNode] = {}
        self.jointTransformNodes: Dict[str, vtkMRMLLinearTransformNode] = {}
        self.jointFixedOffsetTransformNodes: Dict[str, vtkMRMLLinearTransformNode] = {}
        self.jointCollisionPolys: Dict[str, vtk.vtkPolyData] = {}

        self.INTENSITY_THRESHOLD = 65.0
        self.MIN_VOLUME_THRESHOLD = 150.0
        self.MAX_VOLUME_THRESHOLD = 1500.0
        self.DISTANCE_TOLERANCE = 3.0

        self.models_visible = True
        self.markers_visible = True
        self.debug_markers_visible = False

        self.robot_definition = self._define_robot_structure()
        self.robot_definition_dict = {joint["name"]: joint for joint in self.robot_definition}
        # Define the articulated chain in one place
        self.articulated_chain = ["Shoulder1", "Link1", "Shoulder2", "Elbow1", "Wrist", "End"]
        
        self.SAVED_BASEPLATE_TRANSFORM_NODE_NAME = "MamriSavedBaseplateTransform"
        self.TARGET_POSE_TRANSFORM_NODE_NAME = "MamriTargetPoseTransform_DEBUG"
        self.TRAJECTORY_LINE_NODE_NAME = "TrajectoryLine_DEBUG"

        self.MASTER_FOLDER_NAME = "MAMRI Robot Output"

        self.DEBUG_COLLISIONS = False 
        if self.DEBUG_COLLISIONS:
            logging.warning("MamriLogic collision debugging is enabled. This will create temporary models in the scene.")
        
        self.serial_connection = None
        self.execution_thread = None
        self.stop_execution_flag = False
        
        self._discover_robot_nodes_in_scene()
        
    def setRobotPose(self, joint_angles_rad: np.ndarray):
        """
        Sets the robot's joint angles to a specific configuration.
        This function is now robust to the shape of the input array.
        """
        pose_1d = np.asarray(joint_angles_rad).flatten()
        
        if pose_1d.shape[0] != len(self.articulated_chain):
            logging.error(f"setRobotPose received an array with incorrect dimensions: {pose_1d.shape}")
            return

        for j, joint_name in enumerate(self.articulated_chain):
            if node := self.jointTransformNodes.get(joint_name):
                angle_rad_scalar = float(pose_1d[j])
                angle_deg = math.degrees(angle_rad_scalar)
                
                transform = self._get_rotation_transform(angle_deg, self.robot_definition_dict[joint_name].get("articulation_axis"))
                node.SetMatrixTransformToParent(transform.GetMatrix())

    def planHeuristicPath(self, pNode: 'MamriParameterNode', total_steps=100) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
        """
        Generates a collision-avoidance path and logs detailed diagnostic information about each keyframe.
        Returns the dense path for visualization and the keyframes for robot control.
        """
        logging.info("Planning heuristic 'up, over, down' path...")

        start_config = np.array(self._get_current_joint_angles(self.articulated_chain))
        end_config = self.planTrajectory(pNode, solve_only=True)
        
        if end_config is None:
            logging.error("Heuristic planning failed: Could not determine a valid end configuration.")
            return None, None

        waypoint1_config = np.copy(start_config)
        waypoint1_config[1] = math.radians(-15)
        waypoint2_config = np.copy(waypoint1_config)
        waypoint2_config[0] = end_config[0]
        
        keyframes = [start_config, waypoint1_config, waypoint2_config, end_config]
        
        path = []
        segment_steps = [total_steps // 4, total_steps // 4, total_steps // 2]
        
        for i in range(len(keyframes) - 1):
            start_wp, end_wp = keyframes[i], keyframes[i+1]
            steps = segment_steps[i]
            is_last_segment = (i == len(keyframes) - 2)
            
            for j in range(steps):
                t = j / float(steps)
                path.append(start_wp + t * (end_wp - start_wp))
            
            if is_last_segment:
                path.append(end_wp)

        # Find segmentation node by name
        segmentationNode = slicer.mrmlScene.GetFirstNodeByName("AutoBodySegmentation")
        body_poly = self._get_body_polydata(segmentationNode) if segmentationNode else None

        base_tf = vtk.vtkMatrix4x4()
        if base_node := self.jointTransformNodes.get("Baseplate"):
            base_node.GetMatrixTransformToWorld(base_tf)
        
        if body_poly:
            for config in path:
                if self._check_collision(dict(zip(self.articulated_chain, config)), base_tf, body_poly):
                    logging.warning("Heuristic path resulted in a collision. The path may be unsafe.")
                    slicer.util.warningDisplay("Warning: The generated path results in a collision. Manual adjustment may be needed.")
                    break
        else:
            logging.warning("Could not find 'AutoBodySegmentation' for collision checking the path.")

        if keyframes:
            try:
                log_messages = ["\n" + "="*50, "PLANNED TRAJECTORY KEYFRAME DIAGNOSTICS".center(50), "="*50]
                
                for i, pose_rad in enumerate(keyframes):
                    pose_deg = np.round(np.degrees(pose_rad), 2)
                    pose_steps = self._convert_angles_to_steps_array(pose_rad)
                    
                    waypoint_name = "Start" if i == 0 else f"Waypoint {i}"
                    log_messages.append(f"\n--- {waypoint_name} Pose ---")
                    log_messages.append(f"  Target Angles (Deg): {pose_deg.tolist()}")
                    log_messages.append(f"  Target Steps:        {pose_steps.tolist()}")

                log_messages.append("\n" + "="*50 + "\n")
                logging.info("\n".join(log_messages))
                slicer.util.infoDisplay("A detailed keyframe plan has been printed to the Python Console.")
            except Exception as e:
                logging.error(f"Could not generate keyframe diagnostic plan: {e}")
        
        return path, keyframes
    
    def get_current_positions(self) -> Optional[List[int]]:
        """Sends 'P' to the robot and returns the current positions of all joints."""
        if not self.is_robot_connected():
            return None
        try:
            self.send_command_to_robot("P")
            response = self.serial_connection.readline().decode('ascii').strip()
            if not response:
                return None
            
            positions = [int(p.strip()) for p in response.split(',')]
            return positions
        except Exception as e:
            logging.warning(f"Could not get robot position. Error: {e}")
            return None

    def execute_trajectory_on_robot(self, keyframes: List[np.ndarray], status_label: qt.QLabel = None):
        """
        Executes a trajectory by moving the robot to each keyframe pose sequentially.
        This function is BLOCKING and treats each keyframe as a goal pose.
        """
        if not self.is_robot_connected():
            error_msg = "Execution failed: Robot not connected."
            if status_label: status_label.text = error_msg
            logging.error(error_msg)
            return

        self.stop_execution_flag = False

        def update_gui(message: str):
            if status_label:
                status_label.text = message
            slicer.app.processEvents()

        # 1. Synchronize with the robot's actual starting position
        update_gui("Reading initial robot position...")
        initial_robot_positions = self.get_current_positions()
        if initial_robot_positions is None:
            update_gui("FATAL: Could not read robot's initial position. Aborting.")
            logging.error("Aborting execution: Failed to get initial robot position.")
            return
        
        num_joints = len(self.articulated_chain)
        actual_start_steps = np.array(initial_robot_positions[:num_joints])

        # 2. Define the full execution path, including the actual start
        planned_steps = [self._convert_angles_to_steps_array(pose) for pose in keyframes]
        full_path_steps = [actual_start_steps] + planned_steps
        
        unique_path_steps = []
        for step_array in full_path_steps:
            if not unique_path_steps or not np.array_equal(unique_path_steps[-1], step_array):
                unique_path_steps.append(step_array)

        # --- NEW: DETAILED DIAGNOSTIC LOGGING ---
        try:
            log_messages = ["\n" + "="*51, "ROBOT TRAJECTORY DIAGNOSTICS".center(51), "="*51]
            
            # Log Initial State
            initial_angles_rad = self._convert_steps_array_to_angles(actual_start_steps)
            initial_angles_deg = np.round(np.degrees(initial_angles_rad), 2)
            log_messages.append("\n--- Initial State ---")
            log_messages.append(f"Actual Start (Steps): {actual_start_steps.tolist()}")
            log_messages.append(f"Actual Start (Deg):   {initial_angles_deg.tolist()}")

            # Log Waypoints
            log_messages.append("\n--- Trajectory Waypoints ---")
            # We skip unique_path_steps[0] as it's the actual start state, not a planned waypoint
            for i, waypoint_steps in enumerate(unique_path_steps[1:]):
                waypoint_rads = self._convert_steps_array_to_angles(waypoint_steps)
                waypoint_degs = np.round(np.degrees(waypoint_rads), 2)
                log_messages.append(f"Waypoint {i} (Target):")
                log_messages.append(f"  - Steps: {waypoint_steps.tolist()}")
                log_messages.append(f"  - Degs:  {waypoint_degs.tolist()}")
            
            log_messages.append("\n" + "="*51 + "\n")
            logging.info("\n".join(log_messages))
        except Exception as e:
            logging.error(f"Failed to generate diagnostic log: {e}")
        # --- END OF NEW LOGGING CODE ---

        if len(unique_path_steps) < 2:
            update_gui("Robot is already at the target position.")
            return

        # 3. Execute the synchronized path
        current_pose_steps = np.copy(actual_start_steps)
        for i, target_pose_steps in enumerate(unique_path_steps[1:]):
            if self.stop_execution_flag:
                update_gui("Execution stopped by user.")
                return

            logging.info(f"\n--- Moving to Keyframe {i+1}/{len(unique_path_steps)-1} ---")
            logging.info(f"  Target Pose: {target_pose_steps.tolist()}")

            timeout = 120.0
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if self.stop_execution_flag:
                    update_gui("Execution stopped by user.")
                    return
                
                live_positions = self.get_current_positions()
                if live_positions is None:
                    time.sleep(0.2)
                    continue
                
                live_pose_steps = np.array(live_positions[:num_joints])
                current_pose_steps = live_pose_steps

                pose_is_achieved = np.all(np.abs(live_pose_steps - target_pose_steps) <= 2)
                if pose_is_achieved:
                    logging.info(f"Keyframe {i+1} reached.")
                    break

                status_messages = []
                for joint_index in range(num_joints):
                    if abs(live_pose_steps[joint_index] - target_pose_steps[joint_index]) > 2:
                        joint_def = self.robot_definition_dict[self.articulated_chain[joint_index]]
                        command_letter = joint_def["command_letter"]
                        command = f"{command_letter}{target_pose_steps[joint_index]}"
                        self.send_command_to_robot(command)
                        status_messages.append(f"{command_letter}: {live_pose_steps[joint_index]}->{target_pose_steps[joint_index]}")

                update_gui("Moving... " + ", ".join(status_messages))
                
                live_pose_rad = self._convert_steps_array_to_angles(live_pose_steps)
                self.setRobotPose(live_pose_rad)
                
                time.sleep(0.1)

            else:
                error_msg = f"TIMEOUT: Robot failed to reach keyframe {i+1}."
                logging.error(error_msg)
                update_gui(error_msg)
                return

        final_pose_rad = self._convert_steps_array_to_angles(unique_path_steps[-1])
        self.setRobotPose(final_pose_rad)
        update_gui("Trajectory execution complete.")
        logging.info("Trajectory execution complete.")

    def _convert_angles_to_steps_array(self, joint_angles_rad: np.ndarray) -> np.ndarray:
        """Converts an array of joint angles (radians) into an array of motor steps."""
        steps_array = np.zeros(len(joint_angles_rad), dtype=int)
        for i, angle_rad in enumerate(np.asarray(joint_angles_rad).flatten()):
            joint_def = self.robot_definition_dict[self.articulated_chain[i]]
            steps_per_rev = joint_def.get("steps_per_rev", 0)
            if steps_per_rev > 0:
                steps_array[i] = int(angle_rad * (steps_per_rev / (2.0 * math.pi)))
        return steps_array

    def _convert_steps_array_to_angles(self, steps_array: np.ndarray) -> np.ndarray:
        """Converts an array of motor steps back to an array of joint angles (radians)."""
        angles_rad_array = np.zeros(len(steps_array), dtype=float)
        for i, steps in enumerate(np.asarray(steps_array).flatten()):
            angles_rad_array[i] = self._convert_steps_to_angle_rad(steps, i)
        return angles_rad_array

    def _convert_steps_to_angle_rad(self, steps: int, joint_index: int) -> float:
        """Converts motor steps for a specific joint back to radians."""
        joint_def = self.robot_definition_dict[self.articulated_chain[joint_index]]
        steps_per_rev = joint_def.get("steps_per_rev", 0)
        if steps_per_rev > 0:
            return float(steps) * ((2.0 * math.pi) / steps_per_rev)
        return 0.0

    def zeroRobot(self) -> None:
        if not self.jointTransformNodes:
            slicer.util.warningDisplay("Robot model has not been built. Please run 'Start robot pose estimation' first.")
            return

        logging.info("Setting robot pose to zero position for debugging...")
        identity_matrix = vtk.vtkMatrix4x4()
        
        for joint_def in self.robot_definition:
            joint_name = joint_def["name"]
            if joint_def.get("articulation_axis") and "TRANS" not in joint_def.get("articulation_axis"):
                tf_node = self.jointTransformNodes.get(joint_name)
                if tf_node:
                    tf_node.SetMatrixTransformToParent(identity_matrix)
        
        if needle_tf_node := self.jointTransformNodes.get("Needle"):
            needle_tf_node.SetMatrixTransformToParent(identity_matrix)

        logging.info("Robot pose has been reset to zero.")
        
        self._visualize_all_joint_markers_from_fk()
        logging.info("Updated debug marker positions to reflect zero pose.")

    def _organize_node_in_subject_hierarchy(self, mrmlNode, parentFolderName: str, subFolderName: Optional[str] = None):
        if not mrmlNode:
            return
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        if not shNode:
            logging.error("Subject Hierarchy node not found. Cannot organize scene.")
            return

        parentItemID = shNode.GetItemByName(parentFolderName)
        if not parentItemID:
            parentItemID = shNode.CreateFolderItem(shNode.GetSceneItemID(), parentFolderName)

        if subFolderName:
            childItems = vtk.vtkIdList()
            shNode.GetItemChildren(parentItemID, childItems)
            subFolderItemID = None
            for i in range(childItems.GetNumberOfIds()):
                itemID = childItems.GetId(i)
                if shNode.GetItemName(itemID) == subFolderName:
                    subFolderItemID = itemID
                    break
            
            if not subFolderItemID:
                subFolderItemID = shNode.CreateFolderItem(parentItemID, subFolderName)
            
            parentItemID = subFolderItemID

        nodeItemID = shNode.GetItemByDataNode(mrmlNode)
        if nodeItemID:
            shNode.SetItemParent(nodeItemID, parentItemID)

    def _discover_robot_nodes_in_scene(self):
        logging.info("Discovering existing robot nodes in the scene...")
        for joint_info in self.robot_definition:
            jn = joint_info["name"]
            
            model_node = slicer.mrmlScene.GetFirstNodeByName(f"{jn}Model")
            if model_node and isinstance(model_node, slicer.vtkMRMLModelNode):
                self.jointModelNodes[jn] = model_node
            
            tf_node = slicer.mrmlScene.GetFirstNodeByName(f"{jn}ArticulationTransform")
            if tf_node and isinstance(tf_node, slicer.vtkMRMLLinearTransformNode):
                self.jointTransformNodes[jn] = tf_node

        if self.jointTransformNodes:
            logging.info(f"Discovered and relinked {len(self.jointTransformNodes)} transform nodes.")

    @staticmethod
    def _create_offset_matrix(translations: Tuple[float,float,float] = (0,0,0),
                             rotations_deg: List[Tuple[str, float]] = None) -> vtk.vtkMatrix4x4:
        transform = vtk.vtkTransform()
        if rotations_deg:
            for axis, angle_deg in rotations_deg:
                getattr(transform, f"Rotate{axis.upper()}")(angle_deg)
        transform.Translate(translations)
        return transform.GetMatrix()

    def _define_robot_structure(self) -> List[Dict]:
        base_path = r"C:\Users\paul\Documents\UTwente\MSc ROB\MSc Thesis\CAD\Joints"
        return [
            {"name": "Baseplate", "stl_path": os.path.join(base_path, "Baseplate.STL"), "collision_stl_path": os.path.join(base_path, "Baseplate_collision.STL"), "parent": None, "fixed_offset_to_parent": None, "has_markers": True, "local_marker_coords": [(-10.0, 20.0, 5.0), (10.0, 20.0, 5.0), (-10.0, -20.0, 5.0)], "arm_lengths": (40.0, 20.0), "color": (1, 0, 0), "articulation_axis": None},
            {"name": "Shoulder1", "stl_path": os.path.join(base_path, "Shoulder1.STL"), "collision_stl_path": os.path.join(base_path, "Shoulder1_collision.STL"), "parent": "Baseplate", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 20.0)), "has_markers": False, "color": (0, 0.5, 0), "articulation_axis": "IS", "joint_limits": (-180, 180), "command_letter": "A", "steps_per_rev": 3332},
            {"name": "Link1", "stl_path": os.path.join(base_path, "Link1.STL"), "collision_stl_path": os.path.join(base_path, "Link1_collision.STL"), "parent": "Shoulder1", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 30)), "has_markers": True, "local_marker_coords": [(12.5, 45.0, 110.0), (-12.5, 45.0, 110.0), (12.5, 45.0, 40.0)], "arm_lengths": (70.0, 25.0), "color": (0, 1, 0), "articulation_axis": "PA", "joint_limits": (-120, 120), "command_letter": "B", "steps_per_rev": 3332},
            {"name": "Shoulder2", "stl_path": os.path.join(base_path, "Shoulder2.STL"), "collision_stl_path": os.path.join(base_path, "Shoulder2_collision.STL"), "parent": "Link1", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 150)), "has_markers": False, "color": (0, 0.5, 0), "articulation_axis": "PA", "joint_limits": (-120, 120), "command_letter": "C", "steps_per_rev": 3332},
            {"name": "Elbow1", "stl_path": os.path.join(base_path, "Elbow1.STL"), "collision_stl_path": os.path.join(base_path, "Elbow1_collision.STL"), "parent": "Shoulder2", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 0)), "has_markers": True, "local_marker_coords": [(-10, 35.0, 85), (10, 35.0, 85), (-10, -35.0, 85)], "arm_lengths": (70.0, 20.0),  "color": (0, 1, 0), "articulation_axis": "IS", "joint_limits": (-180, 180), "command_letter": "D", "steps_per_rev": 3332},
            {"name": "Wrist", "stl_path": os.path.join(base_path, "Wrist.STL"), "collision_stl_path": os.path.join(base_path, "Wrist_collision.STL"), "parent": "Elbow1", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 150)), "has_markers": False, "color": (0, 0.5, 0), "articulation_axis": "PA", "joint_limits": (-120, 120), "command_letter": "E", "steps_per_rev": 3332},
            {"name": "End", "stl_path": os.path.join(base_path, "End.STL"), "collision_stl_path": os.path.join(base_path, "End_collision.STL"), "parent": "Wrist", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 8)), "has_markers": True, "local_marker_coords": [(-10, 22.5, 26), (10, 22.5, 26), (-10, -22.5, 26)], "arm_lengths": (45.0, 20.0), "color": (1, 0, 0), "articulation_axis": "IS", "joint_limits": (-270, 270), "command_letter": "F", "steps_per_rev": 3332},
            {"name": "Needle", "stl_path": os.path.join(base_path, "Needle.STL"), "collision_stl_path": os.path.join(base_path, "Needle_collision.STL"), "parent": "End", "fixed_offset_to_parent": self._create_offset_matrix((-50, 0, 71)), "has_markers": False, "color": (1, 0, 0), "articulation_axis": "TRANS_X", "joint_limits": (0, 0), "needle_tip_local": (0, 0, 0), "needle_axis_local": (1, 0, 0)}
        ]
        
    def get_available_serial_ports(self) -> List[str]:
        """Returns a list of available serial port device names."""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]

    def connect_to_robot(self, port: str) -> bool:
        """Establishes a serial connection with the robot controller."""
        if self.is_robot_connected():
            logging.info("A connection is already active. Disconnecting first.")
            self.disconnect_from_robot()
        try:
            self.serial_connection = serial.Serial(port, 115200, timeout=1, write_timeout=1)
            logging.info(f"Successfully connected to robot on port {port}.")
            return True
        except serial.SerialException as e:
            logging.error(f"Failed to connect to robot on port {port}: {e}")
            self.serial_connection = None
            return False

    def disconnect_from_robot(self) -> None:
        """Closes the serial connection."""
        if self.serial_connection and self.serial_connection.is_open:
            port = self.serial_connection.port
            self.serial_connection.close()
            logging.info(f"Disconnected from serial port {port}.")
        self.serial_connection = None

    def is_robot_connected(self) -> bool:
        """Checks if the serial connection to the robot is active."""
        return self.serial_connection is not None and self.serial_connection.is_open

    def send_command_to_robot(self, command: str) -> bool:
        """Sends a raw command string to the robot."""
        if not self.is_robot_connected():
            logging.warning(f"Cannot send command '{command}': Robot not connected.")
            return False
        try:
            # logging.debug(f"Sending command: {command}") # This can be too verbose
            self.serial_connection.write(command.encode('ascii'))
            return True
        except Exception as e:
            logging.error(f"Failed to send command '{command}': {e}")
            return False

    def _convert_angles_to_steps_command(self, joint_angles_rad: np.ndarray) -> str:
        """Converts an array of joint angles into a robot command string."""
        command_string = ""
        pose_1d = np.asarray(joint_angles_rad).flatten()
        for i, joint_name in enumerate(self.articulated_chain):
            joint_def = self.robot_definition_dict.get(joint_name, {})
            command_letter = joint_def.get("command_letter")
            steps_per_rev = joint_def.get("steps_per_rev")

            if command_letter and steps_per_rev:
                angle_rad = pose_1d[i]
                steps = int(angle_rad * (steps_per_rev / (2.0 * math.pi)))
                command_string += f"{command_letter}{steps}"
        
        return command_string
        
    def send_pose_to_robot(self, joint_angles_rad: np.ndarray) -> bool:
        """Converts and sends a single robot pose over serial."""
        command = self._convert_angles_to_steps_command(joint_angles_rad)
        if command:
            return self.send_command_to_robot(command)
        logging.warning("Could not generate a command from the provided joint angles.")
        return False
        
    def stop_trajectory_execution(self):
        """
        Signals the trajectory execution loop to stop and sends the robot's
        current position as its new setpoint to perform a controlled stop.
        """
        logging.info("STOP command received. Halting robot motion...")
        
        # 1. Set the flag to stop any Python-side execution loops.
        self.stop_execution_flag = True

        # 2. Check for an active connection.
        if not self.is_robot_connected():
            logging.warning("Cannot send stop command: Robot not connected.")
            return

        # 3. Get the robot's current position.
        current_positions = self.get_current_positions()
        if current_positions is None:
            logging.error("Failed to get current position to send stop command. Motor may not stop.")
            return

        # 4. Send the current position of each joint back as its new target.
        logging.info(f"Sending current positions {current_positions} as new setpoints to halt motion.")
        for i, pos in enumerate(current_positions):
            if i < len(self.articulated_chain):
                joint_name = self.articulated_chain[i]
                joint_def = self.robot_definition_dict.get(joint_name, {})
                command_letter = joint_def.get("command_letter")
                
                if command_letter:
                    command = f"{command_letter}{pos}"
                    self.send_command_to_robot(command)
        
        logging.info("Halt commands sent.")
        
    def getParameterNode(self) -> 'MamriParameterNode':
        return MamriParameterNode(super().getParameterNode())

    def _clear_node_by_name(self, name: str):
        while node := slicer.mrmlScene.GetFirstNodeByName(name):
            slicer.mrmlScene.RemoveNode(node)

    def process(self, parameterNode: 'MamriParameterNode', models_visible: bool, markers_visible: bool) -> Optional[np.ndarray]:
        self.models_visible = models_visible
        self.markers_visible = markers_visible
        
        with slicer.util.MessageDialog("Processing...", "Detecting fiducials and building robot model...") as dialog:
            slicer.app.processEvents()
            self._cleanup_module_nodes()
            self.volume_threshold_segmentation(parameterNode)
            identified_joints_data = self.joint_detection(parameterNode)
            self._handle_joint_detection_results(identified_joints_data)
            baseplate_transform_matrix = vtk.vtkMatrix4x4()
            baseplate_transform_found = False
            if parameterNode.useSavedBaseplate:
                saved_tf_node = slicer.mrmlScene.GetFirstNodeByName(self.SAVED_BASEPLATE_TRANSFORM_NODE_NAME)
                if saved_tf_node and isinstance(saved_tf_node, vtkMRMLLinearTransformNode):
                    saved_tf_node.GetMatrixTransformToWorld(baseplate_transform_matrix)
                    baseplate_transform_found = True
                else:
                    slicer.util.warningDisplay(f"'Use Saved Transform' is checked, but node '{self.SAVED_BASEPLATE_TRANSFORM_NODE_NAME}' was not found.")
            if not baseplate_transform_found:
                if "Baseplate" in identified_joints_data:
                    baseplate_def = self.robot_definition_dict["Baseplate"]
                    alignment_matrix = self._calculate_fiducial_alignment_matrix("BaseplateFiducials", baseplate_def["local_marker_coords"])
                    if alignment_matrix:
                        baseplate_transform_matrix.DeepCopy(alignment_matrix)
                        baseplate_transform_found = True
                    else: logging.warning("Could not calculate baseplate transform from fiducials.")
                else: logging.info("Baseplate fiducials not detected and not using a saved transform. Baseplate will be at origin.")
            
            self._build_robot_model(baseplate_transform_matrix)
            
            apply_correction = parameterNode.applyEndEffectorCorrection

            final_pose_angles = None
            if baseplate_transform_found and "End" in identified_joints_data:
                end_fiducials_node = slicer.mrmlScene.GetFirstNodeByName("EndFiducials")
                if end_fiducials_node and end_fiducials_node.GetNumberOfControlPoints() == 3:
                    # _solve_full_chain_ik now returns the angles
                    final_pose_angles = self._solve_full_chain_ik(end_fiducials_node, apply_correction)
                    if final_pose_angles is not None:
                        self.setRobotPose(final_pose_angles)
                        self._visualize_all_joint_markers_from_fk()
            else:
                logging.info("Prerequisites for full-chain IK not met. Cannot estimate pose.")

            logging.info("Mamri processing finished.")
            
            # If a pose was successfully calculated, convert to steps and return it
            if final_pose_angles is not None:
                return self._convert_angles_to_steps_array(final_pose_angles)
            
            return None
        
    def _get_body_polydata(self, segmentationNode: vtkMRMLSegmentationNode) -> Optional[vtk.vtkPolyData]:
        if not segmentationNode:
            return None
        segmentation = segmentationNode.GetSegmentation()
        bodySegmentID = next((segmentation.GetSegmentIdBySegmentName(s.GetName()) for i in range(segmentation.GetNumberOfSegments()) if (s := segmentation.GetNthSegment(i)) and "Body" in s.GetName()), None)
        if not bodySegmentID:
            slicer.util.errorDisplay("Could not find a segment named 'Body' in the selected segmentation.")
            return None
        body_poly = vtk.vtkPolyData()
        try:
            representationName = slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName()
            if not segmentation.ContainsRepresentation(representationName):
                segmentation.CreateRepresentation(representationName)
            segmentationNode.GetClosedSurfaceRepresentation(bodySegmentID, body_poly)
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to get surface representation from segmentation: {e}")
            return None
        if body_poly.GetNumberOfPoints() == 0:
            return None
        return body_poly

    def findAndSetEntryPoint(self, pNode: 'MamriParameterNode') -> None:
        targetNode = pNode.targetFiducialNode
        # Find the segmentation node by name instead of using the parameter node
        segmentationNode = slicer.mrmlScene.GetFirstNodeByName("AutoBodySegmentation")
        
        if not (targetNode and targetNode.GetNumberOfControlPoints() > 0 and segmentationNode):
            slicer.util.errorDisplay("Please place a target marker and ensure 'AutoBodySegmentation' exists (run pose estimation).")
            return
            
        body_poly = self._get_body_polydata(segmentationNode)
        if not body_poly:
            return
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(body_poly)
        normals.ComputePointNormalsOn()
        normals.SplittingOff()
        normals.Update()
        polydata_with_normals = normals.GetOutput()
        point_normals_array = polydata_with_normals.GetPointData().GetNormals()
        point_locator = vtk.vtkStaticPointLocator()
        point_locator.SetDataSet(polydata_with_normals)
        point_locator.BuildLocator()
        target_pos = targetNode.GetNthControlPointPositionWorld(0)
        result_point_ids = vtk.vtkIdList()
        search_radius = 80.0
        point_locator.FindPointsWithinRadius(search_radius, target_pos, result_point_ids)
        up_vector = np.array([0, 1, 0])
        normal_threshold = 0.2 
        candidate_points = []
        for i in range(result_point_ids.GetNumberOfIds()):
            point_id = result_point_ids.GetId(i)
            point_normal = np.array(point_normals_array.GetTuple(point_id))
            if np.dot(point_normal, up_vector) > normal_threshold:
                point_coords = np.array(polydata_with_normals.GetPoint(point_id))
                distance = np.linalg.norm(point_coords - np.array(target_pos))
                candidate_points.append({"coords": point_coords, "distance": distance})
        if not candidate_points:
            slicer.util.warningDisplay(f"Could not find a suitable entry point on an accessible surface within a {search_radius}mm radius of the target.")
            return
        best_candidate = min(candidate_points, key=lambda x: x["distance"])
        closest_point_coords = best_candidate["coords"]
        optimalEntryPointNodeName = "OptimalEntryPoint"
        if oldEntryNode := slicer.mrmlScene.GetFirstNodeByName(optimalEntryPointNodeName):
            slicer.mrmlScene.RemoveNode(oldEntryNode)
        entryNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", optimalEntryPointNodeName)
        entryNode.GetDisplayNode().SetSelectedColor(1.0, 1.0, 0.0)
        entryNode.AddControlPointWorld(vtk.vtkVector3d(closest_point_coords))
        entryNode.SetNthControlPointLabel(0, "Optimal Entry (Accessible)")
        pNode.entryPointFiducialNode = entryNode
        slicer.util.infoDisplay("Accessible optimal entry point has been calculated and set.")

    def _ik_pose_error_function(self, angles_rad, articulated_joint_names, target_transform, base_transform):
        joint_values_rad = {name: angle for name, angle in zip(articulated_joint_names, angles_rad)}
        fk_transform = self._get_world_transform_for_joint(joint_values_rad, "Needle", base_transform)
        if fk_transform is None:
            return [1e6] * 6 
        fk_mat = np.array([fk_transform.GetElement(i, j) for i in range(4) for j in range(4)]).reshape(4, 4)
        target_mat = np.array([target_transform.GetElement(i, j) for i in range(4) for j in range(4)]).reshape(4, 4)
        pos_error = fk_mat[:3, 3] - target_mat[:3, 3]
        fk_x_axis = fk_mat[:3, 0]
        target_x_axis = target_mat[:3, 0]
        actual_needle_direction = -fk_x_axis
        orientation_error = 50 * (target_x_axis - actual_needle_direction)
        return np.concatenate((pos_error, orientation_error)).tolist()
    
    def _check_collision(self, joint_angles_rad: Dict[str, float], base_transform_vtk: vtk.vtkMatrix4x4, body_polydata: vtk.vtkPolyData) -> bool:
        if not body_polydata or body_polydata.GetNumberOfPoints() == 0:
            return False
        parts_to_check = ["Shoulder1", "Link1", "Shoulder2", "Elbow1", "Wrist", "End"]
        collision_detector = vtk.vtkCollisionDetectionFilter()
        identity_matrix = vtk.vtkMatrix4x4()
        collision_detector.SetInputData(1, body_polydata)
        collision_detector.SetMatrix(1, identity_matrix)
        for part_name in parts_to_check:
            robot_part_local_poly = self.jointCollisionPolys.get(part_name)
            if not robot_part_local_poly:
                continue 
            part_world_transform_vtk = self._get_world_transform_for_joint(joint_angles_rad, part_name, base_transform_vtk)
            if not part_world_transform_vtk: continue
            collision_detector.SetInputData(0, robot_part_local_poly)
            collision_detector.SetMatrix(0, part_world_transform_vtk)
            collision_detector.Update()
            if collision_detector.GetNumberOfContacts() > 0:
                return True
        return False

    def _ik_pose_and_collision_error_function(self, angles_rad, articulated_joint_names, target_transform, base_transform, body_polydata):
        joint_values_rad = {name: angle for name, angle in zip(articulated_joint_names, angles_rad)}
        if self._check_collision(joint_values_rad, base_transform, body_polydata):
            return [1e4] * 6
        return self._ik_pose_error_function(angles_rad, articulated_joint_names, target_transform, base_transform)

    def planTrajectory(self, pNode: 'MamriParameterNode', solve_only=False) -> Optional[np.ndarray]:
        if not solve_only:
            logging.info("Starting trajectory goal calculation...")
        
        targetNode, entryNode = pNode.targetFiducialNode, pNode.entryPointFiducialNode
        # Find the segmentation node by name
        segmentationNode = slicer.mrmlScene.GetFirstNodeByName("AutoBodySegmentation")

        if not (targetNode and targetNode.GetNumberOfControlPoints() > 0 and entryNode and entryNode.GetNumberOfControlPoints() > 0 and segmentationNode):
            if not solve_only: slicer.util.errorDisplay("Set target, entry markers, and ensure 'AutoBodySegmentation' exists to plan trajectory.")
            return None
            
        body_polydata = self._get_body_polydata(segmentationNode)
        if not body_polydata:
            if not solve_only: slicer.util.errorDisplay("Could not get body polydata from 'AutoBodySegmentation'. Aborting.")
            return None
            
        target_pos, entry_pos = np.array(targetNode.GetNthControlPointPositionWorld(0)), np.array(entryNode.GetNthControlPointPositionWorld(0))
        direction_vec = target_pos - entry_pos
        if np.linalg.norm(direction_vec) < 1e-6:
            if not solve_only: slicer.util.errorDisplay("Entry and Target markers are at the same position.")
            return None
            
        x_axis = direction_vec / np.linalg.norm(direction_vec)
        needle_tip_pos = entry_pos - (pNode.safetyDistance * x_axis)
        if not solve_only:
            line_node = self._visualize_trajectory_line(target_pos, needle_tip_pos)
            if line_node:
                self._organize_node_in_subject_hierarchy(line_node, self.MASTER_FOLDER_NAME, "Trajectory Plan")
                
        up_vec = np.array([0, 0, 1.0]); 
        if abs(np.dot(x_axis, up_vec)) > 0.99: up_vec = np.array([0, 1.0, 0])
        y_axis = np.cross(up_vec, x_axis); y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis); target_matrix_np = np.identity(4)
        target_matrix_np[:3, 0] = x_axis; target_matrix_np[:3, 1] = y_axis; target_matrix_np[:3, 2] = z_axis; target_matrix_np[:3, 3] = needle_tip_pos
        target_transform_vtk = vtk.vtkMatrix4x4(); target_transform_vtk.DeepCopy(target_matrix_np.flatten())
        chain_defs = [self.robot_definition_dict[name] for name in self.articulated_chain]
        bounds_rad = [tuple(math.radians(l) for l in jdef["joint_limits"]) for jdef in chain_defs]
        bounds_lower, bounds_upper = zip(*bounds_rad)
        base_node = self.jointTransformNodes.get("Baseplate")
        if not base_node:
            if not solve_only: slicer.util.errorDisplay("Robot model not loaded or baseplate is missing.")
            return None
            
        base_transform_vtk = vtk.vtkMatrix4x4(); base_node.GetMatrixTransformToWorld(base_transform_vtk)
        current_angles_rad = np.array(self._get_current_joint_angles(self.articulated_chain))
        initial_guesses = [current_angles_rad, [0.0] * len(self.articulated_chain)]
        best_result, lowest_error = None, float('inf')
        for initial_guess in initial_guesses:
            try: 
                result = scipy.optimize.least_squares(
                    self._ik_pose_and_collision_error_function, initial_guess, bounds=(bounds_lower, bounds_upper), 
                    args=(self.articulated_chain, target_transform_vtk, base_transform_vtk, body_polydata), 
                    method='trf', ftol=1e-4, xtol=1e-4, max_nfev=200)
                final_error = np.linalg.norm(self._ik_pose_error_function(result.x, self.articulated_chain, target_transform_vtk, base_transform_vtk))
                if result.success and final_error < lowest_error:
                    lowest_error = final_error
                    best_result = result
            except Exception as e: 
                logging.warning(f"IK optimization failed for one guess: {e}")
                
        if not best_result: 
            if not solve_only: slicer.util.errorDisplay("Could not find a valid, collision-free trajectory solution.")
            return None
            
        return np.array(best_result.x)
       
    def _load_collision_models(self):
        logging.info("Loading robot collision models...")
        self.jointCollisionPolys.clear()
        for joint_info in self.robot_definition:
            jn = joint_info["name"]
            collision_path = joint_info.get("collision_stl_path")
            polydata = None
            if collision_path and os.path.exists(collision_path):
                try:
                    reader = vtk.vtkSTLReader()
                    reader.SetFileName(collision_path)
                    reader.Update()
                    polydata = reader.GetOutput()
                except Exception as e:
                    logging.error(f"Failed to load collision STL '{collision_path}' for {jn}: {e}")
            if not polydata:
                if model_node := self.jointModelNodes.get(jn):
                    if model_node.GetPolyData():
                        polydata = model_node.GetPolyData()
            if polydata:
                self.jointCollisionPolys[jn] = polydata

    def _build_robot_model(self, baseplate_transform_matrix: vtk.vtkMatrix4x4):
        for joint_info in self.robot_definition:
            jn = joint_info["name"]
            stl_path = joint_info["stl_path"]
            if not stl_path or not os.path.exists(stl_path): continue
            model_matrix = baseplate_transform_matrix if jn == "Baseplate" else vtk.vtkMatrix4x4()
            modelNode, art_tf_node = self._create_model_and_articulation_transform(jn, stl_path, model_matrix, joint_info.get("color"))
            if not art_tf_node: continue
            self._organize_node_in_subject_hierarchy(art_tf_node, self.MASTER_FOLDER_NAME, "Robot Model")
            if modelNode:
                self._organize_node_in_subject_hierarchy(modelNode, self.MASTER_FOLDER_NAME, "Robot Model")
            if parent_name := joint_info.get("parent"):
                if parent_art_node := self.jointTransformNodes.get(parent_name):
                    offset_matrix = joint_info.get("fixed_offset_to_parent", vtk.vtkMatrix4x4())
                    fixed_offset_node_name = f"{parent_name}To{jn}FixedOffset"
                    fixed_offset_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", fixed_offset_node_name)
                    self._organize_node_in_subject_hierarchy(fixed_offset_node, self.MASTER_FOLDER_NAME, "Robot Model")
                    fixed_offset_node.SetMatrixTransformToParent(offset_matrix)
                    fixed_offset_node.SetAndObserveTransformNodeID(parent_art_node.GetID())
                    self.jointFixedOffsetTransformNodes[jn] = fixed_offset_node
                    art_tf_node.SetAndObserveTransformNodeID(fixed_offset_node.GetID())
        self._load_collision_models()

    def _get_rotation_transform(self, angle_deg: float, axis_str: Optional[str]) -> vtk.vtkTransform:
        transform = vtk.vtkTransform()
        if axis_str == "IS": transform.RotateZ(angle_deg)
        elif axis_str == "PA": transform.RotateY(angle_deg)
        elif axis_str == "LR": transform.RotateX(angle_deg)
        return transform

    def _get_world_transform_for_joint(self, joint_angles_rad: Dict[str, float], target_joint_name: str, base_transform_matrix: vtk.vtkMatrix4x4) -> Optional[vtk.vtkMatrix4x4]:
        world_transforms = {}
        for joint_def in self.robot_definition:
            name = joint_def["name"]
            parent_name = joint_def.get("parent")
            parent_world_tf = base_transform_matrix if not parent_name else world_transforms.get(parent_name)
            if parent_world_tf is None and parent_name is not None: return None
            fixed_offset_tf = joint_def.get("fixed_offset_to_parent") or vtk.vtkMatrix4x4()
            art_tf = vtk.vtkMatrix4x4()
            if axis := joint_def.get("articulation_axis"):
                angle_rad = joint_angles_rad.get(name, 0.0)
                if "TRANS" not in axis: art_tf = self._get_rotation_transform(math.degrees(angle_rad), axis).GetMatrix()
            local_tf = vtk.vtkMatrix4x4(); vtk.vtkMatrix4x4.Multiply4x4(fixed_offset_tf, art_tf, local_tf)
            current_world_tf = vtk.vtkMatrix4x4()
            if parent_world_tf: vtk.vtkMatrix4x4.Multiply4x4(parent_world_tf, local_tf, current_world_tf)
            else: current_world_tf.DeepCopy(local_tf)
            world_transforms[name] = current_world_tf
            if name == target_joint_name: return world_transforms[name]
        return world_transforms.get(target_joint_name)

    def _full_chain_ik_error_function(self, angles_rad, articulated_joint_names, end_target_ras, base_transform, end_def, apply_correction: bool, elbow_target_ras=None, elbow_def=None, elbow_weight=0.05):
        joint_values_rad = {name: angle for name, angle in zip(articulated_joint_names, angles_rad)}
        end_local_coords = list(end_def["local_marker_coords"])
        if apply_correction:
            rotation_transform = vtk.vtkTransform()
            rotation_transform.RotateZ(180)
            end_local_coords = [rotation_transform.TransformPoint(p) for p in end_local_coords]
        tf_end_model_to_world = self._get_world_transform_for_joint(joint_values_rad, end_def["name"], base_transform)
        if tf_end_model_to_world is None:
            num_errors = len(end_local_coords) * 3
            if elbow_target_ras: num_errors += len(elbow_target_ras) * 3
            return [1e6] * num_errors
        end_errors = []
        for i, local_p in enumerate(end_local_coords):
            pred_p_h = tf_end_model_to_world.MultiplyPoint(list(local_p) + [1.0])
            pred_ras = [c / pred_p_h[3] for c in pred_p_h[:3]]
            end_errors.extend([pred_ras[j] - end_target_ras[i][j] for j in range(3)])
        elbow_errors = []
        if elbow_target_ras and elbow_def:
            elbow_local_coords = elbow_def["local_marker_coords"]
            tf_elbow_model_to_world = self._get_world_transform_for_joint(joint_values_rad, elbow_def["name"], base_transform)
            if tf_elbow_model_to_world:
                for i, local_p in enumerate(elbow_local_coords):
                    pred_p_h = tf_elbow_model_to_world.MultiplyPoint(list(local_p) + [1.0])
                    pred_ras = [c / pred_p_h[3] for c in pred_p_h[:3]]
                    elbow_errors.extend([elbow_weight * (pred_ras[j] - elbow_target_ras[i][j]) for j in range(3)])
            else:
                elbow_errors = [1e4] * len(elbow_local_coords) * 3
        return end_errors + elbow_errors

    def _solve_full_chain_ik(self, end_effector_target_node: vtkMRMLMarkupsFiducialNode, apply_correction: bool) -> Optional[np.ndarray]:
        chain_defs = [self.robot_definition_dict[name] for name in self.articulated_chain]
        end_def = self.robot_definition_dict["End"]
        end_target_mri = [end_effector_target_node.GetNthControlPointPositionWorld(i) for i in range(3)]
        base_node = self.jointTransformNodes.get("Baseplate")
        if not base_node:
            logging.error("Full-chain IK requires the Baseplate transform node."); return None
        tf_base_to_world = vtk.vtkMatrix4x4(); base_node.GetMatrixTransformToWorld(tf_base_to_world)
        elbow_target_mri, elbow_def = None, None
        elbow_fiducials_node = slicer.mrmlScene.GetFirstNodeByName("Elbow1Fiducials")
        if elbow_fiducials_node and elbow_fiducials_node.GetNumberOfControlPoints() == 3:
            logging.info("Elbow markers detected. Adding as a secondary objective to the IK solver.")
            elbow_def = self.robot_definition_dict["Elbow1"]
            elbow_target_mri = [elbow_fiducials_node.GetNthControlPointPositionWorld(i) for i in range(3)]
        initial_guesses = [self._get_current_joint_angles(self.articulated_chain), [0.0] * len(self.articulated_chain)]
        best_result, lowest_cost = None, float('inf')
        for i, initial_guess in enumerate(initial_guesses):
            try:
                result = scipy.optimize.least_squares(
                    self._full_chain_ik_error_function, initial_guess, bounds=([math.radians(j["joint_limits"][0]) for j in chain_defs], [math.radians(j["joint_limits"][1]) for j in chain_defs]),
                    args=(self.articulated_chain, end_target_mri, tf_base_to_world, end_def, apply_correction, elbow_target_mri, elbow_def),
                    method='trf', ftol=1e-6, xtol=1e-6, verbose=0)
                if result.success and result.cost < lowest_cost:
                    lowest_cost = result.cost
                    best_result = result
            except Exception as e:
                logging.error(f"Scipy optimization for attempt #{i+1} failed: {e}")
        if not best_result:
            logging.error("Full-chain IK failed to converge for all initial guesses.")
            return None # Return None on failure
            
        final_angles_rad = best_result.x
        self._log_ik_solution_details(final_angles_rad, self.articulated_chain, tf_base_to_world, end_def, end_target_mri, apply_correction, elbow_def, elbow_target_mri)
        
        return final_angles_rad # Return the final angles on success
    
    def saveBaseplateTransform(self, current_baseplate_transform_node: vtkMRMLLinearTransformNode):
        self._clear_node_by_name(self.SAVED_BASEPLATE_TRANSFORM_NODE_NAME)
        saved_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", self.SAVED_BASEPLATE_TRANSFORM_NODE_NAME)
        world_matrix = vtk.vtkMatrix4x4()
        current_baseplate_transform_node.GetMatrixTransformToWorld(world_matrix)
        saved_node.SetMatrixTransformToParent(world_matrix)
        saved_node.SetSelectable(False)
        self._organize_node_in_subject_hierarchy(saved_node, self.MASTER_FOLDER_NAME, "Saved Transforms")
        
    def _log_ik_solution_details(self, final_angles_rad, articulated_chain, base_transform, end_def, end_target_ras, apply_correction: bool, elbow_def=None, elbow_target_ras=None):
        logging.info("--- IK Solution Details ---")
        joint_values_rad = {name: angle for name, angle in zip(articulated_chain, final_angles_rad)}
        logging.info("Final Joint Angles (°):")
        for name, angle_deg in zip(articulated_chain, [math.degrees(a) for a in final_angles_rad]):
            logging.info(f"  - {name}: {angle_deg:.2f}°")
        # ... (rest of logging details) ...

    def _visualize_joint_local_markers_in_world(self, joint_name: str):
        debug_node_name = f"{joint_name}_LocalMarkers_WorldView_DEBUG"; self._clear_node_by_name(debug_node_name)
        joint_def = self.robot_definition_dict.get(joint_name); joint_art_node = self.jointTransformNodes.get(joint_name)
        local_coords = joint_def.get("local_marker_coords") if joint_def else None
        if not all([joint_def, joint_art_node, local_coords]): return
        tf_model_to_world = vtk.vtkMatrix4x4(); joint_art_node.GetMatrixTransformToWorld(tf_model_to_world)
        debug_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", debug_node_name)
        self._organize_node_in_subject_hierarchy(debug_node, self.MASTER_FOLDER_NAME, "Debug Markers")
        if disp := debug_node.GetDisplayNode():
            # Use the new, separate visibility flag
            disp.SetVisibility(self.debug_markers_visible)
            disp.SetGlyphScale(3.0); disp.SetTextScale(3.5); r,g,b = joint_def.get("color", (0.1,0.8,0.8))
            disp.SetSelectedColor(r*0.7, g*0.7, b*0.7); disp.SetColor(r,g,b); disp.SetOpacity(1)
        prefix = "".join(w[0] for w in joint_name.split() if w)[:3].upper()
        for i, local_p in enumerate(local_coords):
            world_p_h = tf_model_to_world.MultiplyPoint(list(local_p) + [1.0]); world_ras = [c/world_p_h[3] for c in world_p_h[:3]]
            idx = debug_node.AddControlPoint(world_ras); debug_node.SetNthControlPointLabel(idx, f"{prefix}_Lm{i+1}")

    def _get_current_joint_angles(self, articulated_joint_names: List[str]) -> List[float]:
        angles_rad = []
        for name in articulated_joint_names:
            angle_rad = 0.0
            if tf_node := self.jointTransformNodes.get(name):
                m = vtk.vtkMatrix4x4()
                tf_node.GetMatrixTransformToParent(m)
                transform = vtk.vtkTransform(); transform.SetMatrix(m)
                orientation_rad = [math.radians(a) for a in transform.GetOrientation()]
                axis = self.robot_definition_dict[name].get("articulation_axis")
                if axis == "IS": angle_rad = orientation_rad[2]
                elif axis == "PA": angle_rad = orientation_rad[1]
                elif axis == "LR": angle_rad = orientation_rad[0]
            angles_rad.append(angle_rad)
        return angles_rad

    def _visualize_all_joint_markers_from_fk(self):
        for joint_def in self.robot_definition:
            if joint_def.get("has_markers"): self._visualize_joint_local_markers_in_world(joint_def["name"])

    def _visualize_trajectory_line(self, target_pos, standoff_pos) -> Optional[vtkMRMLMarkupsFiducialNode]:
        line_node = slicer.mrmlScene.GetFirstNodeByName(self.TRAJECTORY_LINE_NODE_NAME)
        if not line_node:
            line_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", self.TRAJECTORY_LINE_NODE_NAME)
        if disp_node := line_node.GetDisplayNode():
            disp_node.SetSelectedColor(0.0, 1.0, 1.0)
            disp_node.SetColor(0.0, 1.0, 1.0)
        line_node.RemoveAllControlPoints()
        line_node.AddControlPointWorld(vtk.vtkVector3d(standoff_pos))
        line_node.AddControlPointWorld(vtk.vtkVector3d(target_pos))
        return line_node

    def _calculate_fiducial_alignment_matrix(self, node_name: str, local_coords: List[Tuple[float,float,float]]) -> Optional[vtk.vtkMatrix4x4]:
        fiducials_node = slicer.mrmlScene.GetFirstNodeByName(node_name)
        if not (fiducials_node and fiducials_node.GetNumberOfControlPoints() >= 3 and len(local_coords) >= 3): return None
        n_pts = min(fiducials_node.GetNumberOfControlPoints(), len(local_coords), 3)
        target = vtk.vtkPoints(); source = vtk.vtkPoints()
        for i in range(n_pts):
            target.InsertNextPoint(fiducials_node.GetNthControlPointPositionWorld(i)); source.InsertNextPoint(local_coords[i])
        tf = vtk.vtkLandmarkTransform(); tf.SetSourceLandmarks(source); tf.SetTargetLandmarks(target)
        tf.SetModeToRigidBody(); tf.Update(); return tf.GetMatrix()

    def _create_model_and_articulation_transform(self, jn: str, stl: str, tf_mat: vtk.vtkMatrix4x4, color) -> Tuple[Optional[vtkMRMLModelNode], Optional[vtkMRMLLinearTransformNode]]:
        try:
            model = loadModel(stl)
            model.SetName(f"{jn}Model")
            self.jointModelNodes[jn] = model
        except Exception as e:
            logging.error(f"Failed to load STL '{stl}' for {jn}: {e}"); return None, None
        tf_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", f"{jn}ArticulationTransform")
        tf_node.SetMatrixTransformToParent(tf_mat)
        self.jointTransformNodes[jn] = tf_node
        model.SetAndObserveTransformNodeID(tf_node.GetID())
        if disp := model.GetDisplayNode():
            disp.SetVisibility(self.models_visible)
            node_color = color or (0.7, 0.7, 0.7)
            disp.SetColor(node_color); disp.SetOpacity(0.85); disp.SetSelectedColor(node_color)
        return model, tf_node

    def _handle_joint_detection_results(self, identified_joints_data: Dict[str, List[Dict]]):
        if not identified_joints_data: return
        for jn, markers in identified_joints_data.items():
            config = self.robot_definition_dict.get(jn)
            if not (config and config.get("has_markers") and len(markers) == 3): continue
            if jn == "Baseplate":
                avg_y = sum(m["ras_coords"][1] for m in markers) / 3.0
                for m in markers: m["ras_coords"][1] = avg_y
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"{jn}Fiducials")
            self._organize_node_in_subject_hierarchy(node, self.MASTER_FOLDER_NAME, "Detected MRI Markers")
            if disp := node.GetDisplayNode():
                disp.SetVisibility(self.markers_visible)
                disp.SetColor(config.get("color", (0.8,0.8,0.2))); disp.SetSelectedColor(config.get("color", (0.8,0.8,0.2)))
            for i, m in enumerate(markers):
                idx = node.AddControlPoint(m["ras_coords"]); node.SetNthControlPointLabel(idx, f"{jn}_M{i+1}")

    def _sort_l_shaped_markers(self, markers: List[Dict], len1: float, len2: float, tol: float) -> Optional[List[Dict]]:
        if len(markers) != 3: return None
        points = [{'data': m, 'ras': tuple(m["ras_coords"])} for m in markers]
        l_short, l_long = sorted((len1, len2))
        for i in range(3):
            corner, p1, p2 = points[i], points[(i+1)%3], points[(i+2)%3]
            d1, d2 = math.dist(corner['ras'], p1['ras']), math.dist(corner['ras'], p2['ras'])
            if abs(d1 - l_short) <= self.DISTANCE_TOLERANCE and abs(d2 - l_long) <= self.DISTANCE_TOLERANCE: return [corner['data'], p1['data'], p2['data']]
            if abs(d1 - l_long) <= self.DISTANCE_TOLERANCE and abs(d2 - l_short) <= self.DISTANCE_TOLERANCE: return [corner['data'], p2['data'], p1['data']]
        return None

    def joint_detection(self, pNode: 'MamriParameterNode') -> Dict[str, List[Dict]]:
        all_node = slicer.mrmlScene.GetFirstNodeByName("DetectedFiducials")
        if not (all_node and all_node.GetNumberOfControlPoints() >= 3): return {}
        all_fiducials = [ {"id": i, "ras_coords": list(all_node.GetNthControlPointPositionWorld(i))} for i in range(all_node.GetNumberOfControlPoints())]
        identified, used_ids = {}, set()
        for jc in self.robot_definition:
            if not jc.get("has_markers"): continue
            jn = jc["name"]; arm_lengths = jc.get("arm_lengths") 
            if not arm_lengths or len(arm_lengths) != 2: continue
            l1, l2 = arm_lengths[0], arm_lengths[1]; expected_dists = sorted([l1, l2, math.hypot(l1, l2)])
            available = [f for f in all_fiducials if f["id"] not in used_ids]
            if len(available) < 3: continue
            for combo in itertools.combinations(available, 3):
                pts = [c["ras_coords"] for c in combo]; dists = sorted([math.dist(pts[0], pts[1]), math.dist(pts[0], pts[2]), math.dist(pts[1], pts[2])])
                if all(abs(d - e) <= self.DISTANCE_TOLERANCE for d, e in zip(dists, expected_dists)):
                    matched_data = [dict(c) for c in combo]
                    sorted_data = self._sort_l_shaped_markers(matched_data, l1, l2, self.DISTANCE_TOLERANCE)
                    identified[jn] = sorted_data if sorted_data else matched_data
                    used_ids.update(c["id"] for c in combo); break 
        return identified

    def volume_threshold_segmentation(self, pNode: 'MamriParameterNode') -> None:
        try: sitk_img = sitkUtils.PullVolumeFromSlicer(pNode.inputVolume)
        except Exception as e: logging.error(f"Failed to pull volume: {e}"); return
        binary = sitk.BinaryThreshold(sitk_img, self.INTENSITY_THRESHOLD, 65535); closed = sitk.BinaryMorphologicalClosing(binary, [2] * 3, sitk.sitkBall)
        labeled = sitk.ConnectedComponent(closed); stats = sitk.LabelShapeStatisticsImageFilter(); stats.Execute(labeled)
        fiducials_data = [{"vol": stats.GetPhysicalSize(lbl), "centroid": stats.GetCentroid(lbl), "id": lbl} for lbl in stats.GetLabels() if self.MIN_VOLUME_THRESHOLD <= stats.GetPhysicalSize(lbl) <= self.MAX_VOLUME_THRESHOLD]
        self._clear_node_by_name("DetectedFiducials")
        if fiducials_data:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "DetectedFiducials")
            self._organize_node_in_subject_hierarchy(node, self.MASTER_FOLDER_NAME, "Detected MRI Markers")
            if disp := node.GetDisplayNode(): disp.SetVisibility(False)
            for fd in fiducials_data:
                lps = fd["centroid"]; idx = node.AddControlPoint([-lps[0], -lps[1], lps[2]]); node.SetNthControlPointLabel(idx, f"M_{fd['id']}_{fd['vol']:.0f}mm³")
        all_labels = stats.GetLabels()
        if not all_labels: return
        non_fiducial_labels = [lbl for lbl in all_labels if lbl not in {f['id'] for f in fiducials_data}]
        if not non_fiducial_labels: return
        largest_label_id = max(non_fiducial_labels, key=stats.GetPhysicalSize)
        largest_object_img = sitk.Cast(sitk.BinaryThreshold(labeled, largest_label_id, largest_label_id, 1, 0), sitk.sitkUInt8)
        self._clear_node_by_name("TempBodyLabelMap")
        tempLabelmapNode = sitkUtils.PushVolumeToSlicer(largest_object_img, name="TempBodyLabelMap", className="vtkMRMLLabelMapVolumeNode")
        if not tempLabelmapNode: return
        self._clear_node_by_name("AutoBodySegmentation")
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "AutoBodySegmentation")
        self._organize_node_in_subject_hierarchy(segmentationNode, self.MASTER_FOLDER_NAME, "Segmentations")
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(tempLabelmapNode, segmentationNode)
        slicer.mrmlScene.RemoveNode(tempLabelmapNode)
        if segmentationNode.GetSegmentation().GetNumberOfSegments() > 0:
            segment = segmentationNode.GetSegmentation().GetNthSegment(0)
            segment.SetName("Body")
            segment.SetColor([0.8, 0.2, 0.2])
            segmentation = segmentationNode.GetSegmentation()
            closedSurfaceRepresentationName = slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName()
            segmentation.CreateRepresentation(closedSurfaceRepresentationName)
        segmentationNode.CreateDefaultDisplayNodes()
        if dispNode := segmentationNode.GetDisplayNode(): dispNode.SetOpacity(0.75); dispNode.SetVisibility3D(True)
        pNode.segmentationNode = segmentationNode

    def _cleanup_module_nodes(self):
        """Cleans up nodes from the previous run, but preserves persistent nodes like saved transforms."""
        logging.info("Cleaning up module nodes for new run...")
        self.jointCollisionPolys.clear()
        self.jointModelNodes.clear()
        self.jointTransformNodes.clear()
        self.jointFixedOffsetTransformNodes.clear()
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        if not shNode:
            return
        folders_to_clear = [ "Robot Model", "Detected MRI Markers", "Debug Markers", "Segmentations", "Trajectory Plan", "Sequences" ]
        masterFolderItemID = shNode.GetItemByName(self.MASTER_FOLDER_NAME)
        if masterFolderItemID:
            childItems = vtk.vtkIdList()
            shNode.GetItemChildren(masterFolderItemID, childItems)
            for i in range(childItems.GetNumberOfIds()):
                childItemID = childItems.GetId(i)
                childName = shNode.GetItemName(childItemID)
                if childName in folders_to_clear:
                    shNode.RemoveItem(childItemID)
        self._clear_node_by_name(self.TARGET_POSE_TRANSFORM_NODE_NAME)
        if getattr(self, 'DEBUG_COLLISIONS', False):
            all_parts = [j["name"] for j in self.robot_definition] + ["Body"]
            for part_name in all_parts:
                self._clear_node_by_name(f"DEBUG_COLLISION_{part_name}")
        logging.info("Cleanup complete.")
    
    def return_to_zero_position(self, status_label: qt.QLabel = None):
        """
        Commands the robot to move to the home position (all joints at 0 steps).
        This function is BLOCKING.
        """
        if not self.is_robot_connected():
            error_msg = "Execution failed: Robot not connected."
            if status_label: status_label.text = error_msg
            logging.error(error_msg)
            return

        self.stop_execution_flag = False

        def update_gui(message: str):
            if status_label:
                status_label.text = message
            slicer.app.processEvents()

        num_joints = len(self.articulated_chain)
        target_pose_steps = np.zeros(num_joints, dtype=int)
        
        logging.info("\n--- Commanding Robot to Return to Zero Position ---")

        timeout = 120.0  # Generous timeout for the entire homing sequence
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.stop_execution_flag:
                update_gui("Homing stopped by user.")
                return
            
            live_positions = self.get_current_positions()
            if live_positions is None:
                time.sleep(0.2)
                continue
            
            live_pose_steps = np.array(live_positions[:num_joints])

            # Check if all joints are at the zero position
            if np.all(np.abs(live_pose_steps - target_pose_steps) <= 2):
                logging.info("Robot has reached the zero position.")
                break

            status_messages = []
            for joint_index in range(num_joints):
                if abs(live_pose_steps[joint_index] - target_pose_steps[joint_index]) > 2:
                    joint_def = self.robot_definition_dict[self.articulated_chain[joint_index]]
                    command_letter = joint_def["command_letter"]
                    command = f"{command_letter}{target_pose_steps[joint_index]}"
                    self.send_command_to_robot(command)
                    status_messages.append(f"{command_letter}: {live_pose_steps[joint_index]}->0")

            update_gui("Homing... " + ", ".join(status_messages))
            
            live_pose_rad = self._convert_steps_array_to_angles(live_pose_steps)
            self.setRobotPose(live_pose_rad)
            
            time.sleep(0.1)

        else: # This 'else' belongs to the while loop, executing on timeout
            error_msg = "TIMEOUT: Robot failed to reach zero position."
            logging.error(error_msg)
            update_gui(error_msg)
            return

        # Final update to ensure the model is perfectly at zero
        self.setRobotPose(np.zeros(num_joints))
        update_gui("Robot is at zero position.")
        logging.info("Return to zero command finished.")

    def move_to_specific_pose(self, target_pose_steps: np.ndarray, status_label: qt.QLabel = None):
        """
        Commands the robot to move to a specific pose defined by step counts.
        This function is BLOCKING.
        """
        if not self.is_robot_connected():
            error_msg = "Execution failed: Robot not connected."
            if status_label: status_label.text = error_msg
            logging.error(error_msg)
            return

        self.stop_execution_flag = False

        def update_gui(message: str):
            if status_label:
                status_label.text = message
            slicer.app.processEvents()

        num_joints = len(self.articulated_chain)
        logging.info(f"\n--- Commanding Robot to Move to Pose: {target_pose_steps.tolist()} ---")

        timeout = 120.0
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.stop_execution_flag:
                update_gui("Movement stopped by user.")
                return
            
            live_positions = self.get_current_positions()
            if live_positions is None:
                time.sleep(0.2)
                continue
            
            live_pose_steps = np.array(live_positions[:num_joints])

            if np.all(np.abs(live_pose_steps - target_pose_steps) <= 2):
                logging.info("Robot has reached the target pose.")
                break

            status_messages = []
            for joint_index in range(num_joints):
                if abs(live_pose_steps[joint_index] - target_pose_steps[joint_index]) > 2:
                    joint_def = self.robot_definition_dict[self.articulated_chain[joint_index]]
                    command_letter = joint_def["command_letter"]
                    command = f"{command_letter}{target_pose_steps[joint_index]}"
                    self.send_command_to_robot(command)
                    status_messages.append(f"{command_letter}: {live_pose_steps[joint_index]}->{target_pose_steps[joint_index]}")

            update_gui("Moving to pose... " + ", ".join(status_messages))
            
            live_pose_rad = self._convert_steps_array_to_angles(live_pose_steps)
            self.setRobotPose(live_pose_rad)
            
            time.sleep(0.1)

        else:
            error_msg = "TIMEOUT: Robot failed to reach target pose."
            logging.error(error_msg)
            update_gui(error_msg)
            return

        final_pose_rad = self._convert_steps_array_to_angles(target_pose_steps)
        self.setRobotPose(final_pose_rad)
        update_gui("Robot has arrived at the estimated pose.")
        logging.info("Move to pose command finished.")

    def _toggle_mri_fiducials(self, checked: bool):
        """Toggles visibility of only the fiducials detected from the MRI scan."""
        self.markers_visible = checked
        marker_names = set()
        for jc in self.robot_definition:
            if jc.get("has_markers"):
                marker_names.add(f"{jc['name']}Fiducials")
        for name in marker_names:
            if node := slicer.mrmlScene.GetFirstNodeByName(name):
                if disp := node.GetDisplayNode():
                    disp.SetVisibility(checked)

    def _toggle_debug_markers(self, checked: bool):
        """Toggles visibility of only the local coordinate debug markers."""
        self.debug_markers_visible = checked
        marker_names = set()
        for jc in self.robot_definition:
            if jc.get("has_markers"):
                marker_names.add(f"{jc['name']}_LocalMarkers_WorldView_DEBUG")
        for name in marker_names:
            if node := slicer.mrmlScene.GetFirstNodeByName(name):
                if disp := node.GetDisplayNode():
                    disp.SetVisibility(checked)

    def _toggle_robot_models(self, checked: bool):
        for model_node in self.jointModelNodes.values():
            if disp := model_node.GetDisplayNode(): disp.SetVisibility(checked)

#
# MamriTest
#
class MamriTest(ScriptedLoadableModuleTest):
    def setUp(self): slicer.mrmlScene.Clear()
    def runTest(self): self.setUp()