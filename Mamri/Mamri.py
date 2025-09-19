import logging
import os
import itertools
import math
from typing import Annotated, Optional, Dict, List, Tuple
import time
import json
import threading

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

class Mamri(ScriptedLoadableModule):
    '''Defines the module's metadata and properties.'''
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

@parameterNodeWrapper
class MamriParameterNode:
    '''Manages the parameters and nodes used by the module.'''
    inputVolume: vtkMRMLScalarVolumeNode
    useSavedBaseplate: bool = False
    applyEndEffectorCorrection: bool = False
    syncControllerToEncoder: bool = True

    segmentationNode: vtkMRMLSegmentationNode
    targetFiducialNode: vtkMRMLMarkupsFiducialNode
    entryPointFiducialNode: vtkMRMLMarkupsFiducialNode
    safetyDistance: Annotated[float, WithinRange(0.0, 50.0)] = 5.0

class MamriWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    '''Manages the user interface, event handling, and state of the Mamri module.'''
    def __init__(self, parent=None) -> None:
        '''Initializes widget instance variables and timers.'''
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self._animationTimer = None
        self._isPlaying = False
        self.trajectoryPath = None
        self.trajectoryKeyframes = None
        self.lastEstimatedPoseSteps = None
        self.logicTargetSteps = None
        self.num_joints = 0
        self.robotTaskTimer = qt.QTimer()
        self.robotTaskTimer.setInterval(150)
        self.robotTaskTimer.timeout.connect(self._onRobotTaskStep)
        self._task_state = {}
        self.statusUpdateTimer = qt.QTimer()
        self._last_heavy_update_time = 0
        self._targetFiducialObserverTags = []

    def setup(self) -> None:
        '''Loads the UI from XML, connects signals to slots, and initializes module logic.'''
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/Mamri.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.planTrajectoryButton.clicked.connect(self.onPlanHeuristicPathButton)
        self.ui.drawFiducialsCheckBox.connect("toggled(bool)", self.onDrawFiducialsCheckBoxToggled)
        self.ui.drawModelsCheckBox.connect("toggled(bool)", self.onDrawModelsCheckBoxToggled)
        self.ui.saveBaseplateButton.connect("clicked(bool)", self.onSaveBaseplateButton)
        self.ui.findEntryPointButton.connect("clicked(bool)", self.onFindEntryPointButton)
        self.ui.zeroRobotButton.connect("clicked(bool)", self.onZeroRobotButton)
        self.ui.drawDebugMarkersCheckBox.connect("toggled(bool)", self.onDrawDebugMarkersCheckBoxToggled)
        self.ui.trajectorySlider.valueChanged.connect(self.onTrajectorySliderChanged)
        self.ui.playPauseButton.clicked.connect(self.onPlayPauseButton)
        self.ui.zeroHardwareButton.clicked.connect(self.onZeroHardwareButton)
        self.ui.moveToPoseButton.clicked.connect(self.onMoveToPoseButton)
        self.ui.refreshPortsButton.clicked.connect(self.onRefreshPortsButton)
        self.ui.connectButton.toggled.connect(self.onConnectButtonToggled)
        self.ui.executeTrajectoryButton.clicked.connect(self.onExecuteTrajectoryButton)
        self.ui.stopTrajectoryButton.clicked.connect(self.onStopTrajectoryButton)
        self.ui.returnToZeroButton.clicked.connect(self.onReturnToZeroButton)
        self.ui.jogPlusButton.clicked.connect(lambda: self.onJogClicked(True))
        self.ui.jogMinusButton.clicked.connect(lambda: self.onJogClicked(False))
        self.ui.connectEncoderButton.toggled.connect(self.onConnectEncoderButtonToggled)
        self.ui.sendEncoderCommandButton.clicked.connect(self.onSendEncoderCommand)
        self.ui.encoderCommandLineEdit.returnPressed.connect(self.onSendEncoderCommand)

        self._animationTimer = qt.QTimer()
        self._animationTimer.setInterval(50)
        self._animationTimer.timeout.connect(self.doAnimationStep)
        self.statusUpdateTimer.setInterval(40)
        self.statusUpdateTimer.timeout.connect(self.updateStatusDisplay)

        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.logic = MamriLogic()
        self.num_joints = self.logic.num_joints
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.onRefreshPortsButton()
        self._setupUIComponents()
        self.initializeParameterNode()

    def cleanup(self) -> None:
        '''Removes observers, stops timers, and disconnects from hardware upon module closure.'''
        self.removeObservers()
        self._removeTargetFiducialObservers()
        if self._animationTimer:
            self._animationTimer.stop()
        if self.statusUpdateTimer:
            self.statusUpdateTimer.stop()
        if hasattr(self, "robotTaskTimer") and self.robotTaskTimer.isActive():
            self.robotTaskTimer.stop()
            if self.logic:
                self.logic.send_stop_commands()
            self._task_state = {}
            logging.info("Robot task stopped during module cleanup.")
        if self.logic:
            self.logic.disconnect_from_motor_controller()
            self.logic.disconnect_from_encoder()

    def enter(self) -> None:
        '''Called when the module is selected, initializing parameters and updating the UI.'''
        self.initializeParameterNode()
        self.onRefreshPortsButton()
        self.updateStatusDisplay()
        self._checkAllButtons()

    def exit(self) -> None:
        '''Called when the module is deselected, removing observers from the parameter node.'''
        self.remove_parameter_node_observers()

    def onSceneStartClose(self, caller, event) -> None:
        '''Clears the parameter node when the Slicer scene is about to close.'''
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        '''Re-initializes the parameter node after the Slicer scene has closed.'''
        if self.parent.isEntered:
            self.initializeParameterNode()

    def setParameterNode(self, inputParameterNode: Optional[MamriParameterNode]) -> None:
        '''Sets and observes the module's parameter node.'''
        self.remove_parameter_node_observers()
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._onParameterNodeModified)
            self._onTargetFiducialNodeChanged()
            self._updateTargetRobotCoordinates()
        self._checkAllButtons()

    def initializeParameterNode(self) -> None:
        '''Initializes or retrieves the module's parameter node and sets default values.'''
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
        '''Removes all observers associated with the parameter node.'''
        if self._parameterNode:
            if self._parameterNodeGuiTag:
                self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
                self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkAllButtons)

    def onApplyButton(self) -> None:
        '''Initiates the main fiducial detection and robot pose estimation process.'''
        if not self._parameterNode:
            slicer.util.errorDisplay("Parameter node is not initialized.")
            return
        self.lastEstimatedPoseSteps = None
        self.logicTargetSteps = None
        self.ui.moveToPoseButton.enabled = False
        slicer.app.processEvents()
        self._populatePoseTable(self.ui.estimatedPoseTableWidget, None, "Estimated Pose")
        models_visible = self.ui.drawModelsCheckBox.isChecked()
        markers_visible = self.ui.drawFiducialsCheckBox.isChecked()
        estimated_pose_rad, estimated_steps = self.logic.process(
            self._parameterNode,
            models_visible=models_visible,
            markers_visible=markers_visible
        )
        if estimated_pose_rad is not None and estimated_steps is not None:
            self.lastEstimatedPoseSteps = estimated_steps
            self._populatePoseTable(self.ui.estimatedPoseTableWidget, estimated_pose_rad, "Estimated Pose")
        else:
            for i in range(self.num_joints):
                est_table = self.ui.estimatedPoseTableWidget
                est_table.setItem(i, 1, qt.QTableWidgetItem("Failed"))
                est_table.setItem(i, 2, qt.QTableWidgetItem("---"))
        self._updateTargetRobotCoordinates()
        self.updateStatusDisplay()
        self._checkAllButtons()

    def onPlanHeuristicPathButton(self) -> None:
        '''Plans a simple "up, over, down" trajectory to the target.'''
        if not self._parameterNode:
            slicer.util.errorDisplay("Parameter node is not initialized.")
            return
        if self.lastEstimatedPoseSteps is None:
            slicer.util.errorDisplay("Cannot plan a path without a valid estimated pose. Please run Pose Estimation first.")
            return
        if self._isPlaying:
            self.onPlayPauseButton()
        self.ui.trajectoryDistanceLabel.text = "n/a"
        self.ui.trajectoryKeyframesLabel.text = "n/a"
        self.ui.trajectoryCollisionLabel.text = "n/a"
        self.ui.trajectoryCollisionLabel.setStyleSheet("")
        self._populatePoseTable(self.ui.trajectoryStartPoseTable, np.zeros(self.num_joints), "Start Pose")
        self._populatePoseTable(self.ui.trajectoryEndPoseTable, np.zeros(self.num_joints), "End Pose")
        with slicer.util.MessageDialog("Planning...", "Generating heuristic path...") as dialog:
            slicer.app.processEvents()
            path, keyframes, collision_detected = self.logic.planHeuristicPath(self._parameterNode, start_pose_steps=self.lastEstimatedPoseSteps)
        self.trajectoryPath = path
        self.trajectoryKeyframes = keyframes
        if self.trajectoryPath and self.trajectoryKeyframes:
            target_node = self._parameterNode.targetFiducialNode
            entry_node = self._parameterNode.entryPointFiducialNode
            if target_node and entry_node and target_node.GetNumberOfControlPoints() > 0 and entry_node.GetNumberOfControlPoints() > 0:
                p1 = np.array(target_node.GetNthControlPointPositionWorld(0))
                p2 = np.array(entry_node.GetNthControlPointPositionWorld(0))
                distance = np.linalg.norm(p1 - p2)
                self.ui.trajectoryDistanceLabel.text = f"{distance:.2f} mm"
            self.ui.trajectoryKeyframesLabel.text = str(len(self.trajectoryKeyframes))
            if collision_detected:
                self.ui.trajectoryCollisionLabel.text = "Collision Detected"
                self.ui.trajectoryCollisionLabel.setStyleSheet("color: red")
            else:
                self.ui.trajectoryCollisionLabel.text = "Clear"
                self.ui.trajectoryCollisionLabel.setStyleSheet("color: green")
            self._populatePoseTable(self.ui.trajectoryStartPoseTable, self.trajectoryKeyframes[0], "Start Pose")
            self._populatePoseTable(self.ui.trajectoryEndPoseTable, self.trajectoryKeyframes[-1], "End Pose")
            self.logicTargetSteps = self.logic._convert_angles_to_steps_array(self.trajectoryKeyframes[-1])
            slicer.util.infoDisplay(f"Generated heuristic path with {len(self.trajectoryPath)} steps.")
            if hasattr(self.ui, "trajectorySlider"):
                self.ui.trajectorySlider.blockSignals(True)
                self.ui.trajectorySlider.maximum = len(self.trajectoryPath) - 1
                self.ui.trajectorySlider.value = 0
                self.ui.trajectorySlider.blockSignals(False)
            self.onTrajectorySliderChanged(0)
        else:
            slicer.util.errorDisplay("Failed to generate heuristic path.")
            self.logicTargetSteps = None
        self.updateStatusDisplay()
        self._checkAllButtons()

    def onTrajectorySliderChanged(self, value):
        '''Updates the robot's 3D model pose when the trajectory simulation slider is moved.'''
        if self.trajectoryPath is None or value >= len(self.trajectoryPath):
            return
        self.logic.setRobotPose(self.trajectoryPath[value])
        if hasattr(self.ui, "trajectoryStatusLabel"):
            percent = (value / self.ui.trajectorySlider.maximum) * 100 if self.ui.trajectorySlider.maximum > 0 else 0
            self.ui.trajectoryStatusLabel.text = f"Path: {percent:.0f}%"

    def onPlayPauseButton(self):
        '''Starts or stops the trajectory simulation animation.'''
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

    def doAnimationStep(self):
        '''Moves the trajectory simulation forward by one step.'''
        current_value = self.ui.trajectorySlider.value
        if current_value < self.ui.trajectorySlider.maximum:
            self.ui.trajectorySlider.value = current_value + 1
        else:
            self._animationTimer.stop()
            self._isPlaying = False
            self.ui.playPauseButton.text = "Play"

    def onRefreshPortsButton(self):
        '''Refreshes the list of available serial ports for the motor controller and encoder.'''
        ports = self.logic.get_available_serial_ports()
        self.ui.serialPortComboBox.clear()
        self.ui.serialPortComboBox.addItems(ports)
        if not ports:
            self.ui.serialPortComboBox.addItem("No ports found")
        self.ui.encoderPortComboBox.clear()
        self.ui.encoderPortComboBox.addItems(ports)
        if not ports:
            self.ui.encoderPortComboBox.addItem("No ports found")

    def onConnectButtonToggled(self, checked):
        '''Handles connection and disconnection of the motor controller hardware.'''
        port = self.ui.serialPortComboBox.currentText
        if checked:
            if self.logic.connect_to_motor_controller(port):
                self.ui.connectionStatusLabel.text = f"Status: Connected to {port}"
                self.ui.connectButton.text = "Disconnect"
            else:
                self.ui.connectionStatusLabel.text = f"Status: Failed to connect"
                self.ui.connectButton.setChecked(False)
        else:
            self.logic.disconnect_from_motor_controller()
            self.ui.connectionStatusLabel.text = "Status: Not Connected"
            self.ui.connectButton.text = "Connect"
        self._updatePollingTimerState()
        self.updateStatusDisplay()
        self._checkAllButtons()

    def onConnectEncoderButtonToggled(self, checked):
        '''Handles connection and disconnection of the encoder hardware.'''
        port = self.ui.encoderPortComboBox.currentText
        if checked:
            if self.logic.connect_to_encoder(port):
                self.ui.encoderConnectionStatusLabel.text = f"Status: Connected to {port}"
                self.ui.connectEncoderButton.text = "Disconnect"
            else:
                self.ui.encoderConnectionStatusLabel.text = f"Status: Failed to connect"
                self.ui.connectEncoderButton.setChecked(False)
        else:
            self.logic.disconnect_from_encoder()
            self.ui.encoderConnectionStatusLabel.text = "Status: Not Connected"
            self.ui.connectEncoderButton.text = "Connect"
        self._updatePollingTimerState()
        self.updateStatusDisplay()
        self._checkAllButtons()

    def onMoveToPoseButton(self):
        '''Commands the physical robot to move to the last calculated pose from MRI analysis.'''
        if self.robotTaskTimer.isActive():
            slicer.util.warningDisplay("An action is already being executed.")
            return
        if self.lastEstimatedPoseSteps is None:
            slicer.util.errorDisplay("No pose has been estimated yet.")
            return
        self.logicTargetSteps = self.lastEstimatedPoseSteps
        self.updateStatusDisplay()
        self._startRobotTask(mode="move_to_pose", target_steps=self.lastEstimatedPoseSteps)

    def onExecuteTrajectoryButton(self):
        '''Commands the physical robot to execute the planned trajectory.'''
        if self.robotTaskTimer.isActive():
            slicer.util.warningDisplay("A trajectory is already being executed.")
            return
        if not self.trajectoryKeyframes:
            slicer.util.errorDisplay("Please plan a path first.")
            return
        self.logicTargetSteps = self.logic._convert_angles_to_steps_array(self.trajectoryKeyframes[-1])
        self.updateStatusDisplay()
        self._startRobotTask(mode="trajectory", keyframes=self.trajectoryKeyframes)

    def onReturnToZeroButton(self):
        '''Commands the physical robot to move to its zero (home) position.'''
        if self.robotTaskTimer.isActive():
            slicer.util.warningDisplay("An action is already being executed.")
            return
        self.logicTargetSteps = np.zeros(self.num_joints, dtype=int)
        self.updateStatusDisplay()
        self._startRobotTask(mode="homing", target_steps=self.logicTargetSteps)

    def onStopTrajectoryButton(self):
        '''Sets a flag to stop the currently running robot task.'''
        if not self.robotTaskTimer.isActive():
            return
        self.logic.stop_execution_flag = True

    def onJogClicked(self, is_positive: bool):
        '''Commands the robot to move a single joint by a small, specified amount.'''
        if self.robotTaskTimer.isActive():
            slicer.util.warningDisplay("Cannot jog robot while another action is running.")
            return
        current_pos = self.logic.get_current_positions()
        if not current_pos:
            slicer.util.errorDisplay("Could not get robot's current position to execute jog.")
            return
        target_steps = np.array(current_pos[:self.num_joints])
        joint_index = self.ui.jogJointComboBox.currentIndex
        steps_to_move = int(self.ui.jogStepSpinBox.value)
        if not is_positive:
            steps_to_move *= -1
        target_steps[joint_index] += steps_to_move
        self.logicTargetSteps = target_steps
        self.updateStatusDisplay()
        self._startRobotTask(mode="jog", target_steps=target_steps)

    def onZeroHardwareButton(self):
        '''Sends commands to zero the internal step counters of both the encoder and motor controller.'''
        if self.robotTaskTimer.isActive():
            slicer.util.errorDisplay("Cannot zero hardware while a task is running.")
            return
        if self.logic:
            self.logic.zero_hardware_pose()
            self.updateStatusDisplay()

    def onSendEncoderCommand(self):
        '''Sends a manually entered text command to the encoder hardware.'''
        if not self.logic or not self.logic.is_encoder_connected():
            slicer.util.warningDisplay("Encoder is not connected.")
            return
        command = self.ui.encoderCommandLineEdit.text.strip()
        if not command:
            return
        if self.logic.send_command_to_encoder(command):
            logging.info(f"Manually sent command to encoder: '{command}'")
            self.ui.encoderCommandLineEdit.clear()
        else:
            slicer.util.errorDisplay(f"Failed to send command '{command}' to encoder.")

    def onSaveBaseplateButton(self) -> None:
        '''Saves the currently calculated baseplate transform to a persistent node in the scene.'''
        if not self.logic:
            slicer.util.errorDisplay("Logic module is not initialized."); return
        baseplate_tf_node = self.logic.jointTransformNodes.get("Baseplate")
        if not baseplate_tf_node:
            slicer.util.errorDisplay("Baseplate has not been processed yet. Run fiducial detection first to establish its transform."); return
        self.logic.saveBaseplateTransform(baseplate_tf_node)
        slicer.util.infoDisplay(f"Baseplate transform saved successfully to node: '{self.logic.SAVED_BASEPLATE_TRANSFORM_NODE_NAME}'.")

    def onFindEntryPointButton(self) -> None:
        '''Calculates and places a fiducial marker at the optimal entry point on the patient's skin.'''
        if not self._parameterNode:
            slicer.util.errorDisplay("Parameter node is not initialized.")
            return
        if not (self._parameterNode.targetFiducialNode and self._parameterNode.targetFiducialNode.GetNumberOfControlPoints() > 0 and self._parameterNode.segmentationNode):
            slicer.util.errorDisplay("Please select a body segmentation and place a target marker first.")
            return
        self.logic.findAndSetEntryPoint(self._parameterNode)

    def onZeroRobotButton(self) -> None:
        '''Resets the 3D robot model in the Slicer scene to its zero-angle configuration.'''
        if self.logic:
            self.logic.zeroRobot()

    def onDrawFiducialsCheckBoxToggled(self, checked: bool) -> None:
        '''Toggles the visibility of fiducial markers detected in the MRI scan.'''
        self.logic._toggle_mri_fiducials(checked)

    def onDrawModelsCheckBoxToggled(self, checked: bool) -> None:
        '''Toggles the visibility of the 3D robot model parts.'''
        self.logic._toggle_robot_models(checked)

    def onDrawDebugMarkersCheckBoxToggled(self, checked: bool):
        '''Toggles the visibility of debug markers showing local joint coordinate systems.'''
        self.logic._toggle_debug_markers(checked)

    def _startRobotTask(self, mode, **kwargs):
        '''Initializes and starts a new robot task (e.g., trajectory, homing) using a QTimer.'''
        if self.robotTaskTimer.isActive():
            slicer.util.warningDisplay("A robot task is already running.")
            return
        self.logic.stop_execution_flag = False
        self.statusUpdateTimer.stop()
        self._task_state = {
            "mode": mode,
            "start_time": time.time(),
            "timeout": 120.0,
            "target_steps": kwargs.get("target_steps"),
            "keyframes": kwargs.get("keyframes"),
            "keyframe_index": 0,
            "last_command_time": 0,
            "last_encoder_pos": None,
            "stall_start_time": 0,
        }
        now = time.time()
        if mode == "trajectory":
            target_steps = self.logic._convert_angles_to_steps_array(self._task_state["keyframes"][0])
            self._task_state["target_steps"] = target_steps
            self.logic.send_target_pose_commands(target_steps)
            self._task_state["last_command_time"] = now
            self._task_state["stall_start_time"] = now
        else:
            self.logic.send_target_pose_commands(self._task_state["target_steps"])
            self._task_state["last_command_time"] = now
            self._task_state["stall_start_time"] = now
        self.robotTaskTimer.start()
        self._checkAllButtons()

    def _onRobotTaskStep(self):
        '''Executes a single step of the active robot task, checking for arrival, stalls, or user stop requests.'''
        if self.logic.stop_execution_flag:
            logging.info("Stop flag detected. Halting task.")
            self.logic.send_stop_commands()
            self._stopRobotTask(success=False, message="Stopped by user.")
            return
        if time.time() - self._task_state["start_time"] > self._task_state["timeout"]:
            logging.error("Task timed out.")
            self.logic.send_stop_commands()
            self._stopRobotTask(success=False, message="Task timed out.")
            return
        live_encoder_pos = None
        if self.logic and self.logic.is_encoder_connected():
            with self.logic.encoder_data_lock:
                live_encoder_pos = self.logic.true_encoder_position
        if not live_encoder_pos:
            return
        live_mc_pos = self.logic.get_current_positions()
        live_pose_rad = self.logic._convert_steps_array_to_angles(np.array(live_encoder_pos))
        self.logic.setRobotPose(live_pose_rad)
        self.updateStatusDisplay(live_mc_pos)
        current_steps = np.array(live_encoder_pos)
        target_steps = self._task_state["target_steps"]
        last_pos = self._task_state.get("last_encoder_pos")
        now = time.time()
        arrival_tolerance = 0
        if np.all(np.abs(current_steps - target_steps) <= arrival_tolerance):
            if self._task_state["mode"] == "trajectory":
                self._task_state["keyframe_index"] += 1
                if self._task_state["keyframe_index"] < len(self._task_state["keyframes"]):
                    logging.info(f"Reached keyframe {self._task_state['keyframe_index']}. Moving to next.")
                    next_target = self.logic._convert_angles_to_steps_array(self._task_state["keyframes"][self._task_state["keyframe_index"]])
                    self._task_state["target_steps"] = next_target
                    self.logic.send_target_pose_commands(next_target)
                    self._task_state["last_command_time"] = now
                    self._task_state["stall_start_time"] = now
                else:
                    self._stopRobotTask(success=True, message="Trajectory executed successfully.")
            else:
                self._stopRobotTask(success=True, message=f"Task '{self._task_state['mode']}' finished.")
            return
        is_moving = last_pos is None or not np.array_equal(current_steps, last_pos)
        if is_moving:
            self._task_state["last_encoder_pos"] = current_steps
            self._task_state["stall_start_time"] = now
        else:
            STALL_THRESHOLD_SEC = 2.0
            if now - self._task_state.get("stall_start_time", 0) > STALL_THRESHOLD_SEC:
                if now - self._task_state.get("last_command_time", 0) > 1.0:
                    logging.info(f"Robot stalled for >{STALL_THRESHOLD_SEC}s. Re-issuing move command.")
                    self.logic.send_target_pose_commands(target_steps)
                    self._task_state["last_command_time"] = now

    def _stopRobotTask(self, success=False, message=""):
        '''Stops the active robot task timer and cleans up the task state.'''
        self.robotTaskTimer.stop()
        logging.info(message)
        self.logic.robot_state = self.logic.STATE_IDLE
        self._task_state = {}
        if self.logic.is_motor_controller_connected():
            self.statusUpdateTimer.start()
        self.updateStatusDisplay()
        self._checkAllButtons()

    def updateStatusDisplay(self, live_mc_pose_steps=None):
        '''Refreshes all live status information in the UI, such as joint positions and TCP coordinates.'''
        table = self.ui.jointStatusTableWidget
        now = time.time()
        is_executing = self.robotTaskTimer.isActive()
        if self.logic and self.logic.is_encoder_connected():
            with self.logic.encoder_data_lock:
                live_encoder_steps = list(self.logic.true_encoder_position)
            for i in range(self.num_joints):
                enc_item = qt.QTableWidgetItem(f"{live_encoder_steps[i]: >5}")
                enc_item.setFlags(enc_item.flags() & ~qt.Qt.ItemIsEditable)
                enc_item.setTextAlignment(qt.Qt.AlignCenter)
                table.setItem(i, 1, enc_item)
        if now - self._last_heavy_update_time > 0.25:
            self._last_heavy_update_time = now
            if live_mc_pose_steps is None and not is_executing:
                if self.logic and self.logic.is_motor_controller_connected():
                    live_mc_pose_steps = self.logic.get_current_positions()
            pose_for_tcp_calc = live_mc_pose_steps
            ik_error_text, tcp_x, tcp_y, tcp_z = "n/a", "---", "---", "---"
            if self.logic and self.logic.last_ik_error is not None:
                ik_error_text = f"{self.logic.last_ik_error:.4f} mm"
            if pose_for_tcp_calc is not None and self.logic:
                if base_node := self.logic.jointTransformNodes.get("Baseplate"):
                    base_transform_vtk = vtk.vtkMatrix4x4()
                    base_node.GetMatrixTransformToWorld(base_transform_vtk)
                    pose_rad = self.logic._convert_steps_array_to_angles(np.array(pose_for_tcp_calc))
                    joint_values_rad = dict(zip(self.logic.articulated_chain, pose_rad))
                    tcp_transform = self.logic._get_world_transform_for_joint(joint_values_rad, "Needle", base_transform_vtk)
                    if tcp_transform:
                        tcp_x, tcp_y, tcp_z = f"{tcp_transform.GetElement(0, 3):.2f}", f"{tcp_transform.GetElement(1, 3):.2f}", f"{tcp_transform.GetElement(2, 3):.2f}"
                else:
                    tcp_x, tcp_y, tcp_z = "Run Pose", "Estimation", "First"
            self.ui.ikErrorLabel.text = ik_error_text
            self.ui.tcpXLabel.text = tcp_x
            self.ui.tcpYLabel.text = tcp_y
            self.ui.tcpZLabel.text = tcp_z
            for i in range(self.num_joints):
                if live_mc_pose_steps is not None:
                    mc_item = qt.QTableWidgetItem(f"{live_mc_pose_steps[i]: >5}")
                else:
                    mc_item = qt.QTableWidgetItem("---")
                target_steps_for_ui = self._task_state.get("target_steps") if is_executing else self.logicTargetSteps
                if target_steps_for_ui is not None and i < len(target_steps_for_ui):
                    target_item = qt.QTableWidgetItem(f"{int(target_steps_for_ui[i]): >5}")
                else:
                    target_item = qt.QTableWidgetItem("---")
                for col, item in enumerate([mc_item, target_item], 2):
                    item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
                    item.setTextAlignment(qt.Qt.AlignCenter)
                    table.setItem(i, col, item)
        if is_executing:
            self.ui.trajectoryStatusLabel.text = "Executing... (See status table)"
        elif self.trajectoryPath is None:
            self.ui.trajectoryStatusLabel.text = "(No trajectory planned)"
        slicer.app.processEvents()

    def _updatePollingTimerState(self):
        '''Starts or stops the live status polling timer based on hardware connection status.'''
        is_mc_connected = self.logic and self.logic.is_motor_controller_connected()
        is_encoder_connected = self.logic and self.logic.is_encoder_connected()
        if is_mc_connected or is_encoder_connected:
            if not self.statusUpdateTimer.isActive():
                self.statusUpdateTimer.start()
        else:
            if self.statusUpdateTimer.isActive():
                self.statusUpdateTimer.stop()

    def _checkAllButtons(self, caller=None, event=None) -> None:
        '''Updates the enabled/disabled state of all UI buttons based on the current module state.'''
        can_apply = self._parameterNode and self._parameterNode.inputVolume is not None
        if hasattr(self.ui, "applyButton"):
            self.ui.applyButton.enabled = can_apply
            self.ui.applyButton.toolTip = _("Run fiducial detection and robot model rendering.") if can_apply else _("Select an input volume node.")
        model_is_built = self.logic and self.logic.jointTransformNodes.get("Baseplate") is not None
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
            is_executing_check = self.robotTaskTimer.isActive()
            if not is_executing_check and self.trajectoryPath is None:
                self.ui.trajectoryStatusLabel.text = "(No trajectory planned)"
        is_mc_connected = self.logic and self.logic.is_motor_controller_connected()
        is_encoder_connected = self.logic and self.logic.is_encoder_connected()
        is_executing = self.robotTaskTimer.isActive()
        if hasattr(self.ui, "connectButton"):
            self.ui.connectButton.enabled = not is_executing
        if hasattr(self.ui, "refreshPortsButton"):
            self.ui.refreshPortsButton.enabled = not is_executing
        if hasattr(self.ui, "executeTrajectoryButton"):
            self.ui.executeTrajectoryButton.enabled = is_mc_connected and (self.trajectoryKeyframes is not None) and not is_executing
        if hasattr(self.ui, "stopTrajectoryButton"):
            self.ui.stopTrajectoryButton.enabled = is_executing
        if hasattr(self.ui, "returnToZeroButton"):
            self.ui.returnToZeroButton.enabled = is_mc_connected and not is_executing
        if hasattr(self.ui, "moveToPoseButton"):
            self.ui.moveToPoseButton.enabled = is_mc_connected and not is_executing and (self.lastEstimatedPoseSteps is not None)
        if hasattr(self.ui, "manualControlCollapsibleButton"):
            self.ui.manualControlCollapsibleButton.enabled = is_mc_connected and not is_executing
        if hasattr(self.ui, "connectEncoderButton"):
            self.ui.connectEncoderButton.enabled = not is_executing
        if hasattr(self.ui, "zeroHardwareButton"):
            can_zero_hw = is_mc_connected and is_encoder_connected and not is_executing
            self.ui.zeroHardwareButton.enabled = can_zero_hw
            self.ui.zeroHardwareButton.toolTip = "Zero the encoder and motor controller hardware." if can_zero_hw else "Connect both encoder and motor controller to enable."
        if hasattr(self.ui, "encoderCommandGroupBox"):
            can_send_encoder_cmd = is_encoder_connected and not is_executing
            self.ui.encoderCommandGroupBox.enabled = can_send_encoder_cmd
            self.ui.encoderCommandGroupBox.toolTip = "Sends a manual command to the encoder." if can_send_encoder_cmd else "Connect to the encoder and stop any running tasks to enable."

    def _populatePoseTable(self, table: qt.QTableWidget, pose_rad: Optional[np.ndarray], title: str = None):
        '''Populates a QTableWidget with robot pose data in both steps and degrees.'''
        if title:
            headerItem = qt.QTableWidgetItem(title)
            table.setHorizontalHeaderItem(0, headerItem)
        if pose_rad is None:
            for i in range(self.num_joints):
                table.setItem(i, 1, qt.QTableWidgetItem("..."))
                table.setItem(i, 2, qt.QTableWidgetItem("..."))
            return
        pose_steps = self.logic._convert_angles_to_steps_array(pose_rad)
        for i in range(self.num_joints):
            angle_deg = math.degrees(pose_rad[i])
            steps = int(pose_steps[i])
            step_item = qt.QTableWidgetItem(str(steps))
            deg_item = qt.QTableWidgetItem(f"{angle_deg:.2f}")
            step_item.setTextAlignment(qt.Qt.AlignCenter)
            deg_item.setTextAlignment(qt.Qt.AlignCenter)
            table.setItem(i, 1, step_item)
            table.setItem(i, 2, deg_item)

    def _setupUIComponents(self):
        '''Initializes and configures UI components like tables and combo boxes on startup.'''
        if not hasattr(self.ui, "jointStatusTableWidget"):
            return
        self.ui.jogJointComboBox.clear()
        self.ui.jogJointComboBox.addItems(self.logic.articulated_chain if self.logic else [])
        self.num_joints = len(self.logic.articulated_chain) if self.logic else 6
        def configure_table(table, headers, row_labels):
            table.clear()
            table.setRowCount(len(row_labels))
            table.setColumnCount(len(headers))
            table.setHorizontalHeaderLabels(headers)
            table.verticalHeader().setVisible(False)
            for i, label in enumerate(row_labels):
                item = qt.QTableWidgetItem(label)
                item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
                table.setItem(i, 0, item)
            table.resizeColumnsToContents()
            table.horizontalHeader().setSectionResizeMode(qt.QHeaderView.Stretch)
            table.horizontalHeader().setSectionResizeMode(0, qt.QHeaderView.ResizeToContents)
        joint_names = self.logic.articulated_chain if self.logic else [f"J{i+1}" for i in range(self.num_joints)]
        status_headers = ["Joint", "Encoder\n(steps)", "Controller\n(steps)", "Target\n(steps)"]
        self.ui.jointStatusTableWidget.setColumnCount(len(status_headers))
        configure_table(self.ui.jointStatusTableWidget, status_headers, joint_names)
        pose_headers = ["Joint", "Steps", "Degrees (°)"]
        configure_table(self.ui.estimatedPoseTableWidget, pose_headers, joint_names)
        configure_table(self.ui.trajectoryStartPoseTable, ["Start Pose", "Steps", "Degrees (°)"], joint_names)
        configure_table(self.ui.trajectoryEndPoseTable, ["End Pose", "Steps", "Degrees (°)"], joint_names)

    def _onParameterNodeModified(self, caller, event) -> None:
        '''Handles any modification to the parameter node, triggering UI and observer updates.'''
        self._onTargetFiducialNodeChanged()
        self._updateTargetRobotCoordinates()
        self._checkAllButtons()

    def _onTargetFiducialNodeChanged(self) -> None:
        '''Sets up observers on the currently selected target fiducial node to detect changes.'''
        targetNode = self._parameterNode.targetFiducialNode if self._parameterNode else None
        if self._targetFiducialObserverTags and self._targetFiducialObserverTags[0][0] == targetNode:
            return
        self._removeTargetFiducialObservers()
        if targetNode:
            events = (slicer.vtkMRMLMarkupsNode.PointAddedEvent, slicer.vtkMRMLMarkupsNode.PointModifiedEvent)
            for event in events:
                tag = targetNode.AddObserver(event, self._updateTargetRobotCoordinates)
                self._targetFiducialObserverTags.append([targetNode, tag])

    def _removeTargetFiducialObservers(self) -> None:
        '''Removes all observers from the target fiducial node.'''
        for node, tag in self._targetFiducialObserverTags:
            if node:
                node.RemoveObserver(tag)
        self._targetFiducialObserverTags = []

    def _updateTargetRobotCoordinates(self, caller=None, event=None) -> None:
        '''Calculates and displays the target's position in the robot's local coordinate system.'''
        targetNode = self._parameterNode.targetFiducialNode if self._parameterNode else None
        base_node = self.logic.jointTransformNodes.get("Baseplate") if self.logic else None
        if not (targetNode and targetNode.GetNumberOfControlPoints() > 0 and base_node):
            self.ui.targetRobotXLabel.text = "---"
            self.ui.targetRobotYLabel.text = "---"
            self.ui.targetRobotZLabel.text = "---"
            if not base_node:
                 self.ui.targetRobotXLabel.text = "(Run Pose"
                 self.ui.targetRobotYLabel.text = "Estimation)"
            return
        target_pos_world = [0.0, 0.0, 0.0]
        targetNode.GetNthControlPointPositionWorld(0, target_pos_world)
        T_base_to_world = vtk.vtkMatrix4x4()
        base_node.GetMatrixTransformToWorld(T_base_to_world)
        T_world_to_base = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(T_base_to_world, T_world_to_base)
        target_pos_world_h = list(target_pos_world) + [1.0]
        target_pos_base_h = T_world_to_base.MultiplyPoint(target_pos_world_h)
        self.ui.targetRobotXLabel.text = f"{target_pos_base_h[0]:.2f}"
        self.ui.targetRobotYLabel.text = f"{target_pos_base_h[1]:.2f}"
        self.ui.targetRobotZLabel.text = f"{target_pos_base_h[2]:.2f}"

class MamriLogic(ScriptedLoadableModuleLogic):
    '''Contains the core processing, hardware communication, and algorithmic logic for the Mamri module.'''
    def __init__(self) -> None:
        '''Initializes logic instance variables, robot definition, and hardware communication handlers.'''
        ScriptedLoadableModuleLogic.__init__(self)
        self.jointModelNodes: Dict[str, vtkMRMLModelNode] = {}
        self.jointTransformNodes: Dict[str, vtkMRMLLinearTransformNode] = {}
        self.jointFixedOffsetTransformNodes: Dict[str, vtkMRMLLinearTransformNode] = {}
        self.jointCollisionPolys: Dict[str, vtk.vtkPolyData] = {}
        self.INTENSITY_THRESHOLD = 65.0
        self.MIN_VOLUME_THRESHOLD = 50.0
        self.MAX_VOLUME_THRESHOLD = 1500.0
        self.DISTANCE_TOLERANCE = 5.0
        self.models_visible = True
        self.markers_visible = True
        self.debug_markers_visible = False
        self.robot_definition = self._load_robot_definition()
        self.robot_definition_dict = {joint["name"]: joint for joint in self.robot_definition}
        self.articulated_chain = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]
        self.num_joints = len(self.articulated_chain)
        self.SAVED_BASEPLATE_TRANSFORM_NODE_NAME = "MamriSavedBaseplateTransform"
        self.TARGET_POSE_TRANSFORM_NODE_NAME = "MamriTargetPoseTransform_DEBUG"
        self.TRAJECTORY_LINE_NODE_NAME = "TrajectoryLine_DEBUG"
        self.MASTER_FOLDER_NAME = "MAMRI Robot Output"
        self.DEBUG_COLLISIONS = False
        self.motor_controller_serial = None
        self.encoder_serial = None
        self.encoder_listener_thread = None
        self.stop_encoder_thread_flag = threading.Event()
        self.encoder_data_lock = threading.Lock()
        self.true_encoder_position = [0] * len(self.articulated_chain)
        self.STATE_IDLE = 0
        self.STATE_MOVING = 1
        self.robot_state = self.STATE_IDLE
        self.DISCREPANCY_THRESHOLD = 0
        self.sync_timer = qt.QTimer()
        self.sync_timer.setInterval(250)
        self.sync_timer.timeout.connect(self._perform_sync_check)
        self.last_sync_check_pos = None
        self.movement_detected_since_last_sync = False
        self.stop_execution_flag = False
        self.last_ik_error = None
        self.last_known_mc_position = None
        self._discover_robot_nodes_in_scene()

    def getParameterNode(self) -> 'MamriParameterNode':
        '''Returns the singleton instance of the module's parameter node.'''
        return MamriParameterNode(super().getParameterNode())

    def process(self, parameterNode: 'MamriParameterNode', models_visible: bool, markers_visible: bool) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        '''Executes the full pipeline: segmentation, fiducial detection, model building, and inverse kinematics.'''
        self.models_visible = models_visible
        self.markers_visible = markers_visible
        with slicer.util.MessageDialog("Processing...", "Detecting fiducials and building robot model...") as dialog:
            slicer.app.processEvents()
            self._cleanup_module_nodes()
            self.volume_threshold_segmentation(parameterNode)
            identified_joints_data = self.joint_detection(parameterNode)
            self._handle_joint_detection_results(identified_joints_data)
            baseplate_transform_matrix = self._get_baseplate_transform(parameterNode, identified_joints_data)
            if not baseplate_transform_matrix:
                self._build_robot_model(vtk.vtkMatrix4x4())
                return None, None
            self._build_robot_model(baseplate_transform_matrix)
            apply_correction = parameterNode.applyEndEffectorCorrection
            final_pose_angles = None
            if "Joint6" in identified_joints_data:
                joint6_fiducials_node = slicer.mrmlScene.GetFirstNodeByName("Joint6Fiducials")
                if joint6_fiducials_node and joint6_fiducials_node.GetNumberOfControlPoints() == 3:
                    final_pose_angles = self._solve_full_chain_ik(joint6_fiducials_node, apply_correction)
                    if final_pose_angles is not None:
                        self.setRobotPose(final_pose_angles)
                        self._visualize_all_joint_markers_from_fk()
            else:
                logging.info("Prerequisites for full-chain IK not met (e.g., Joint6 markers not found). Cannot estimate pose.")
            logging.info("Mamri processing finished.")
            if final_pose_angles is not None:
                final_pose_steps = self._convert_angles_to_steps_array(final_pose_angles)
                return final_pose_angles, final_pose_steps
            return None, None

    def planTrajectory(self, pNode: 'MamriParameterNode', solve_only=False) -> Optional[np.ndarray]:
        '''Calculates the target joint angles for a biopsy trajectory using collision-aware inverse kinematics.'''
        if not solve_only:
            logging.info("Starting trajectory goal calculation...")
        targetNode, entryNode = pNode.targetFiducialNode, pNode.entryPointFiducialNode
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
                final_error_vector = self._ik_pose_and_collision_error_function(result.x, self.articulated_chain, target_transform_vtk, base_transform_vtk, body_polydata)
                final_error = np.linalg.norm(final_error_vector[:3])
                if result.success and final_error < lowest_error:
                    lowest_error = final_error
                    best_result = result
            except Exception as e:
                logging.warning(f"IK optimization failed for one guess: {e}")
        if not best_result:
            if not solve_only: slicer.util.errorDisplay("Could not find a valid, collision-free trajectory solution.")
            return None
        return np.array(best_result.x)

    def planHeuristicPath(self, pNode: 'MamriParameterNode', start_pose_steps: Optional[np.ndarray] = None, total_steps=100) -> Optional[Tuple[List[np.ndarray], List[np.ndarray], bool]]:
        '''Generates a simple collision-avoidance path using an "up, over, down" strategy.'''
        logging.info("Planning heuristic 'up, over, down' path...")
        collision_detected = False
        if start_pose_steps is not None:
            start_config = self._convert_steps_array_to_angles(start_pose_steps)
            logging.info(f"Planning from estimated start pose: {np.rad2deg(start_config).round(2).tolist()}")
        else:
            start_config = np.array(self._get_current_joint_angles(self.articulated_chain))
            logging.warning("No estimated start pose provided. Planning from current simulated pose.")
        joint6_config = self.planTrajectory(pNode, solve_only=True)
        if joint6_config is None:
            logging.error("Heuristic planning failed: Could not determine a valid joint6 configuration.")
            return None, None, collision_detected
        waypoint1_config = np.copy(start_config)
        waypoint1_config[1] = math.radians(-15)
        waypoint2_config = np.copy(waypoint1_config)
        waypoint2_config[0] = joint6_config[0]
        keyframes = [start_config, waypoint1_config, waypoint2_config, joint6_config]
        path = []
        segment_steps = [total_steps // 4, total_steps // 4, total_steps // 2]
        for i in range(len(keyframes) - 1):
            start_wp, joint6_wp = keyframes[i], keyframes[i+1]
            steps = segment_steps[i]
            is_last_segment = (i == len(keyframes) - 2)
            for j in range(steps):
                t = j / float(steps)
                path.append(start_wp + t * (joint6_wp - start_wp))
            if is_last_segment:
                path.append(joint6_wp)
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
                    collision_detected = True
                    break
        else:
            logging.warning("Could not find 'AutoBodySegmentation' for collision checking the path.")
        return path, keyframes, collision_detected

    def findAndSetEntryPoint(self, pNode: 'MamriParameterNode') -> None:
        '''Finds and marks the closest suitable entry point on the body surface for the biopsy needle.'''
        targetNode = pNode.targetFiducialNode
        segmentationNode = slicer.mrmlScene.GetFirstNodeByName("AutoBodySegmentation")
        if not (targetNode and targetNode.GetNumberOfControlPoints() > 0 and segmentationNode):
            slicer.util.errorDisplay("Please place a target marker and ensure 'AutoBodySegmentation' exists.")
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
        target_pos = np.array(targetNode.GetNthControlPointPositionWorld(0))
        result_point_ids = vtk.vtkIdList()
        search_radius = 80.0
        point_locator.FindPointsWithinRadius(search_radius, target_pos, result_point_ids)
        suitable_points = []
        for i in range(result_point_ids.GetNumberOfIds()):
            point_id = result_point_ids.GetId(i)
            point_normal = np.array(point_normals_array.GetTuple(point_id))
            suitability_score = abs(point_normal[0]) - 2 * abs(point_normal[1])
            if suitability_score > -0.5:
                point_coords = np.array(polydata_with_normals.GetPoint(point_id))
                distance = np.linalg.norm(point_coords - target_pos)
                suitable_points.append({"coords": point_coords, "distance": distance})
        if not suitable_points:
            slicer.util.warningDisplay(f"Could not find a suitable side-entry point within {search_radius}mm of the target.")
            return
        best_candidate = min(suitable_points, key=lambda x: x["distance"])
        closest_point_coords = best_candidate["coords"]
        newNodeName = "ClosestSuitableEntryPoint"
        if oldEntryNode := slicer.mrmlScene.GetFirstNodeByName(newNodeName):
            slicer.mrmlScene.RemoveNode(oldEntryNode)
        entryNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", newNodeName)
        entryNode.GetDisplayNode().SetSelectedColor(0.0, 1.0, 0.5)
        entryNode.AddControlPointWorld(vtk.vtkVector3d(closest_point_coords))
        entryNode.SetNthControlPointLabel(0, "Suitable Entry")
        pNode.entryPointFiducialNode = entryNode
        slicer.util.infoDisplay("Closest suitable side-entry point has been calculated and set.")

    def saveBaseplateTransform(self, current_baseplate_transform_node: vtkMRMLLinearTransformNode):
        '''Saves the given baseplate transform to a persistent node in the Slicer scene.'''
        self._clear_node_by_name(self.SAVED_BASEPLATE_TRANSFORM_NODE_NAME)
        saved_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", self.SAVED_BASEPLATE_TRANSFORM_NODE_NAME)
        world_matrix = vtk.vtkMatrix4x4()
        current_baseplate_transform_node.GetMatrixTransformToWorld(world_matrix)
        saved_node.SetMatrixTransformToParent(world_matrix)
        saved_node.SetSelectable(False)
        self._organize_node_in_subject_hierarchy(saved_node, self.MASTER_FOLDER_NAME, "Saved Transforms")

    def zeroRobot(self) -> None:
        '''Resets the simulated robot model in the 3D view to its zero-angle configuration.'''
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

    def get_available_serial_ports(self) -> List[str]:
        '''Returns a list of available serial port device names on the system.'''
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]

    def connect_to_motor_controller(self, port: str) -> bool:
        '''Establishes a serial connection with the robot motor controller and performs a handshake.'''
        if self.is_motor_controller_connected():
            self.disconnect_from_motor_controller()
        try:
            self.motor_controller_serial = serial.Serial(port, 115200, timeout=0.05, write_timeout=2)
            time.sleep(1.5)
            self.motor_controller_serial.flushInput()
            self.motor_controller_serial.write(b'X\n')
            response = self.motor_controller_serial.readline().decode('ascii').strip()
            if "Hello world!" in response:
                logging.info(f"Motor controller handshake successful on port {port}.")
                self.sync_timer.start()
                return True
            else:
                logging.error(f"Motor controller handshake failed on {port}. Expected 'Hello world!', got '{response}'. Is this the correct port?")
                self.motor_controller_serial.close()
                self.motor_controller_serial = None
                return False
        except serial.SerialException as e:
            logging.error(f"Failed to connect to motor controller on port {port}: {e}")
            if self.motor_controller_serial:
                self.motor_controller_serial.close()
            self.motor_controller_serial = None
            return False

    def disconnect_from_motor_controller(self) -> None:
        '''Closes the serial connection to the motor controller and stops the sync timer.'''
        self.sync_timer.stop()
        if self.motor_controller_serial and self.motor_controller_serial.is_open:
            self.motor_controller_serial.close()
            logging.info(f"Disconnected from motor controller serial port.")
        self.motor_controller_serial = None
        self.robot_state = self.STATE_IDLE

    def is_motor_controller_connected(self) -> bool:
        '''Checks if the serial connection to the motor controller is active.'''
        return self.motor_controller_serial is not None and self.motor_controller_serial.is_open

    def connect_to_encoder(self, port: str) -> bool:
        '''Establishes connection to the encoder, verifies the data stream, and starts the listener thread.'''
        if self.is_encoder_connected():
            self.disconnect_from_encoder()
        try:
            self.encoder_serial = serial.Serial(port, 115200, timeout=2)
            time.sleep(0.5)
            line = self.encoder_serial.readline().decode('ascii').strip()
            parts = line.split(',')
            if len(parts) == len(self.articulated_chain) and all(p.strip().lstrip('-').isdigit() for p in parts):
                logging.info(f"Encoder handshake successful on {port}. Received valid data: '{line}'")
                self.stop_encoder_thread_flag.clear()
                self.encoder_listener_thread = threading.Thread(target=self._encoder_listener_thread_func)
                self.encoder_listener_thread.daemon = True
                self.encoder_listener_thread.start()
                return True
            else:
                logging.error(f"Encoder handshake failed on {port}. Expected 6 comma-separated integers, got '{line}'. Is this the correct port?")
                self.encoder_serial.close()
                self.encoder_serial = None
                return False
        except (serial.SerialException, UnicodeDecodeError, ValueError) as e:
            logging.error(f"Failed to connect to Encoder on port {port}: {e}")
            if self.encoder_serial:
                self.encoder_serial.close()
            self.encoder_serial = None
            return False

    def disconnect_from_encoder(self) -> None:
        '''Stops the listener thread and closes the encoder serial connection.'''
        if self.encoder_listener_thread and self.encoder_listener_thread.is_alive():
            self.stop_encoder_thread_flag.set()
            self.encoder_listener_thread.join(timeout=1.0)
            if self.encoder_listener_thread.is_alive():
                logging.warning("Encoder listener thread did not terminate cleanly.")
        if self.encoder_serial and self.encoder_serial.is_open:
            self.encoder_serial.close()
            logging.info("Encoder serial port closed.")
        self.encoder_serial = None
        self.encoder_listener_thread = None
        self.stop_encoder_thread_flag.clear()
        logging.info("Disconnected from Encoder.")

    def is_encoder_connected(self) -> bool:
        '''Checks if the serial connection to the encoder is active.'''
        return self.encoder_serial is not None and self.encoder_serial.is_open

    def send_command_to_robot(self, command: str) -> bool:
        '''Sends a raw command string to the motor controller, terminated by a newline.'''
        if not self.is_motor_controller_connected():
            logging.warning(f"Cannot send command '{command}': Robot not connected.")
            return False
        try:
            full_command = f"{command}\n"
            self.motor_controller_serial.write(full_command.encode('ascii'))
            return True
        except Exception as e:
            logging.error(f"Failed to send command '{command}': {e}")
            return False

    def get_current_positions(self) -> Optional[List[int]]:
        '''Sends 'P' to the robot and returns the current step positions for all joints.'''
        if not self.is_motor_controller_connected():
            return None
        try:
            self.send_command_to_robot("P")
            response = self.motor_controller_serial.readline().decode('ascii').strip()
            if not response:
                return None
            positions = [int(p.strip()) for p in response.split(',')]
            return positions[:self.num_joints]
        except Exception as e:
            logging.warning(f"Could not get robot position. Error: {e}")
            return None

    def send_command_to_encoder(self, command: str) -> bool:
        '''Sends a raw command string to the encoder, terminated by a newline.'''
        if not self.is_encoder_connected():
            logging.warning(f"Cannot send command to encoder '{command}': Not connected.")
            return False
        try:
            full_command = f"{command}\n"
            self.encoder_serial.write(full_command.encode('ascii'))
            return True
        except Exception as e:
            logging.error(f"Failed to send command '{command}' to encoder: {e}")
            return False

    def send_target_pose_commands(self, target_pose_steps: np.ndarray):
        '''Sends individual move commands for each joint to reach a target pose.'''
        if not self.is_motor_controller_connected():
            return
        logging.info(f"Sending robot to target steps: {target_pose_steps.tolist()}")
        self.robot_state = self.STATE_MOVING
        for idx, pos in enumerate(target_pose_steps):
            joint_def = self.get_joint_definition(idx)
            command = f"{joint_def['command_letter']}{int(pos)}"
            self.send_command_to_robot(command)

    def send_stop_commands(self):
        '''Sends commands to halt robot motion by commanding it to its last known position.'''
        if not self.is_motor_controller_connected():
            return
        current_positions = self.last_known_mc_position or self.get_current_positions()
        if not current_positions:
            logging.error("Cannot send stop command: failed to get current position.")
            return
        logging.info(f"Sending soft stop/hold command to position: {current_positions[:self.num_joints]}")
        for i, pos in enumerate(current_positions[:self.num_joints]):
            joint_def = self.get_joint_definition(i)
            command = f"{joint_def['command_letter']}{int(pos)}"
            self.send_command_to_robot(command)

    def zero_hardware_pose(self):
        '''Sends commands to zero both the encoder's and the motor controller's internal step counts.'''
        logging.info("Attempting to zero hardware poses...")
        if not self.is_encoder_connected() or not self.is_motor_controller_connected():
            slicer.util.errorDisplay("Both the Encoder and Motor Controller must be connected to zero the hardware.")
            return
        try:
            logging.info("Sending 'R' to zero encoder.")
            self.encoder_serial.write(b'R\n')
        except Exception as e:
            logging.error(f"Failed to send zero command to encoder: {e}")
            slicer.util.errorDisplay(f"Failed to send zero command to encoder: {e}")
            return
        command = "S" + ",".join(["0"] * 8)
        logging.info(f"Sending '{command}' to zero motor controller.")
        if not self.send_command_to_robot(command):
            slicer.util.errorDisplay("Failed to send zero command to motor controller.")
            return
        slicer.util.infoDisplay("Hardware zero commands sent successfully.")

    def sync_controller_position(self, true_position_steps: List[int]):
        '''Sends the 'S' command to force the motor controller's position to match the encoder's position.'''
        if not self.is_motor_controller_connected():
            logging.warning("Cannot sync controller position: not connected.")
            return
        command_payload = ",".join(map(str, true_position_steps)) + ",0,0"
        logging.info(f"Syncing controller position with command: S{command_payload}")
        self.send_command_to_robot(f"S{command_payload}")

    def _encoder_listener_thread_func(self):
        '''Continuously reads and parses encoder data from the serial port in a background thread.'''
        logging.info("Encoder listener thread started.")
        while not self.stop_encoder_thread_flag.is_set():
            try:
                if self.encoder_serial and self.encoder_serial.is_open:
                    line_bytes = self.encoder_serial.readline()
                    if not line_bytes:
                        continue
                    line = line_bytes.decode('ascii').strip()
                    if not line:
                        continue
                    if not (line[0].isdigit() or line[0] == '-'):
                        logging.info(f"Received status message from encoder: '{line}'")
                        continue
                    parts = line.split(',')
                    if len(parts) == self.num_joints:
                        new_pos = [int(p.strip()) for p in parts]
                        with self.encoder_data_lock:
                            self.true_encoder_position = new_pos
                    else:
                        logging.warning(f"Skipping malformed encoder data (wrong number of parts): '{line}'")
            except (serial.SerialException, UnicodeDecodeError, ValueError) as e:
                if not self.stop_encoder_thread_flag.is_set():
                    logging.exception(f"An error occurred in the encoder listener thread: {e}. The thread will attempt to continue.")
                else:
                    break
        logging.info("Encoder listener thread stopped.")

    def _perform_sync_check(self):
        '''Periodically checks for and corrects discrepancies between encoder and motor controller positions after a movement.'''
        if not (self.is_motor_controller_connected() and self.is_encoder_connected()):
            return
        with self.encoder_data_lock:
            current_pos = list(self.true_encoder_position)
        if self.last_sync_check_pos is None:
            self.last_sync_check_pos = current_pos
            return
        if np.any(np.array(current_pos) != np.array(self.last_sync_check_pos)):
            self.movement_detected_since_last_sync = True
            self.last_sync_check_pos = current_pos
            return
        if self.movement_detected_since_last_sync:
            logging.info("Robot is idle after a movement. Performing sync check.")
            controller_pos = self.get_current_positions()
            if controller_pos is None:
                return
            needs_correction = np.any(np.abs(np.array(controller_pos) - np.array(current_pos)) > self.DISCREPANCY_THRESHOLD)
            if needs_correction:
                logging.info(f"Post-movement discrepancy detected. Encoder: {current_pos}, Controller: {controller_pos}. Correcting.")
                command_payload = ",".join(map(str, current_pos)) + ",0,0"
                self.send_command_to_robot(f"S{command_payload}")
            self.movement_detected_since_last_sync = False

    def volume_threshold_segmentation(self, pNode: 'MamriParameterNode') -> None:
        '''Segments the input MRI volume to isolate fiducials and the main anatomical structure.'''
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

    def joint_detection(self, pNode: 'MamriParameterNode') -> Dict[str, List[Dict]]:
        '''Identifies known L-shaped fiducial patterns from a list of detected points.'''
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

    def _handle_joint_detection_results(self, identified_joints_data: Dict[str, List[Dict]]):
        '''Creates and visualizes markup nodes for the successfully identified joint fiducials.'''
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

    def _get_baseplate_transform(self, pNode: 'MamriParameterNode', identified_joints_data: Dict[str, List[Dict]]) -> Optional[vtk.vtkMatrix4x4]:
        '''Determines the baseplate transform by either using a saved node or detecting it from the scan.'''
        baseplate_transform_matrix = vtk.vtkMatrix4x4()
        if pNode.useSavedBaseplate:
            logging.info("Attempting to use saved baseplate transform as requested.")
            saved_tf_node = slicer.mrmlScene.GetFirstNodeByName(self.SAVED_BASEPLATE_TRANSFORM_NODE_NAME)
            if saved_tf_node and isinstance(saved_tf_node, vtkMRMLLinearTransformNode):
                saved_tf_node.GetMatrixTransformToWorld(baseplate_transform_matrix)
                return baseplate_transform_matrix
            else:
                slicer.util.warningDisplay(f"'Use Saved Transform' is checked, but node '{self.SAVED_BASEPLATE_TRANSFORM_NODE_NAME}' was not found. Will attempt detection from scan.")
        if "Baseplate" in identified_joints_data:
            logging.info("Attempting to calculate baseplate transform from detected fiducials.")
            baseplate_def = self.robot_definition_dict["Baseplate"]
            alignment_matrix = self._calculate_fiducial_alignment_matrix("BaseplateFiducials", baseplate_def["local_marker_coords"])
            if alignment_matrix:
                baseplate_transform_matrix.DeepCopy(alignment_matrix)
                return baseplate_transform_matrix
        logging.warning("Baseplate fiducials not detected in scan. Attempting fallback to saved transform.")
        saved_tf_node = slicer.mrmlScene.GetFirstNodeByName(self.SAVED_BASEPLATE_TRANSFORM_NODE_NAME)
        if saved_tf_node and isinstance(saved_tf_node, vtkMRMLLinearTransformNode):
            saved_tf_node.GetMatrixTransformToWorld(baseplate_transform_matrix)
            slicer.util.infoDisplay("Baseplate not found in scan; successfully used saved transform instead.")
            return baseplate_transform_matrix
        logging.error("CRITICAL: Could not determine baseplate transform from scan or saved node.")
        slicer.util.errorDisplay("Pose estimation failed. A scan containing the baseplate is required, or a previously saved baseplate transform must exist.")
        return None

    def _solve_full_chain_ik(self, joint6_effector_target_node: vtkMRMLMarkupsFiducialNode, apply_correction: bool) -> Optional[np.ndarray]:
        '''Solves the inverse kinematics for the entire robot chain to match observed marker positions.'''
        chain_defs = [self.robot_definition_dict[name] for name in self.articulated_chain]
        joint6_def = self.robot_definition_dict["Joint6"]
        joint6_target_mri = [joint6_effector_target_node.GetNthControlPointPositionWorld(i) for i in range(3)]
        base_node = self.jointTransformNodes.get("Baseplate")
        if not base_node:
            logging.error("Full-chain IK requires the Baseplate transform node."); return None
        tf_base_to_world = vtk.vtkMatrix4x4(); base_node.GetMatrixTransformToWorld(tf_base_to_world)
        joint4_target_mri, joint4_def = None, None
        joint4_fiducials_node = slicer.mrmlScene.GetFirstNodeByName("Joint4Fiducials")
        if joint4_fiducials_node and joint4_fiducials_node.GetNumberOfControlPoints() == 3:
            logging.info("joint4 markers detected. Adding as a secondary objective to the IK solver.")
            joint4_def = self.robot_definition_dict["Joint4"]
            joint4_target_mri = [joint4_fiducials_node.GetNthControlPointPositionWorld(i) for i in range(3)]
        initial_guesses = [self._get_current_joint_angles(self.articulated_chain), [0.0] * len(self.articulated_chain)]
        best_result, lowest_cost, best_guess_index = None, float('inf'), -1
        self.last_ik_error = None
        for i, initial_guess in enumerate(initial_guesses):
            try:
                result = scipy.optimize.least_squares(
                    self._full_chain_ik_error_function, initial_guess, bounds=([math.radians(j["joint_limits"][0]) for j in chain_defs], [math.radians(j["joint_limits"][1]) for j in chain_defs]),
                    args=(self.articulated_chain, joint6_target_mri, tf_base_to_world, joint6_def, apply_correction, joint4_target_mri, joint4_def),
                    method='trf', ftol=1e-6, xtol=1e-6, verbose=0)
                if result.success and result.cost < lowest_cost:
                    lowest_cost = result.cost
                    best_result = result
                    best_guess_index = i
            except Exception as e:
                logging.error(f"Scipy optimization for attempt #{i+1} failed: {e}")
        if not best_result:
            logging.error("Full-chain IK failed to converge for all initial guesses.")
            return None
        final_angles_rad = best_result.x
        final_error_vector = self._full_chain_ik_error_function(final_angles_rad, self.articulated_chain, joint6_target_mri, tf_base_to_world, joint6_def, apply_correction)
        self.last_ik_error = np.sqrt(np.mean(np.square(final_error_vector)))
        self._log_ik_solution_details(final_angles_rad, self.articulated_chain, tf_base_to_world, joint6_def, joint6_target_mri, apply_correction, joint4_def, joint4_target_mri, best_guess_index)
        return final_angles_rad

    def _build_robot_model(self, baseplate_transform_matrix: vtk.vtkMatrix4x4):
        '''Creates and assembles all 3D Slicer nodes (models, transforms) for the robot.'''
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

    def setRobotPose(self, joint_angles_rad: np.ndarray):
        '''Sets the robot's 3D model to a specific configuration of joint angles.'''
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

    def _get_world_transform_for_joint(self, joint_angles_rad: Dict[str, float], target_joint_name: str, base_transform_matrix: vtk.vtkMatrix4x4) -> Optional[vtk.vtkMatrix4x4]:
        '''Calculates the world transform of a specific joint using forward kinematics.'''
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

    def _full_chain_ik_error_function(self, angles_rad, articulated_joint_names, joint6_target_ras, base_transform, joint6_def, apply_correction: bool, joint4_target_ras=None, joint4_def=None, joint4_weight=0.05):
        '''Calculates the error between predicted and observed marker positions for the full chain IK solver.'''
        joint_values_rad = {name: angle for name, angle in zip(articulated_joint_names, angles_rad)}
        joint6_local_coords = list(joint6_def["local_marker_coords"])
        if apply_correction:
            rotation_transform = vtk.vtkTransform()
            rotation_transform.RotateZ(180)
            joint6_local_coords = [rotation_transform.TransformPoint(p) for p in joint6_local_coords]
        tf_joint6_model_to_world = self._get_world_transform_for_joint(joint_values_rad, joint6_def["name"], base_transform)
        if tf_joint6_model_to_world is None:
            num_errors = len(joint6_local_coords) * 3
            if joint4_target_ras: num_errors += len(joint4_target_ras) * 3
            return [1e6] * num_errors
        joint6_errors = []
        for i, local_p in enumerate(joint6_local_coords):
            pred_p_h = tf_joint6_model_to_world.MultiplyPoint(list(local_p) + [1.0])
            pred_ras = [c / pred_p_h[3] for c in pred_p_h[:3]]
            joint6_errors.extend([pred_ras[j] - joint6_target_ras[i][j] for j in range(3)])
        joint4_errors = []
        if joint4_target_ras and joint4_def:
            joint4_local_coords = joint4_def["local_marker_coords"]
            tf_joint4_model_to_world = self._get_world_transform_for_joint(joint_values_rad, joint4_def["name"], base_transform)
            if tf_joint4_model_to_world:
                for i, local_p in enumerate(joint4_local_coords):
                    pred_p_h = tf_joint4_model_to_world.MultiplyPoint(list(local_p) + [1.0])
                    pred_ras = [c / pred_p_h[3] for c in pred_p_h[:3]]
                    joint4_errors.extend([joint4_weight * (pred_ras[j] - joint4_target_ras[i][j]) for j in range(3)])
            else:
                joint4_errors = [1e4] * len(joint4_local_coords) * 3
        return joint6_errors + joint4_errors

    def _ik_pose_and_collision_error_function(self, angles_rad, articulated_joint_names, target_transform, base_transform, body_polydata):
        '''Calculates position and orientation error for trajectory IK, with a high penalty for collisions.'''
        joint_values_rad = {name: angle for name, angle in zip(articulated_joint_names, angles_rad)}
        if self._check_collision(joint_values_rad, base_transform, body_polydata):
            return [1e4] * 6
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
        '''Checks for collision between the robot's collision models and the patient's body model.'''
        if not body_polydata or body_polydata.GetNumberOfPoints() == 0:
            return False
        parts_to_check = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]
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

    def _load_robot_definition(self) -> List[Dict]:
        '''Loads the robot's kinematic and geometric definition from a JSON file.'''
        try:
            module_name = self.__class__.__name__.replace('Logic', '')
            module_path = slicer.util.modulePath(module_name)
            module_dir = os.path.dirname(module_path)
        except Exception as e:
            logging.error(f"Could not determine module path using slicer.util.modulePath: {e}")
            try:
                module_dir = os.path.dirname(__file__)
            except NameError:
                logging.error("Could not determine module path. Fallback failed.")
                return []
        base_path = os.path.join(module_dir, "Resources", "Robot")
        config_path = os.path.join(module_dir, "Resources", "Robot", "robot_config.json")
        if not os.path.exists(config_path):
            logging.error(f"FATAL: Robot config file not found at {config_path}")
            return []
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        for joint in config_data:
            if joint.get("stl_path"):
                joint["stl_path"] = os.path.join(base_path, joint["stl_path"])
            if joint.get("collision_stl_path"):
                joint["collision_stl_path"] = os.path.join(base_path, joint["collision_stl_path"])
            offset_data = joint.get("fixed_offset_to_parent")
            if isinstance(offset_data, dict):
                transform = vtk.vtkTransform()
                if 'translate' in offset_data:
                    transform.Translate(offset_data['translate'])
                if 'rotate' in offset_data:
                    for axis, angle_deg in offset_data['rotate']:
                        getattr(transform, f"Rotate{axis.upper()}", lambda a: None)(angle_deg)
                joint["fixed_offset_to_parent"] = transform.GetMatrix()
            else:
                joint["fixed_offset_to_parent"] = None
        return config_data

    def get_joint_definition(self, joint_index: int) -> dict:
        '''Returns the definition dictionary for a joint by its index.'''
        if 0 <= joint_index < len(self.articulated_chain):
            return self.robot_definition_dict[self.articulated_chain[joint_index]]
        return {}

    def _convert_angles_to_steps_array(self, joint_angles_rad: np.ndarray) -> np.ndarray:
        '''Converts an array of joint angles in radians to an array of motor steps.'''
        steps_array = np.zeros(len(joint_angles_rad), dtype=int)
        for i, angle_rad in enumerate(np.asarray(joint_angles_rad).flatten()):
            joint_def = self.robot_definition_dict[self.articulated_chain[i]]
            steps_per_rev = joint_def.get("steps_per_rev", 0)
            if steps_per_rev > 0:
                steps_array[i] = int(angle_rad * (steps_per_rev / (2.0 * math.pi)))
        return steps_array

    def _convert_steps_to_angle_rad(self, steps: int, joint_index: int) -> float:
        '''Converts motor steps for a specific joint back to radians.'''
        joint_def = self.robot_definition_dict[self.articulated_chain[joint_index]]
        steps_per_rev = joint_def.get("steps_per_rev", 0)
        if steps_per_rev > 0:
            return float(steps) * ((2.0 * math.pi) / steps_per_rev)
        return 0.0

    def _convert_steps_array_to_angles(self, steps_array: np.ndarray) -> np.ndarray:
        '''Converts an array of motor steps to an array of joint angles in radians.'''
        angles_rad_array = np.zeros(len(steps_array), dtype=float)
        for i, steps in enumerate(np.asarray(steps_array).flatten()):
            angles_rad_array[i] = self._convert_steps_to_angle_rad(steps, i)
        return angles_rad_array

    def _cleanup_module_nodes(self):
        '''Removes all nodes created by the module during a previous run.'''
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

    def _clear_node_by_name(self, name: str):
        '''Removes all nodes with a specific name from the scene.'''
        while node := slicer.mrmlScene.GetFirstNodeByName(name):
            slicer.mrmlScene.RemoveNode(node)

    def _organize_node_in_subject_hierarchy(self, mrmlNode, parentFolderName: str, subFolderName: Optional[str] = None):
        '''Organizes a given MRML node into a specified folder structure in the Subject Hierarchy.'''
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
        '''Finds and re-links existing robot model nodes when the module is reloaded.'''
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

    def _load_collision_models(self):
        '''Loads simplified STL models for use in collision detection.'''
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

    def _create_model_and_articulation_transform(self, jn: str, stl: str, tf_mat: vtk.vtkMatrix4x4, color) -> Tuple[Optional[vtkMRMLModelNode], Optional[vtkMRMLLinearTransformNode]]:
        '''Creates a model node from an STL file and its associated parent transform node.'''
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

    def _get_rotation_transform(self, angle_deg: float, axis_str: Optional[str]) -> vtk.vtkTransform:
        '''Creates a VTK rotation transform, correcting for coordinate system inconsistencies.'''
        transform = vtk.vtkTransform()
        if axis_str == "IS":
            transform.RotateZ(angle_deg)
        elif axis_str == "PA":
            transform.RotateY(-angle_deg)
        elif axis_str == "LR":
            transform.RotateX(angle_deg)
        return transform

    def _calculate_fiducial_alignment_matrix(self, node_name: str, local_coords: List[Tuple[float,float,float]]) -> Optional[vtk.vtkMatrix4x4]:
        '''Calculates a rigid transform matrix to align a set of local points to a set of world points.'''
        fiducials_node = slicer.mrmlScene.GetFirstNodeByName(node_name)
        if not (fiducials_node and fiducials_node.GetNumberOfControlPoints() >= 3 and len(local_coords) >= 3): return None
        n_pts = min(fiducials_node.GetNumberOfControlPoints(), len(local_coords), 3)
        target = vtk.vtkPoints(); source = vtk.vtkPoints()
        for i in range(n_pts):
            target.InsertNextPoint(fiducials_node.GetNthControlPointPositionWorld(i)); source.InsertNextPoint(local_coords[i])
        tf = vtk.vtkLandmarkTransform(); tf.SetSourceLandmarks(source); tf.SetTargetLandmarks(target)
        tf.SetModeToRigidBody(); tf.Update(); return tf.GetMatrix()

    def _sort_l_shaped_markers(self, markers: List[Dict], len1: float, len2: float, tol: float) -> Optional[List[Dict]]:
        '''Sorts a triplet of markers into a consistent order (corner, short arm, long arm) based on distances.'''
        if len(markers) != 3: return None
        points = [{'data': m, 'ras': tuple(m["ras_coords"])} for m in markers]
        l_short, l_long = sorted((len1, len2))
        for i in range(3):
            corner, p1, p2 = points[i], points[(i+1)%3], points[(i+2)%3]
            d1, d2 = math.dist(corner['ras'], p1['ras']), math.dist(corner['ras'], p2['ras'])
            if abs(d1 - l_short) <= self.DISTANCE_TOLERANCE and abs(d2 - l_long) <= self.DISTANCE_TOLERANCE: return [corner['data'], p1['data'], p2['data']]
            if abs(d1 - l_long) <= self.DISTANCE_TOLERANCE and abs(d2 - l_short) <= self.DISTANCE_TOLERANCE: return [corner['data'], p2['data'], p1['data']]
        return None

    def _get_body_polydata(self, segmentationNode: vtkMRMLSegmentationNode) -> Optional[vtk.vtkPolyData]:
        '''Extracts the polygonal surface mesh from the body segmentation node.'''
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

    def _get_current_joint_angles(self, articulated_joint_names: List[str]) -> List[float]:
        '''Reads the current joint angles directly from the articulation transforms of the 3D model.'''
        angles_rad = []
        for name in articulated_joint_names:
            angle_rad = 0.0
            if tf_node := self.jointTransformNodes.get(name):
                m = vtk.vtkMatrix4x4()
                tf_node.GetMatrixTransformToParent(m)
                transform = vtk.vtkTransform(); transform.SetMatrix(m)
                orientation_rad = [math.radians(a) for a in transform.GetOrientation()]
                axis = self.robot_definition_dict[name].get("articulation_axis")
                if axis == "IS":
                    angle_rad = orientation_rad[2]
                elif axis == "PA":
                    angle_rad = -orientation_rad[1]
                elif axis == "LR":
                    angle_rad = orientation_rad[0]
            angles_rad.append(angle_rad)
        return angles_rad

    def _log_ik_solution_details(self, final_angles_rad, articulated_chain, base_transform, joint6_def, joint6_target_ras, apply_correction: bool, joint4_def=None, joint4_target_ras=None, best_guess_index=-1):
        '''Logs detailed information about the inverse kinematics solution, including errors and final angles.'''
        logging.info("--- IK Solution Details ---")
        guess_desc = "current pose" if best_guess_index == 0 else "zero pose" if best_guess_index == 1 else "unknown"
        logging.info(f"Solution found using initial guess #{best_guess_index+1} ({guess_desc}).")
        joint_values_rad = {name: angle for name, angle in zip(articulated_chain, final_angles_rad)}
        logging.info("Final Joint Angles (°):")
        for name, angle_deg in zip(articulated_chain, [math.degrees(a) for a in final_angles_rad]):
            logging.info(f"  - {name}: {angle_deg:.2f}°")
        if self.last_ik_error is not None:
            logging.info(f"Final Root Mean Square Error (RMSE): {self.last_ik_error:.4f} mm")
        def log_point_comparison(joint_name, joint_def, target_ras_list, fk_transform, correction):
            logging.info(f"--- Comparison for {joint_name} Markers ---")
            local_coords = list(joint_def["local_marker_coords"])
            if correction and joint_name == "Joint6":
                rotation_transform = vtk.vtkTransform()
                rotation_transform.RotateZ(180)
                local_coords = [rotation_transform.TransformPoint(p) for p in local_coords]
            for i, local_p in enumerate(local_coords):
                pred_p_h = fk_transform.MultiplyPoint(list(local_p) + [1.0])
                pred_ras = np.array([c / pred_p_h[3] for c in pred_p_h[:3]])
                target_ras = np.array(target_ras_list[i])
                error = np.linalg.norm(pred_ras - target_ras)
                logging.info(f"  Marker {i+1}:")
                logging.info(f"    - Target RAS:   ({target_ras[0]:.2f}, {target_ras[1]:.2f}, {target_ras[2]:.2f})")
                logging.info(f"    - Predicted RAS: ({pred_ras[0]:.2f}, {pred_ras[1]:.2f}, {pred_ras[2]:.2f})")
                logging.info(f"    - Point Error:  {error:.3f} mm")
        tf_joint6_world = self._get_world_transform_for_joint(joint_values_rad, "Joint6", base_transform)
        if tf_joint6_world:
            log_point_comparison("Joint6", joint6_def, joint6_target_ras, tf_joint6_world, apply_correction)
        if joint4_def and joint4_target_ras:
            tf_joint4_world = self._get_world_transform_for_joint(joint_values_rad, "Joint4", base_transform)
            if tf_joint4_world:
                log_point_comparison("Joint4", joint4_def, joint4_target_ras, tf_joint4_world, False)
        logging.info("--- End IK Solution Details ---")

    def _toggle_mri_fiducials(self, checked: bool):
        '''Toggles the visibility of the fiducial markers detected from the MRI scan.'''
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
        '''Toggles the visibility of the local coordinate system debug markers for each joint.'''
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
        '''Toggles the visibility of the 3D robot models.'''
        for model_node in self.jointModelNodes.values():
            if disp := model_node.GetDisplayNode(): disp.SetVisibility(checked)

    def _visualize_all_joint_markers_from_fk(self):
        '''Visualizes the local coordinate markers for all joints based on the current robot pose.'''
        for joint_def in self.robot_definition:
            if joint_def.get("has_markers"): self._visualize_joint_local_markers_in_world(joint_def["name"])

    def _visualize_joint_local_markers_in_world(self, joint_name: str):
        '''Visualizes the local coordinate markers for a single joint in the world frame.'''
        debug_node_name = f"{joint_name}_LocalMarkers_WorldView_DEBUG"; self._clear_node_by_name(debug_node_name)
        joint_def = self.robot_definition_dict.get(joint_name); joint_art_node = self.jointTransformNodes.get(joint_name)
        local_coords = joint_def.get("local_marker_coords") if joint_def else None
        if not all([joint_def, joint_art_node, local_coords]): return
        tf_model_to_world = vtk.vtkMatrix4x4(); joint_art_node.GetMatrixTransformToWorld(tf_model_to_world)
        debug_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", debug_node_name)
        self._organize_node_in_subject_hierarchy(debug_node, self.MASTER_FOLDER_NAME, "Debug Markers")
        if disp := debug_node.GetDisplayNode():
            disp.SetVisibility(self.debug_markers_visible)
            disp.SetGlyphScale(3.0); disp.SetTextScale(3.5); r,g,b = joint_def.get("color", (0.1,0.8,0.8))
            disp.SetSelectedColor(r*0.7, g*0.7, b*0.7); disp.SetColor(r,g,b); disp.SetOpacity(1)
        prefix = "".join(w[0] for w in joint_name.split() if w)[:3].upper()
        for i, local_p in enumerate(local_coords):
            world_p_h = tf_model_to_world.MultiplyPoint(list(local_p) + [1.0]); world_ras = [c/world_p_h[3] for c in world_p_h[:3]]
            idx = debug_node.AddControlPoint(world_ras); debug_node.SetNthControlPointLabel(idx, f"{prefix}_Lm{i+1}")

    def _visualize_trajectory_line(self, target_pos, standoff_pos) -> Optional[vtkMRMLMarkupsFiducialNode]:
        '''Creates or updates a line markup to visualize the needle's trajectory path.'''
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