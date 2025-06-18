import logging
import os
import itertools
import math
from typing import Annotated, Optional, Dict, List, Tuple
import time
import threading

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
from slicer import vtkMRMLLabelMapVolumeNode


#
# Mamri (Module Class)
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

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/Mamri.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        
        if hasattr(self.ui, "applyButton"):
            self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        else:
            logging.error("MamriWidget.setup: 'applyButton' not found in UI. Processing cannot be triggered.")
        if hasattr(self.ui, "planTrajectoryButton"):
            self.ui.planTrajectoryButton.connect("clicked(bool)", self.onPlanTrajectoryButton)
        else:
            logging.error("MamriWidget.setup: 'planTrajectory' not found in UI. Processing cannot be triggered.")    
        if hasattr(self.ui, "drawFiducialsCheckBox"):
            self.ui.drawFiducialsCheckBox.connect("toggled(bool)", self.onDrawFiducialsCheckBoxToggled)
        else:
            logging.warning("MamriWidget.setup: 'drawFiducialsCheckBox' not found in UI.")
        if hasattr(self.ui, "drawModelsCheckBox"):
            self.ui.drawModelsCheckBox.connect("toggled(bool)", self.onDrawModelsCheckBoxToggled)
        else:
            logging.warning("MamriWidget.setup: 'drawModelsCheckBox' not found in UI.")

        if hasattr(self.ui, "useSavedBaseplateCheckBox"):
            self.ui.useSavedBaseplateCheckBox.setToolTip(_("If checked, use the previously saved baseplate transform instead of detecting it from the current scan."))
        else:
            logging.warning("MamriWidget.setup: 'useSavedBaseplateCheckBox' not found in UI.")

        if hasattr(self.ui, "saveBaseplateButton"):
            self.ui.saveBaseplateButton.setToolTip(_("Saves the current Baseplate transform for later use. Requires the Baseplate to be detected and processed in the current scan."))
            self.ui.saveBaseplateButton.connect("clicked(bool)", self.onSaveBaseplateButton)
        else:
            logging.warning("MamriWidget.setup: 'saveBaseplateButton' not found in UI.")
        
        if hasattr(self.ui, "findEntryPointButton"):
            self.ui.findEntryPointButton.setToolTip(_("Automatically find the point on the patient's skin closest to the target marker and set it as the Entry Point."))
            self.ui.findEntryPointButton.connect("clicked(bool)", self.onFindEntryPointButton)
        else:
            logging.warning("MamriWidget.setup: 'findEntryPointButton' not found in UI.")
            
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = MamriLogic()
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.initializeParameterNode()

    def cleanup(self) -> None: self.removeObservers()
    def enter(self) -> None: 
        self.initializeParameterNode()
        self._checkCanPlanTrajectory()

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
            if self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply):
                self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            if self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanPlanTrajectory):
                self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanPlanTrajectory)

    def setParameterNode(self, inputParameterNode: Optional[MamriParameterNode]) -> None:
        self.remove_parameter_node_observers()
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanPlanTrajectory)
        self._checkCanApply()
        self._checkCanPlanTrajectory()

    def _checkCanApply(self, caller=None, event=None) -> None:
        can_apply = self._parameterNode and self._parameterNode.inputVolume is not None
        if hasattr(self.ui, "applyButton") and self.ui.applyButton:
            self.ui.applyButton.enabled = can_apply
            tooltip = _("Run fiducial detection and robot model rendering with IK.") if can_apply else _("Select an input volume node.")
            self.ui.applyButton.toolTip = tooltip

    def _checkCanPlanTrajectory(self, caller=None, event=None) -> None:
        if not hasattr(self.ui, "planTrajectoryButton"):
            return

        can_plan_base = (self._parameterNode and
                         self._parameterNode.targetFiducialNode and
                         self._parameterNode.targetFiducialNode.GetNumberOfControlPoints() > 0 and
                         self._parameterNode.entryPointFiducialNode and
                         self._parameterNode.entryPointFiducialNode.GetNumberOfControlPoints() > 0 and
                         self._parameterNode.segmentationNode)

        model_is_built = self.logic and self.logic.jointTransformNodes.get("Baseplate") is not None
        can_plan_total = can_plan_base and model_is_built
        self.ui.planTrajectoryButton.enabled = bool(can_plan_total)

        tooltip = ""
        if not model_is_built:
            tooltip = _("Please run 'Start robot pose estimation' first to build the robot model.")
        elif not self._parameterNode.segmentationNode:
            tooltip = _("Select a body segmentation node to enable planning with collision detection.")
        elif not (self._parameterNode.targetFiducialNode and self._parameterNode.entryPointFiducialNode):
            tooltip = _("Select a target marker and an entry marker to enable planning.")
        else:
            tooltip = _("Calculate the robot joint angles to align the needle, with collision avoidance.")
        self.ui.planTrajectoryButton.toolTip = tooltip


    def onApplyButton(self) -> None:
        if not self._parameterNode:
            slicer.util.errorDisplay("Parameter node is not initialized.")
            return
        
        models_visible = self.ui.drawModelsCheckBox.isChecked() if hasattr(self.ui, "drawModelsCheckBox") else True
        markers_visible = self.ui.drawFiducialsCheckBox.isChecked() if hasattr(self.ui, "drawFiducialsCheckBox") else True
        
        self.logic.process(self._parameterNode, models_visible=models_visible, markers_visible=markers_visible)
        self._checkCanPlanTrajectory()
    
    def onPlanTrajectoryButton(self) -> None:
        if not self._parameterNode:
            slicer.util.errorDisplay("Parameter node is not initialized."); return
        self.logic.planTrajectory(self._parameterNode)

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
        self.logic._toggle_robot_markers(checked)

    def onDrawModelsCheckBoxToggled(self, checked: bool) -> None:
        self.logic._toggle_robot_models(checked)


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

        self.robot_definition = self._define_robot_structure()
        self.robot_definition_dict = {joint["name"]: joint for joint in self.robot_definition}
        
        self.SAVED_BASEPLATE_TRANSFORM_NODE_NAME = "MamriSavedBaseplateTransform"
        self.TARGET_POSE_TRANSFORM_NODE_NAME = "MamriTargetPoseTransform_DEBUG"
        self.TRAJECTORY_LINE_NODE_NAME = "TrajectoryLine_DEBUG"

        self.DEBUG_COLLISIONS = False 
        if self.DEBUG_COLLISIONS:
            logging.warning("MamriLogic collision debugging is enabled. This will create temporary models in the scene.")

        self.ik_chains_config = [
            {"parent_of_proximal": "Baseplate", "proximal": "Shoulder1", "distal_with_markers": "Link1", "log_name": "Shoulder1/Link1"},
            {"parent_of_proximal": "Link1", "proximal": "Shoulder2", "distal_with_markers": "Elbow1", "log_name": "Shoulder2/Elbow1"},
            {"parent_of_proximal": "Elbow1", "proximal": "Wrist", "distal_with_markers": "End", "log_name": "Wrist/End"},
        ]
        
        self._discover_robot_nodes_in_scene()

    def _discover_robot_nodes_in_scene(self):
        """
        Populates the logic's node dictionaries by finding existing robot-related nodes in the scene.
        This is useful for state recovery after a module reload.
        """
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
            {"name": "Shoulder1", "stl_path": os.path.join(base_path, "Shoulder1.STL"), "collision_stl_path": os.path.join(base_path, "Shoulder1_collision.STL"), "parent": "Baseplate", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 20.0)), "has_markers": False, "color": (0, 0.5, 0), "articulation_axis": "IS", "joint_limits": (-180, 180)},
            {"name": "Link1", "stl_path": os.path.join(base_path, "Link1.STL"), "collision_stl_path": os.path.join(base_path, "Link1_collision.STL"), "parent": "Shoulder1", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 30)), "has_markers": True, "local_marker_coords": [(12.5, 45.0, 110.0), (-12.5, 45.0, 110.0), (12.5, 45.0, 40.0)], "arm_lengths": (70.0, 25.0), "color": (0, 1, 0), "articulation_axis": "PA", "joint_limits": (-120, 120)},
            {"name": "Shoulder2", "stl_path": os.path.join(base_path, "Shoulder2.STL"), "collision_stl_path": os.path.join(base_path, "Shoulder2_collision.STL"), "parent": "Link1", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 150)), "has_markers": False, "color": (0, 0.5, 0), "articulation_axis": "PA", "joint_limits": (-120, 120)},
            {"name": "Elbow1", "stl_path": os.path.join(base_path, "Elbow1.STL"), "collision_stl_path": os.path.join(base_path, "Elbow1_collision.STL"), "parent": "Shoulder2", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 0)), "has_markers": True, "local_marker_coords": [(10, 35.0, 85), (-10, 35.0, 85), (-10, -35.0, 85)], "arm_lengths": (70.0, 20.0),  "color": (0, 1, 0), "articulation_axis": "IS", "joint_limits": (-180, 180)},
            {"name": "Wrist", "stl_path": os.path.join(base_path, "Wrist.STL"), "collision_stl_path": os.path.join(base_path, "Wrist_collision.STL"), "parent": "Elbow1", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 150)), "has_markers": False, "color": (0, 0.5, 0), "articulation_axis": "PA", "joint_limits": (-120, 120)},
            {"name": "End", "stl_path": os.path.join(base_path, "End.STL"), "collision_stl_path": os.path.join(base_path, "End_collision.STL"), "parent": "Wrist", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 8)), "has_markers": True, "local_marker_coords": [(-10, 22.5, 26), (10, 22.5, 26), (-10, -22.5, 26)], "arm_lengths": (45.0, 20.0), "color": (1, 0, 0), "articulation_axis": "IS", "joint_limits": (-180, 180)},
            {"name": "Needle", "stl_path": os.path.join(base_path, "Needle.STL"), "collision_stl_path": os.path.join(base_path, "Needle_collision.STL"), "parent": "End", "fixed_offset_to_parent": self._create_offset_matrix((-50, 0, 71)), "has_markers": False, "color": (1, 0, 0), "articulation_axis": "TRANS_X", "joint_limits": (0, 0), "needle_tip_local": (0, 0, 0), "needle_axis_local": (1, 0, 0)}
        ]

    def getParameterNode(self) -> MamriParameterNode:
        return MamriParameterNode(super().getParameterNode())

    def _clear_node_by_name(self, name: str):
        while node := slicer.mrmlScene.GetFirstNodeByName(name):
            slicer.mrmlScene.RemoveNode(node)

    def process(self, parameterNode: MamriParameterNode, models_visible: bool, markers_visible: bool) -> None:
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
                    logging.info(f"Using saved baseplate transform from '{self.SAVED_BASEPLATE_TRANSFORM_NODE_NAME}'.")
                else:
                    slicer.util.warningDisplay(f"'Use Saved Transform' is checked, but node '{self.SAVED_BASEPLATE_TRANSFORM_NODE_NAME}' was not found.")
            if not baseplate_transform_found:
                if "Baseplate" in identified_joints_data:
                    baseplate_def = self.robot_definition_dict["Baseplate"]
                    alignment_matrix = self._calculate_fiducial_alignment_matrix("BaseplateFiducials", baseplate_def["local_marker_coords"])
                    if alignment_matrix:
                        baseplate_transform_matrix.DeepCopy(alignment_matrix)
                        baseplate_transform_found = True
                        logging.info("Calculated baseplate transform from detected fiducials.")
                    else: logging.warning("Could not calculate baseplate transform from fiducials.")
                else: logging.info("Baseplate fiducials not detected and not using a saved transform. Baseplate will be at origin.")
            self._build_robot_model(baseplate_transform_matrix)
            if baseplate_transform_found and "End" in identified_joints_data:
                logging.info("Attempting full-chain IK from baseplate to detected 'End' joint.")
                end_fiducials_node = slicer.mrmlScene.GetFirstNodeByName("EndFiducials")
                if end_fiducials_node and end_fiducials_node.GetNumberOfControlPoints() == 3:
                    self._solve_full_chain_ik(end_fiducials_node)
                    self._visualize_all_joint_markers_from_fk()
                else:
                    logging.warning("Could not perform full-chain IK. 'End' fiducials not found or incomplete. Falling back to sequential IK.")
                    self._solve_all_ik_chains(identified_joints_data)
            else:
                logging.info("Prerequisites for full-chain IK not met. Using standard sequential IK for any detected intermediate joints.")
                self._solve_all_ik_chains(identified_joints_data)
            logging.info("Mamri processing finished.")

    def _get_body_polydata(self, segmentationNode: vtkMRMLSegmentationNode) -> Optional[vtk.vtkPolyData]:
        """Extracts the 'Body' segment's polydata from a segmentation node."""
        if not segmentationNode:
            logging.warning("No segmentation node provided to get body polydata.")
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
                logging.info(f"Creating '{representationName}' for segmentation...")
                segmentation.CreateRepresentation(representationName)
            segmentationNode.GetClosedSurfaceRepresentation(bodySegmentID, body_poly)
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to get surface representation from segmentation: {e}")
            return None

        if body_poly.GetNumberOfPoints() == 0:
            slicer.util.warningDisplay("The 'Body' segment polydata is empty.")
            return None

        return body_poly

    def findAndSetEntryPoint(self, pNode: 'MamriParameterNode') -> None:
        targetNode = pNode.targetFiducialNode
        segmentationNode = pNode.segmentationNode

        if not (targetNode and targetNode.GetNumberOfControlPoints() > 0 and segmentationNode):
            slicer.util.errorDisplay("Please select a body segmentation and place a target marker first.")
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
        logging.info(f"Searching for accessible entry points within a {search_radius}mm radius of the target.")
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
            slicer.util.warningDisplay(f"Could not find a suitable entry point on an accessible surface within a {search_radius}mm radius of the target. Please ensure the target is placed correctly.")
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
        
        logging.info(f"Found accessible optimal entry point at {closest_point_coords}. Distance to target: {best_candidate['distance']:.2f} mm")
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
    
    def _visualize_collision_check(self, part_name: str, part_local_poly: vtk.vtkPolyData, part_transform: vtk.vtkMatrix4x4, body_poly: vtk.vtkPolyData):
        """Creates temporary models to visualize a single step of the collision check."""
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform = vtk.vtkTransform(); transform.SetMatrix(part_transform)
        transform_filter.SetInputData(part_local_poly); transform_filter.SetTransform(transform); transform_filter.Update()
        part_world_poly = transform_filter.GetOutput()
        
        part_debug_node_name = f"DEBUG_COLLISION_{part_name}"
        part_debug_node = slicer.mrmlScene.GetFirstNodeByName(part_debug_node_name)
        if not part_debug_node:
            part_debug_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", part_debug_node_name)
            part_debug_node.CreateDefaultDisplayNodes()
            disp_node = part_debug_node.GetDisplayNode()
            disp_node.SetColor(1, 1, 0); disp_node.SetOpacity(0.5); disp_node.SetSliceIntersectionVisibility(False)
        part_debug_node.SetAndObservePolyData(part_world_poly)
        part_debug_node.GetDisplayNode().SetVisibility(True)

        body_debug_node_name = "DEBUG_COLLISION_Body"
        body_debug_node = slicer.mrmlScene.GetFirstNodeByName(body_debug_node_name)
        if not slicer.mrmlScene.IsNodePresent(body_debug_node):
            body_debug_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", body_debug_node_name)
            body_debug_node.CreateDefaultDisplayNodes()
            disp_node = body_debug_node.GetDisplayNode()
            disp_node.SetColor(1, 0, 1); disp_node.SetOpacity(0.3); disp_node.SetSliceIntersectionVisibility(False)
            body_debug_node.SetAndObservePolyData(body_poly)
        body_debug_node.GetDisplayNode().SetVisibility(True)

    def _check_collision(self, joint_angles_rad: Dict[str, float], base_transform_vtk: vtk.vtkMatrix4x4, body_polydata: vtk.vtkPolyData) -> bool:
        """Checks for collision using pre-loaded simplified collision models."""
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
                logging.debug(f"COLLISION DETECTED for part: {part_name}")
                if self.DEBUG_COLLISIONS:
                    self._visualize_collision_check(part_name, robot_part_local_poly, part_world_transform_vtk, body_polydata)
                return True
        return False

    def _ik_pose_and_collision_error_function(self, angles_rad, articulated_joint_names, target_transform, base_transform, body_polydata):
        """Wraps the pose error function with a collision check, returning a large penalty on collision."""
        joint_values_rad = {name: angle for name, angle in zip(articulated_joint_names, angles_rad)}
        if self._check_collision(joint_values_rad, base_transform, body_polydata):
            return [1e4] * 6 
        return self._ik_pose_error_function(angles_rad, articulated_joint_names, target_transform, base_transform)

    def planTrajectory(self, pNode: MamriParameterNode) -> None:
        logging.info("Starting trajectory planning for needle alignment with collision avoidance...")
        targetNode, entryNode, segmentationNode = pNode.targetFiducialNode, pNode.entryPointFiducialNode, pNode.segmentationNode
        if not (targetNode and targetNode.GetNumberOfControlPoints() > 0 and entryNode and entryNode.GetNumberOfControlPoints() > 0):
            slicer.util.errorDisplay("Set a target and an entry marker to plan trajectory."); return
        if not segmentationNode:
            slicer.util.errorDisplay("A body segmentation is required for collision detection."); return

        body_polydata = self._get_body_polydata(segmentationNode)
        if not body_polydata:
            slicer.util.errorDisplay("Could not get body polydata from segmentation. Aborting."); return

        original_poly_count = body_polydata.GetNumberOfPolys()
        logging.info(f"Simplifying body model for collision detection (Original polys: {original_poly_count})...")
        
        decimator = vtk.vtkDecimatePro()
        decimator.SetInputData(body_polydata)
        decimator.SetTargetReduction(0.95)
        decimator.PreserveTopologyOn()
        decimator.Update()
        simplified_body_polydata = decimator.GetOutput()
        
        new_poly_count = simplified_body_polydata.GetNumberOfPolys()
        logging.info(f"...simplification complete (New polys: {new_poly_count}).")

        target_pos, entry_pos = np.array(targetNode.GetNthControlPointPositionWorld(0)), np.array(entryNode.GetNthControlPointPositionWorld(0))
        direction_vec = target_pos - entry_pos
        if np.linalg.norm(direction_vec) < 1e-6: slicer.util.errorDisplay("Entry and Target markers are at the same position."); return
        
        x_axis = direction_vec / np.linalg.norm(direction_vec)
        needle_tip_pos = entry_pos - (pNode.safetyDistance * x_axis)
        self._visualize_trajectory_line(target_pos, needle_tip_pos)
        
        up_vec = np.array([0, 0, 1.0]); 
        if abs(np.dot(x_axis, up_vec)) > 0.99: up_vec = np.array([0, 1.0, 0])
        y_axis = np.cross(up_vec, x_axis); y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis); target_matrix_np = np.identity(4)
        target_matrix_np[:3, 0] = x_axis; target_matrix_np[:3, 1] = y_axis; target_matrix_np[:3, 2] = z_axis; target_matrix_np[:3, 3] = needle_tip_pos
        target_transform_vtk = vtk.vtkMatrix4x4(); target_transform_vtk.DeepCopy(target_matrix_np.flatten())
        
        articulated_chain = ["Shoulder1", "Link1", "Shoulder2", "Elbow1", "Wrist", "End"]
        chain_defs = [self.robot_definition_dict[name] for name in articulated_chain]
        bounds_rad = [tuple(math.radians(l) for l in jdef["joint_limits"]) for jdef in chain_defs]
        bounds_lower, bounds_upper = zip(*bounds_rad)

        base_node = self.jointTransformNodes.get("Baseplate")
        if not base_node: slicer.util.errorDisplay("Cannot plan trajectory: Robot model not loaded or baseplate is missing."); return
        base_transform_vtk = vtk.vtkMatrix4x4(); base_node.GetMatrixTransformToWorld(base_transform_vtk)

        # --- PRE-FLIGHT CHECK: See if the current pose is already good enough ---
        ACCEPTANCE_THRESHOLD = 1.0
        current_angles_rad = self._get_current_joint_angles(articulated_chain)
        current_angles_dict = {name: angle for name, angle in zip(articulated_chain, current_angles_rad)}
        is_colliding_at_start = self._check_collision(current_angles_dict, base_transform_vtk, simplified_body_polydata)
        
        if not is_colliding_at_start:
            current_error = np.linalg.norm(self._ik_pose_error_function(current_angles_rad, articulated_chain, target_transform_vtk, base_transform_vtk))
            logging.info(f"Pre-check of current pose: Collision={is_colliding_at_start}, Error={current_error:.4f}")
            if current_error < ACCEPTANCE_THRESHOLD:
                slicer.util.infoDisplay("Current robot pose is already a valid solution. No optimization needed.")
                logging.info("Current robot pose is already a valid solution. Skipping optimization.")
                return 
        # --- END PRE-FLIGHT CHECK ---

        initial_guesses = [current_angles_rad, [0.0] * len(articulated_chain), [math.radians(a) for a in [0, 30, 30, 0, 0, 0]], [math.radians(a) for a in [0, -30, -30, 0, 0, 0]]]
        
        best_result, lowest_error = None, float('inf')
        best_colliding_result, lowest_colliding_error = None, float('inf')
        
        wasCancelled = False
        progressDialog = slicer.util.createProgressDialog(parent=slicer.util.mainWindow(), windowTitle="Planning Trajectory...", maximum=len(initial_guesses))
        
        try:
            for i, initial_guess in enumerate(initial_guesses):
                progressDialog.setValue(i)
                progressDialog.setLabelText(f"Trying initial guess #{i+1}...")
                slicer.app.processEvents()
                if progressDialog.wasCanceled:
                    wasCancelled = True
                    break
                
                try: 
                    result = scipy.optimize.least_squares(
                        self._ik_pose_and_collision_error_function, 
                        initial_guess, 
                        bounds=(bounds_lower, bounds_upper), 
                        args=(articulated_chain, target_transform_vtk, base_transform_vtk, simplified_body_polydata), 
                        method='trf', 
                        ftol=1e-3, xtol=1e-3, max_nfev=100, verbose=0)
                except Exception as e: 
                    logging.warning(f"IK optimization for guess #{i+1} failed: {e}"); continue

                final_angles_dict = {name: angle for name, angle in zip(articulated_chain, result.x)}
                is_colliding = self._check_collision(final_angles_dict, base_transform_vtk, simplified_body_polydata)
                final_error = np.linalg.norm(self._ik_pose_error_function(result.x, articulated_chain, target_transform_vtk, base_transform_vtk))
                logging.info(f"    > Guess #{i+1} finished. Pose Error: {final_error:.4f}. Collision: {is_colliding}")
                
                if not is_colliding and result.success:
                    if final_error < lowest_error:
                        lowest_error = final_error; best_result = result
                    if final_error < ACCEPTANCE_THRESHOLD:
                        logging.info(f"Found a high-quality solution with error {final_error:.4f}. Halting search early.")
                        break
                elif is_colliding and result.success:
                    if final_error < lowest_colliding_error:
                        lowest_colliding_error = final_error; best_colliding_result = result
        finally:
            progressDialog.setValue(len(initial_guesses))
            progressDialog = None

        if wasCancelled:
            logging.warning("Trajectory planning was cancelled by the user.")
            return

        final_result = best_result if best_result is not None else best_colliding_result
        if final_result is None: slicer.util.errorDisplay("IK optimization failed for all initial guesses."); return
        
        final_is_colliding = best_result is None
        final_error = lowest_error if not final_is_colliding else lowest_colliding_error
        
        logging.info(f"Applying best IK solution found (Pose Error: {final_error:.2f}, Collision: {final_is_colliding}):")
        if final_is_colliding:
            slicer.util.warningDisplay(f"Could not find a collision-free path. The best solution found still results in a collision (Error: {final_error:.2f}). Please adjust target/entry points.")
        
        for i, name in enumerate(articulated_chain):
            angle_deg = math.degrees(final_result.x[i]); logging.info(f"  - {name}: {angle_deg:.2f}°")
            if tf_node := self.jointTransformNodes.get(name): tf_node.SetMatrixTransformToParent(self._get_rotation_transform(angle_deg, self.robot_definition_dict[name].get("articulation_axis")).GetMatrix())
        
        if needle_tf := self.jointTransformNodes.get("Needle"): needle_tf.SetMatrixTransformToParent(vtk.vtkMatrix4x4())

        if not final_is_colliding:
            slicer.util.infoDisplay(f"Collision-free trajectory planned successfully. Final pose error: {final_error:.2f}.")
        
    def _load_collision_models(self):
        """
        Loads the simplified collision models from the paths defined in the robot structure.
        If a specific collision model doesn't exist, it falls back to using the visual model's geometry.
        """
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
                    logging.info(f"  - Loaded custom collision model for {jn}")
                except Exception as e:
                    logging.error(f"Failed to load collision STL '{collision_path}' for {jn}: {e}")
            
            if not polydata:
                if model_node := self.jointModelNodes.get(jn):
                    if model_node.GetPolyData():
                        polydata = model_node.GetPolyData()
                        logging.warning(f"  - No custom collision model for {jn}. Falling back to visual model polydata.")

            if polydata:
                self.jointCollisionPolys[jn] = polydata

    def _build_robot_model(self, baseplate_transform_matrix: vtk.vtkMatrix4x4):
        # Build the visual models first
        for joint_info in self.robot_definition:
            jn = joint_info["name"]
            stl_path = joint_info["stl_path"]
            if not stl_path or not os.path.exists(stl_path): logging.error(f"STL for {jn} not found at {stl_path}. Skipping."); continue
            model_matrix = baseplate_transform_matrix if jn == "Baseplate" else vtk.vtkMatrix4x4()
            art_tf_node = self._create_model_and_articulation_transform(jn, stl_path, model_matrix, joint_info.get("color"))
            if not art_tf_node: continue
            if parent_name := joint_info.get("parent"):
                if parent_art_node := self.jointTransformNodes.get(parent_name):
                    offset_matrix = joint_info.get("fixed_offset_to_parent", vtk.vtkMatrix4x4())
                    fixed_offset_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", f"{parent_name}To{jn}FixedOffset")
                    fixed_offset_node.SetMatrixTransformToParent(offset_matrix)
                    fixed_offset_node.SetAndObserveTransformNodeID(parent_art_node.GetID())
                    self.jointFixedOffsetTransformNodes[jn] = fixed_offset_node
                    art_tf_node.SetAndObserveTransformNodeID(fixed_offset_node.GetID())
                else: logging.error(f"Parent '{parent_name}' articulation node not found for '{jn}'.")
        
        self._load_collision_models()

    def _solve_all_ik_chains(self, identified_joints_data: Dict[str, List[Dict]]):
        for chain in self.ik_chains_config:
            distal_name = chain["distal_with_markers"]
            distal_def = self.robot_definition_dict.get(distal_name)
            fiducials_node = slicer.mrmlScene.GetFirstNodeByName(f"{distal_name}Fiducials")
            if (distal_name in identified_joints_data and distal_def and distal_def.get("has_markers") and fiducials_node and fiducials_node.GetNumberOfControlPoints() == 3):
                logging.info(f"Solving IK for chain: {chain['log_name']}.")
                self._solve_and_apply_generic_two_joint_ik(chain["parent_of_proximal"], chain["proximal"], distal_name, fiducials_node)
                self._visualize_joint_local_markers_in_world(distal_name)
            else: logging.warning(f"Skipping IK for {chain['log_name']} due to missing data or nodes.")

    def _get_rotation_transform(self, angle_deg: float, axis_str: Optional[str]) -> vtk.vtkTransform:
        transform = vtk.vtkTransform()
        if axis_str == "IS": transform.RotateZ(angle_deg)
        elif axis_str == "PA": transform.RotateY(angle_deg)
        elif axis_str == "LR": transform.RotateX(angle_deg)
        return transform

    def _generic_two_joint_ik_error_function(self, angles_rad, target_ras, local_coords, tf_parent_to_world, prox_def, dist_def) -> List[float]:
        prox_rad, dist_rad = angles_rad
        prox_offset = prox_def.get("fixed_offset_to_parent", vtk.vtkMatrix4x4()); prox_rot = self._get_rotation_transform(math.degrees(prox_rad), prox_def["articulation_axis"]).GetMatrix()
        dist_offset = dist_def.get("fixed_offset_to_parent", vtk.vtkMatrix4x4()); dist_rot = self._get_rotation_transform(math.degrees(dist_rad), dist_def["articulation_axis"]).GetMatrix()
        tf_prox_fixed = vtk.vtkMatrix4x4(); vtk.vtkMatrix4x4.Multiply4x4(tf_parent_to_world, prox_offset, tf_prox_fixed)
        tf_prox_art = vtk.vtkMatrix4x4(); vtk.vtkMatrix4x4.Multiply4x4(tf_prox_fixed, prox_rot, tf_prox_art)
        tf_dist_fixed = vtk.vtkMatrix4x4(); vtk.vtkMatrix4x4.Multiply4x4(tf_prox_art, dist_offset, tf_dist_fixed)
        tf_dist_model = vtk.vtkMatrix4x4(); vtk.vtkMatrix4x4.Multiply4x4(tf_dist_fixed, dist_rot, tf_dist_model)
        errors = []
        for i, local_p in enumerate(local_coords):
            pred_p_h = tf_dist_model.MultiplyPoint(list(local_p) + [1.0]); pred_ras = [c / pred_p_h[3] for c in pred_p_h[:3]]
            errors.extend([pred_ras[j] - target_ras[i][j] for j in range(3)])
        return errors

    def _solve_and_apply_generic_two_joint_ik(self, parent_name, prox_name, dist_name, target_node):
        target_mri = [target_node.GetNthControlPointPositionWorld(i) for i in range(3)]
        prox_def = self.robot_definition_dict.get(prox_name); dist_def = self.robot_definition_dict.get(dist_name)
        parent_node = self.jointTransformNodes.get(parent_name); dist_local = dist_def.get("local_marker_coords")
        if not all([prox_def, dist_def, parent_node, dist_local]): logging.error(f"IK setup failed for {prox_name}/{dist_name}. Data missing."); return
        tf_parent_to_world = vtk.vtkMatrix4x4(); parent_node.GetMatrixTransformToWorld(tf_parent_to_world)
        try: result = scipy.optimize.least_squares(self._generic_two_joint_ik_error_function, [0.0, 0.0], bounds=([-math.pi]*2, [math.pi]*2), args=(target_mri, dist_local, tf_parent_to_world, prox_def, dist_def))
        except Exception as e: logging.error(f"Scipy optimization for {prox_name}/{dist_name} failed: {e}"); return
        if not result.success: logging.warning(f"IK failed for {prox_name}/{dist_name}. Status: {result.status}, Cost: {result.cost:.4f}")
        prox_rad, dist_rad = result.x
        logging.info(f"IK Solution {prox_name}/{dist_name}: Prox={math.degrees(prox_rad):.2f}°, Dist={math.degrees(dist_rad):.2f}°, Cost={result.cost:.4f}")
        if prox_node := self.jointTransformNodes.get(prox_name): prox_node.SetMatrixTransformToParent(self._get_rotation_transform(math.degrees(prox_rad), prox_def["articulation_axis"]).GetMatrix())
        if dist_node := self.jointTransformNodes.get(dist_name): dist_node.SetMatrixTransformToParent(self._get_rotation_transform(math.degrees(dist_rad), dist_def["articulation_axis"]).GetMatrix())
        self._log_ik_results(result.x, target_mri, dist_local, tf_parent_to_world, prox_def, dist_def, dist_name)

    def _get_world_transform_for_joint(self, joint_angles_rad: Dict[str, float], target_joint_name: str, base_transform_matrix: vtk.vtkMatrix4x4) -> Optional[vtk.vtkMatrix4x4]:
        world_transforms = {}
        for joint_def in self.robot_definition:
            name = joint_def["name"]
            parent_name = joint_def.get("parent")
            parent_world_tf = base_transform_matrix if not parent_name else world_transforms.get(parent_name)
            if parent_world_tf is None and parent_name is not None: logging.error(f"FK failed: Parent '{parent_name}' of '{name}' not found."); return None
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

    def _full_chain_ik_error_function(self, angles_rad, articulated_joint_names, target_ras, base_transform, end_def):
        joint_values_rad = {name: angle for name, angle in zip(articulated_joint_names, angles_rad)}
        local_coords = end_def["local_marker_coords"]
        tf_end_model_to_world = self._get_world_transform_for_joint(joint_values_rad, end_def["name"], base_transform)
        if tf_end_model_to_world is None: return [1e6] * (len(local_coords) * 3)
        errors = []
        for i, local_p in enumerate(local_coords):
            pred_p_h = tf_end_model_to_world.MultiplyPoint(list(local_p) + [1.0]); pred_ras = [c / pred_p_h[3] for c in pred_p_h[:3]]
            errors.extend([pred_ras[j] - target_ras[i][j] for j in range(3)])
        return errors

    def _solve_full_chain_ik(self, end_effector_target_node: vtkMRMLMarkupsFiducialNode):
        articulated_chain = ["Shoulder1", "Link1", "Shoulder2", "Elbow1", "Wrist", "End"]
        chain_defs = [self.robot_definition_dict[name] for name in articulated_chain]
        end_def = self.robot_definition_dict["End"]
        target_mri = [end_effector_target_node.GetNthControlPointPositionWorld(i) for i in range(3)]
        base_node = self.jointTransformNodes.get("Baseplate")
        if not base_node: logging.error("Full-chain IK requires the Baseplate transform node in the scene."); return
        tf_base_to_world = vtk.vtkMatrix4x4(); base_node.GetMatrixTransformToWorld(tf_base_to_world)
        initial_guess = [0.0] * len(articulated_chain)
        bounds_rad = [tuple(math.radians(l) for l in jdef["joint_limits"]) for jdef in chain_defs]
        bounds_lower, bounds_upper = zip(*bounds_rad)
        try: result = scipy.optimize.least_squares(self._full_chain_ik_error_function, initial_guess, bounds=(bounds_lower, bounds_upper), args=(articulated_chain, target_mri, tf_base_to_world, end_def))
        except Exception as e: logging.error(f"Scipy optimization for full chain IK failed: {e}"); return
        if not result.success: logging.warning(f"Full-chain IK failed to converge. Status: {result.status}, Cost: {result.cost:.4f}")
        logging.info(f"Full-Chain IK Solution Cost: {result.cost:.4f}. Applying angles:")
        for i, name in enumerate(articulated_chain):
            angle_deg = math.degrees(result.x[i]); logging.info(f"  - {name}: {angle_deg:.2f}°")
            if tf_node := self.jointTransformNodes.get(name): tf_node.SetMatrixTransformToParent(self._get_rotation_transform(angle_deg, self.robot_definition_dict[name].get("articulation_axis")).GetMatrix())

    def saveBaseplateTransform(self, current_baseplate_transform_node: vtkMRMLLinearTransformNode):
        self._clear_node_by_name(self.SAVED_BASEPLATE_TRANSFORM_NODE_NAME)
        saved_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", self.SAVED_BASEPLATE_TRANSFORM_NODE_NAME)
        world_matrix = vtk.vtkMatrix4x4(); current_baseplate_transform_node.GetMatrixTransformToWorld(world_matrix)
        saved_node.SetMatrixTransformToParent(world_matrix); saved_node.SetSelectable(False)

    def _log_ik_results(self, angles, targets, locals, tf_parent, prox_def, dist_def, dist_name):
        final_errors = self._generic_two_joint_ik_error_function(angles, targets, locals, tf_parent, prox_def, dist_def)
        num_markers = len(locals); total_rmse_sq = 0
        logging.info(f"{dist_name} Marker Distances (Post-IK):")
        for i in range(num_markers):
            err = final_errors[i*3 : i*3+3]; dist_sq = sum(e**2 for e in err); total_rmse_sq += dist_sq
            logging.info(f"  Marker {i+1}: {math.sqrt(dist_sq):.2f} mm")
        logging.info(f"  Overall RMSE: {math.sqrt(total_rmse_sq / num_markers):.2f} mm")

    def _visualize_joint_local_markers_in_world(self, joint_name: str):
        debug_node_name = f"{joint_name}_LocalMarkers_WorldView_DEBUG"; self._clear_node_by_name(debug_node_name)
        joint_def = self.robot_definition_dict.get(joint_name); joint_art_node = self.jointTransformNodes.get(joint_name)
        local_coords = joint_def.get("local_marker_coords") if joint_def else None
        if not all([joint_def, joint_art_node, local_coords]): logging.warning(f"Cannot visualize {joint_name} local markers - data missing."); return
        tf_model_to_world = vtk.vtkMatrix4x4(); joint_art_node.GetMatrixTransformToWorld(tf_model_to_world)
        debug_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", debug_node_name)
        if disp := debug_node.GetDisplayNode():
            disp.SetVisibility(self.markers_visible)
            disp.SetGlyphScale(3.0); disp.SetTextScale(3.5); r,g,b = joint_def.get("color", (0.1,0.8,0.8))
            disp.SetSelectedColor(r*0.7, g*0.7, b*0.7); disp.SetColor(r,g,b); disp.SetOpacity(0)
        prefix = "".join(w[0] for w in joint_name.split() if w)[:3].upper()
        for i, local_p in enumerate(local_coords):
            world_p_h = tf_model_to_world.MultiplyPoint(list(local_p) + [1.0]); world_ras = [c/world_p_h[3] for c in world_p_h[:3]]
            idx = debug_node.AddControlPoint(world_ras); debug_node.SetNthControlPointLabel(idx, f"{prefix}_Lm{i+1}")
        logging.info(f"Created '{debug_node_name}' with {len(local_coords)} points.")

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
        logging.info("Visualizing all joint marker locations based on forward kinematics.")
        for joint_def in self.robot_definition:
            if joint_def.get("has_markers"): self._visualize_joint_local_markers_in_world(joint_def["name"])

    def _visualize_trajectory_line(self, target_pos, standoff_pos):
        """Creates or updates a markups line to show the planned trajectory."""
        line_node = slicer.mrmlScene.GetFirstNodeByName(self.TRAJECTORY_LINE_NODE_NAME)
        if not line_node:
            line_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", self.TRAJECTORY_LINE_NODE_NAME)
        
        if disp_node := line_node.GetDisplayNode():
            disp_node.SetGlyphScale(2.0)
            disp_node.SetTextScale(0.0)
            disp_node.SetSelectedColor(0.0, 1.0, 1.0)
            disp_node.SetColor(0.0, 1.0, 1.0)
            disp_node.SetLineThickness(0.5)

        line_node.RemoveAllControlPoints()
        line_node.AddControlPointWorld(vtk.vtkVector3d(standoff_pos))
        line_node.AddControlPointWorld(vtk.vtkVector3d(target_pos))

    def _calculate_fiducial_alignment_matrix(self, node_name: str, local_coords: List[Tuple[float,float,float]]) -> Optional[vtk.vtkMatrix4x4]:
        fiducials_node = slicer.mrmlScene.GetFirstNodeByName(node_name)
        if not (fiducials_node and fiducials_node.GetNumberOfControlPoints() >= 3 and len(local_coords) >= 3): logging.warning(f"Need >= 3 points for alignment of '{node_name}'."); return None
        n_pts = min(fiducials_node.GetNumberOfControlPoints(), len(local_coords), 3)
        target = vtk.vtkPoints(); source = vtk.vtkPoints()
        for i in range(n_pts):
            target.InsertNextPoint(fiducials_node.GetNthControlPointPositionWorld(i)); source.InsertNextPoint(local_coords[i])
        tf = vtk.vtkLandmarkTransform(); tf.SetSourceLandmarks(source); tf.SetTargetLandmarks(target)
        tf.SetModeToRigidBody(); tf.Update(); return tf.GetMatrix()

    def _create_model_and_articulation_transform(self, jn: str, stl: str, tf_mat: vtk.vtkMatrix4x4, color) -> Optional[vtkMRMLLinearTransformNode]:
        try:
            model = loadModel(stl)
            model.SetName(f"{jn}Model")
            self.jointModelNodes[jn] = model
        except Exception as e:
            logging.error(f"Failed to load STL '{stl}' for {jn}: {e}")
            return None
            
        tf_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", f"{jn}ArticulationTransform")
        tf_node.SetMatrixTransformToParent(tf_mat)
        self.jointTransformNodes[jn] = tf_node
        model.SetAndObserveTransformNodeID(tf_node.GetID())
        
        if disp := model.GetDisplayNode():
            disp.SetVisibility(self.models_visible)
            node_color = color or (0.7, 0.7, 0.7)
            disp.SetColor(node_color)
            disp.SetOpacity(0.85)
            disp.SetSelectedColor(node_color)
            
        return tf_node

    def _handle_joint_detection_results(self, identified_joints_data: Dict[str, List[Dict]]):
        if not identified_joints_data: logging.info("No joints identified."); return
        for jn, markers in identified_joints_data.items():
            config = self.robot_definition_dict.get(jn)
            if not (config and config.get("has_markers") and len(markers) == 3): logging.warning(f"Skipping node creation for '{jn}' (missing config or != 3 markers)."); continue
            if jn == "Baseplate":
                avg_y = sum(m["ras_coords"][1] for m in markers) / 3.0
                for m in markers: m["ras_coords"][1] = avg_y
                logging.info(f"Adjusted Baseplate fiducial Y-coords to {avg_y:.2f}")
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"{jn}Fiducials")
            if disp := node.GetDisplayNode():
                disp.SetVisibility(self.markers_visible)
                disp.SetGlyphScale(2.5); disp.SetTextScale(3.0); disp.SetColor(config.get("color", (0.8,0.8,0.2))); disp.SetSelectedColor(config.get("color", (0.8,0.8,0.2)))
            for i, m in enumerate(markers):
                idx = node.AddControlPoint(m["ras_coords"]); node.SetNthControlPointLabel(idx, f"{jn}_M{i+1}")
            logging.info(f"Created '{node.GetName()}' with 3 points.")

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
        if not (all_node and all_node.GetNumberOfControlPoints() >= 3): logging.warning("Need >= 3 detected fiducials for joint detection. ⚠️"); return {}
        all_fiducials = [ {"id": i, "ras_coords": list(all_node.GetNthControlPointPositionWorld(i))} for i in range(all_node.GetNumberOfControlPoints())]
        identified, used_ids = {}, set()
        for jc in self.robot_definition:
            if not jc.get("has_markers"): continue
            jn = jc["name"]; arm_lengths = jc.get("arm_lengths") 
            if not arm_lengths or len(arm_lengths) != 2: logging.warning(f"'{jn}' missing 'arm_lengths'. Skipping."); continue
            l1, l2 = arm_lengths[0], arm_lengths[1]; expected_dists = sorted([l1, l2, math.hypot(l1, l2)])
            available = [f for f in all_fiducials if f["id"] not in used_ids]
            if len(available) < 3: continue
            for combo in itertools.combinations(available, 3):
                pts = [c["ras_coords"] for c in combo]; dists = sorted([math.dist(pts[0], pts[1]), math.dist(pts[0], pts[2]), math.dist(pts[1], pts[2])])
                if all(abs(d - e) <= self.DISTANCE_TOLERANCE for d, e in zip(dists, expected_dists)):
                    matched_data = [dict(c) for c in combo]
                    sorted_data = self._sort_l_shaped_markers(matched_data, l1, l2, self.DISTANCE_TOLERANCE)
                    identified[jn] = sorted_data if sorted_data else matched_data
                    used_ids.update(c["id"] for c in combo); logging.info(f"Identified '{jn}' (IDs: {[c['id'] for c in combo]}). Sorted: {bool(sorted_data)}"); break 
            else: logging.info(f"No suitable combo found for '{jn}'.")
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
            if disp := node.GetDisplayNode(): disp.SetGlyphScale(1.0); disp.SetColor(0.9,0.9,0.1); disp.SetSelectedColor(1.0,0.5,0.0); disp.SetVisibility(False)
            for fd in fiducials_data:
                lps = fd["centroid"]; idx = node.AddControlPoint([-lps[0], -lps[1], lps[2]]); node.SetNthControlPointLabel(idx, f"M_{fd['id']}_{fd['vol']:.0f}mm³")
            logging.info(f"Created 'DetectedFiducials' with {len(fiducials_data)} points. 🔬")
        all_labels = stats.GetLabels()
        if not all_labels: logging.warning("No objects found in image after thresholding."); return
        non_fiducial_labels = [lbl for lbl in all_labels if lbl not in {f['id'] for f in fiducials_data}]
        if not non_fiducial_labels: logging.warning("No large components to segment as body."); return
        largest_label_id = max(non_fiducial_labels, key=stats.GetPhysicalSize)
        logging.info(f"Largest object (label {largest_label_id}, volume {stats.GetPhysicalSize(largest_label_id):.2f} mm³) will be segmented as body.")
        largest_object_img = sitk.Cast(sitk.BinaryThreshold(labeled, largest_label_id, largest_label_id, 1, 0), sitk.sitkUInt8)
        self._clear_node_by_name("TempBodyLabelMap")
        tempLabelmapNode = sitkUtils.PushVolumeToSlicer(largest_object_img, name="TempBodyLabelMap", className="vtkMRMLLabelMapVolumeNode")
        if not tempLabelmapNode: logging.error("sitkUtils.PushVolumeToSlicer failed to create a temporary labelmap."); return
        self._clear_node_by_name("AutoBodySegmentation"); segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "AutoBodySegmentation")
        if not slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(tempLabelmapNode, segmentationNode):
            logging.error("Failed to import labelmap into segmentation node."); slicer.mrmlScene.RemoveNode(segmentationNode); return
        slicer.mrmlScene.RemoveNode(tempLabelmapNode)
        
        if segmentationNode.GetSegmentation().GetNumberOfSegments() > 0:
            segment = segmentationNode.GetSegmentation().GetNthSegment(0)
            segment.SetName("Body")
            segment.SetColor([0.8, 0.2, 0.2])
            
            segmentation = segmentationNode.GetSegmentation()
            closedSurfaceRepresentationName = slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName()
            logging.info(f"Generating '{closedSurfaceRepresentationName}' representation for 'Body' segment...")
            segmentation.CreateRepresentation(closedSurfaceRepresentationName)
            logging.info("...representation generation complete. ✅")

        segmentationNode.CreateDefaultDisplayNodes()
        if dispNode := segmentationNode.GetDisplayNode(): dispNode.SetOpacity(0.75); dispNode.SetVisibility3D(True)
        pNode.segmentationNode = segmentationNode; logging.info(f"Successfully created and selected 'AutoBodySegmentation'.")

    def _cleanup_module_nodes(self):
        logging.info("Cleaning up module nodes...")
        self.jointCollisionPolys.clear()

        if getattr(self, 'DEBUG_COLLISIONS', False):
            logging.info("Cleaning up collision debug nodes...")
            all_parts = [j["name"] for j in self.robot_definition] + ["Body"]
            for part_name in all_parts:
                self._clear_node_by_name(f"DEBUG_COLLISION_{part_name}")
        
        all_node_names_to_clear = {
            "DetectedFiducials", 
            self.TARGET_POSE_TRANSFORM_NODE_NAME,
            self.TRAJECTORY_LINE_NODE_NAME
        }
        for jc in self.robot_definition:
            jn = jc["name"]; all_node_names_to_clear.add(f"{jn}Model"); all_node_names_to_clear.add(f"{jn}ArticulationTransform")
            if jc.get("parent"): all_node_names_to_clear.add(f"{jc['parent']}To{jn}FixedOffset")
            if jc.get("has_markers"): all_node_names_to_clear.add(f"{jn}Fiducials"); all_node_names_to_clear.add(f"{jn}_LocalMarkers_WorldView_DEBUG")
        for name in all_node_names_to_clear: self._clear_node_by_name(name)
        self.jointModelNodes.clear(); self.jointTransformNodes.clear(); self.jointFixedOffsetTransformNodes.clear()
        logging.info("Cleanup complete.")

    def _toggle_robot_markers(self, checked: bool):
        marker_names = set()
        for jc in self.robot_definition:
            if jc.get("has_markers"):
                marker_names.add(f"{jc['name']}Fiducials")
        
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