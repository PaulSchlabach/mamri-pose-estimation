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

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/Mamri.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        
        if hasattr(self.ui, "applyButton"):
            self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        if hasattr(self.ui, "planTrajectoryButton"):
            self.ui.planTrajectoryButton.connect("clicked(bool)", self.onPlanTrajectoryButton)
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
            
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = MamriLogic()
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.initializeParameterNode()

    def cleanup(self) -> None: self.removeObservers()
    
    def enter(self) -> None: 
        self.initializeParameterNode()
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
        
        if hasattr(self.ui, "planTrajectoryButton"):
            can_plan_base = (self._parameterNode and
                             self._parameterNode.targetFiducialNode and self._parameterNode.targetFiducialNode.GetNumberOfControlPoints() > 0 and
                             self._parameterNode.entryPointFiducialNode and self._parameterNode.entryPointFiducialNode.GetNumberOfControlPoints() > 0 and
                             self._parameterNode.segmentationNode)
            self.ui.planTrajectoryButton.enabled = can_plan_base and model_is_built
        
        if hasattr(self.ui, "zeroRobotButton"):
            self.ui.zeroRobotButton.enabled = model_is_built
            self.ui.zeroRobotButton.toolTip = _("Sets all robot joint angles to zero.") if model_is_built else _("Run 'Start robot pose estimation' first to build the model.")

    def onApplyButton(self) -> None:
        if not self._parameterNode:
            slicer.util.errorDisplay("Parameter node is not initialized.")
            return
        
        models_visible = self.ui.drawModelsCheckBox.isChecked() if hasattr(self.ui, "drawModelsCheckBox") else True
        markers_visible = self.ui.drawFiducialsCheckBox.isChecked() if hasattr(self.ui, "drawFiducialsCheckBox") else True
        
        self.logic.process(self._parameterNode, models_visible=models_visible, markers_visible=markers_visible)
        self._checkAllButtons()
    
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

    def onZeroRobotButton(self) -> None:
        if self.logic:
            self.logic.zeroRobot()

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

        self.MASTER_FOLDER_NAME = "MAMRI Robot Output"

        self.DEBUG_COLLISIONS = False 
        if self.DEBUG_COLLISIONS:
            logging.warning("MamriLogic collision debugging is enabled. This will create temporary models in the scene.")

        self.ik_chains_config = [
            {"parent_of_proximal": "Baseplate", "proximal": "Shoulder1", "distal_with_markers": "Link1", "log_name": "Shoulder1/Link1"},
            {"parent_of_proximal": "Link1", "proximal": "Shoulder2", "distal_with_markers": "Elbow1", "log_name": "Shoulder2/Elbow1"},
            {"parent_of_proximal": "Elbow1", "proximal": "Wrist", "distal_with_markers": "End", "log_name": "Wrist/End"},
        ]
        
        self._discover_robot_nodes_in_scene()

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
            {"name": "Shoulder1", "stl_path": os.path.join(base_path, "Shoulder1.STL"), "collision_stl_path": os.path.join(base_path, "Shoulder1_collision.STL"), "parent": "Baseplate", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 20.0)), "has_markers": False, "color": (0, 0.5, 0), "articulation_axis": "IS", "joint_limits": (-180, 180)},
            {"name": "Link1", "stl_path": os.path.join(base_path, "Link1.STL"), "collision_stl_path": os.path.join(base_path, "Link1_collision.STL"), "parent": "Shoulder1", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 30)), "has_markers": True, "local_marker_coords": [(12.5, 45.0, 110.0), (-12.5, 45.0, 110.0), (12.5, 45.0, 40.0)], "arm_lengths": (70.0, 25.0), "color": (0, 1, 0), "articulation_axis": "PA", "joint_limits": (-120, 120)},
            {"name": "Shoulder2", "stl_path": os.path.join(base_path, "Shoulder2.STL"), "collision_stl_path": os.path.join(base_path, "Shoulder2_collision.STL"), "parent": "Link1", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 150)), "has_markers": False, "color": (0, 0.5, 0), "articulation_axis": "PA", "joint_limits": (-120, 120)},
            {"name": "Elbow1", "stl_path": os.path.join(base_path, "Elbow1.STL"), "collision_stl_path": os.path.join(base_path, "Elbow1_collision.STL"), "parent": "Shoulder2", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 0)), "has_markers": True, "local_marker_coords": [(-10, 35.0, 85), (10, 35.0, 85), (-10, -35.0, 85)], "arm_lengths": (70.0, 20.0),  "color": (0, 1, 0), "articulation_axis": "IS", "joint_limits": (-180, 180)},
            {"name": "Wrist", "stl_path": os.path.join(base_path, "Wrist.STL"), "collision_stl_path": os.path.join(base_path, "Wrist_collision.STL"), "parent": "Elbow1", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 150)), "has_markers": False, "color": (0, 0.5, 0), "articulation_axis": "PA", "joint_limits": (-120, 120)},
            {"name": "End", "stl_path": os.path.join(base_path, "End.STL"), "collision_stl_path": os.path.join(base_path, "End_collision.STL"), "parent": "Wrist", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 8)), "has_markers": True, "local_marker_coords": [(-10, 22.5, 26), (10, 22.5, 26), (-10, -22.5, 26)], "arm_lengths": (45.0, 20.0), "color": (1, 0, 0), "articulation_axis": "IS", "joint_limits": (-270, 270)},
            {"name": "Needle", "stl_path": os.path.join(base_path, "Needle.STL"), "collision_stl_path": os.path.join(base_path, "Needle_collision.STL"), "parent": "End", "fixed_offset_to_parent": self._create_offset_matrix((-50, 0, 71)), "has_markers": False, "color": (1, 0, 0), "articulation_axis": "TRANS_X", "joint_limits": (0, 0), "needle_tip_local": (0, 0, 0), "needle_axis_local": (1, 0, 0)}
        ]

    def getParameterNode(self) -> MamriParameterNode:
        return MamriParameterNode(super().getParameterNode())

    def _clear_node_by_name(self, name: str):
        """A simple utility to remove all nodes with a given name from the scene."""
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
            
            # Pass the correction flag to the IK solvers
            apply_correction = parameterNode.applyEndEffectorCorrection

            if baseplate_transform_found and "End" in identified_joints_data:
                logging.info("Attempting full-chain IK from baseplate to detected 'End' joint.")
                end_fiducials_node = slicer.mrmlScene.GetFirstNodeByName("EndFiducials")
                if end_fiducials_node and end_fiducials_node.GetNumberOfControlPoints() == 3:
                    self._solve_full_chain_ik(end_fiducials_node, apply_correction)
                    self._visualize_all_joint_markers_from_fk()
                else:
                    logging.warning("Could not perform full-chain IK. 'End' fiducials not found or incomplete. Falling back to sequential IK.")
                    self._solve_all_ik_chains(identified_joints_data, apply_correction)
            else:
                logging.info("Prerequisites for full-chain IK not met. Using standard sequential IK for any detected intermediate joints.")
                self._solve_all_ik_chains(identified_joints_data, apply_correction)
            logging.info("Mamri processing finished.")

    def _get_body_polydata(self, segmentationNode: vtkMRMLSegmentationNode) -> Optional[vtk.vtkPolyData]:
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

    def planTrajectory(self, pNode: MamriParameterNode) -> None:
        targetNode, entryNode, segmentationNode = pNode.targetFiducialNode, pNode.entryPointFiducialNode, pNode.segmentationNode
        if not (targetNode and targetNode.GetNumberOfControlPoints() > 0 and entryNode and entryNode.GetNumberOfControlPoints() > 0 and segmentationNode):
            slicer.util.errorDisplay("Set target, entry, and segmentation to plan trajectory."); return

        body_polydata = self._get_body_polydata(segmentationNode)
        if not body_polydata:
            slicer.util.errorDisplay("Could not get body polydata from segmentation. Aborting."); return

        decimator = vtk.vtkDecimatePro()
        decimator.SetInputData(body_polydata)
        decimator.SetTargetReduction(0.95)
        decimator.PreserveTopologyOn()
        decimator.Update()
        simplified_body_polydata = decimator.GetOutput()

        target_pos, entry_pos = np.array(targetNode.GetNthControlPointPositionWorld(0)), np.array(entryNode.GetNthControlPointPositionWorld(0))
        direction_vec = target_pos - entry_pos
        if np.linalg.norm(direction_vec) < 1e-6: slicer.util.errorDisplay("Entry and Target markers are at the same position."); return
        
        x_axis = direction_vec / np.linalg.norm(direction_vec)
        needle_tip_pos = entry_pos - (pNode.safetyDistance * x_axis)
        line_node = self._visualize_trajectory_line(target_pos, needle_tip_pos)
        if line_node:
            self._organize_node_in_subject_hierarchy(line_node, self.MASTER_FOLDER_NAME, "Trajectory Plan")
        
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
        if not base_node: slicer.util.errorDisplay("Robot model not loaded or baseplate is missing."); return
        base_transform_vtk = vtk.vtkMatrix4x4(); base_node.GetMatrixTransformToWorld(base_transform_vtk)

        initial_guesses = [self._get_current_joint_angles(articulated_chain), [0.0] * len(articulated_chain)]
        
        best_result, lowest_error = None, float('inf')
        
        for initial_guess in initial_guesses:
            try: 
                result = scipy.optimize.least_squares(
                    self._ik_pose_and_collision_error_function, initial_guess, bounds=(bounds_lower, bounds_upper), 
                    args=(articulated_chain, target_transform_vtk, base_transform_vtk, simplified_body_polydata), 
                    method='trf', ftol=1e-3, xtol=1e-3, max_nfev=200)
            except Exception as e: 
                logging.warning(f"IK optimization failed for one guess: {e}"); continue

            final_angles_dict = {name: angle for name, angle in zip(articulated_chain, result.x)}
            is_colliding = self._check_collision(final_angles_dict, base_transform_vtk, simplified_body_polydata)
            final_error = np.linalg.norm(self._ik_pose_error_function(result.x, articulated_chain, target_transform_vtk, base_transform_vtk))
            
            if not is_colliding and result.success and final_error < lowest_error:
                lowest_error = final_error; best_result = result
        
        if not best_result: slicer.util.errorDisplay("Could not find a valid, collision-free trajectory solution."); return

        for i, name in enumerate(articulated_chain):
            angle_deg = math.degrees(best_result.x[i])
            if tf_node := self.jointTransformNodes.get(name): tf_node.SetMatrixTransformToParent(self._get_rotation_transform(angle_deg, self.robot_definition_dict[name].get("articulation_axis")).GetMatrix())
        
        if needle_tf := self.jointTransformNodes.get("Needle"): needle_tf.SetMatrixTransformToParent(vtk.vtkMatrix4x4())

        slicer.util.infoDisplay(f"Collision-free trajectory planned. Final pose error: {lowest_error:.2f}.")
        
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

    def _solve_all_ik_chains(self, identified_joints_data: Dict[str, List[Dict]], apply_correction: bool):
        for chain in self.ik_chains_config:
            distal_name = chain["distal_with_markers"]
            distal_def = self.robot_definition_dict.get(distal_name)
            fiducials_node = slicer.mrmlScene.GetFirstNodeByName(f"{distal_name}Fiducials")
            if (distal_name in identified_joints_data and distal_def and distal_def.get("has_markers") and fiducials_node and fiducials_node.GetNumberOfControlPoints() == 3):
                self._solve_and_apply_generic_two_joint_ik(chain["parent_of_proximal"], chain["proximal"], distal_name, fiducials_node, apply_correction)
                self._visualize_joint_local_markers_in_world(distal_name)

    def _get_rotation_transform(self, angle_deg: float, axis_str: Optional[str]) -> vtk.vtkTransform:
        transform = vtk.vtkTransform()
        if axis_str == "IS": transform.RotateZ(angle_deg)
        elif axis_str == "PA": transform.RotateY(angle_deg)
        elif axis_str == "LR": transform.RotateX(angle_deg)
        return transform

    def _generic_two_joint_ik_error_function(self, angles_rad, target_ras, local_coords, tf_parent_to_world, prox_def, dist_def, apply_correction: bool) -> List[float]:
        prox_rad, dist_rad = angles_rad
        
        corrected_local_coords = list(local_coords)
        if dist_def["name"] == "End" and apply_correction:
            rotation_transform = vtk.vtkTransform()
            rotation_transform.RotateZ(180)
            corrected_local_coords = [rotation_transform.TransformPoint(p) for p in local_coords]
        
        prox_offset = prox_def.get("fixed_offset_to_parent", vtk.vtkMatrix4x4()); prox_rot = self._get_rotation_transform(math.degrees(prox_rad), prox_def["articulation_axis"]).GetMatrix()
        dist_offset = dist_def.get("fixed_offset_to_parent", vtk.vtkMatrix4x4()); dist_rot = self._get_rotation_transform(math.degrees(dist_rad), dist_def["articulation_axis"]).GetMatrix()
        tf_prox_fixed = vtk.vtkMatrix4x4(); vtk.vtkMatrix4x4.Multiply4x4(tf_parent_to_world, prox_offset, tf_prox_fixed)
        tf_prox_art = vtk.vtkMatrix4x4(); vtk.vtkMatrix4x4.Multiply4x4(tf_prox_fixed, prox_rot, tf_prox_art)
        tf_dist_fixed = vtk.vtkMatrix4x4(); vtk.vtkMatrix4x4.Multiply4x4(tf_prox_art, dist_offset, tf_dist_fixed)
        tf_dist_model = vtk.vtkMatrix4x4(); vtk.vtkMatrix4x4.Multiply4x4(tf_dist_fixed, dist_rot, tf_dist_model)
        errors = []
        for i, local_p in enumerate(corrected_local_coords):
            pred_p_h = tf_dist_model.MultiplyPoint(list(local_p) + [1.0]); pred_ras = [c / pred_p_h[3] for c in pred_p_h[:3]]
            errors.extend([pred_ras[j] - target_ras[i][j] for j in range(3)])
        return errors

    def _solve_and_apply_generic_two_joint_ik(self, parent_name, prox_name, dist_name, target_node, apply_correction: bool):
        target_mri = [target_node.GetNthControlPointPositionWorld(i) for i in range(3)]
        prox_def = self.robot_definition_dict.get(prox_name); dist_def = self.robot_definition_dict.get(dist_name)
        parent_node = self.jointTransformNodes.get(parent_name); dist_local = dist_def.get("local_marker_coords")
        if not all([prox_def, dist_def, parent_node, dist_local]): return
        tf_parent_to_world = vtk.vtkMatrix4x4(); parent_node.GetMatrixTransformToWorld(tf_parent_to_world)
        try: result = scipy.optimize.least_squares(self._generic_two_joint_ik_error_function, [0.0, 0.0], bounds=([-math.pi]*2, [math.pi]*2), args=(target_mri, dist_local, tf_parent_to_world, prox_def, dist_def, apply_correction))
        except Exception as e: logging.error(f"Scipy optimization for {prox_name}/{dist_name} failed: {e}"); return
        if not result.success: logging.warning(f"IK failed for {prox_name}/{dist_name}.")
        prox_rad, dist_rad = result.x
        if prox_node := self.jointTransformNodes.get(prox_name): prox_node.SetMatrixTransformToParent(self._get_rotation_transform(math.degrees(prox_rad), prox_def["articulation_axis"]).GetMatrix())
        if dist_node := self.jointTransformNodes.get(dist_name): dist_node.SetMatrixTransformToParent(self._get_rotation_transform(math.degrees(dist_rad), dist_def["articulation_axis"]).GetMatrix())
        self._log_ik_results(result.x, target_mri, dist_local, tf_parent_to_world, prox_def, dist_def, dist_name, apply_correction)

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
        
        end_local_coords = end_def["local_marker_coords"]
        
        # Apply correction if the flag is set
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

    def _solve_full_chain_ik(self, end_effector_target_node: vtkMRMLMarkupsFiducialNode, apply_correction: bool):
        articulated_chain = ["Shoulder1", "Link1", "Shoulder2", "Elbow1", "Wrist", "End"]
        chain_defs = [self.robot_definition_dict[name] for name in articulated_chain]
        end_def = self.robot_definition_dict["End"]
        end_target_mri = [end_effector_target_node.GetNthControlPointPositionWorld(i) for i in range(3)]
        
        base_node = self.jointTransformNodes.get("Baseplate")
        if not base_node:
            logging.error("Full-chain IK requires the Baseplate transform node."); return
        tf_base_to_world = vtk.vtkMatrix4x4()
        base_node.GetMatrixTransformToWorld(tf_base_to_world)
        
        elbow_target_mri = None
        elbow_def = None
        elbow_fiducials_node = slicer.mrmlScene.GetFirstNodeByName("Elbow1Fiducials")
        if elbow_fiducials_node and elbow_fiducials_node.GetNumberOfControlPoints() == 3:
            logging.info("Elbow markers detected. Adding as a secondary objective to the IK solver.")
            elbow_def = self.robot_definition_dict["Elbow1"]
            elbow_target_mri = [elbow_fiducials_node.GetNthControlPointPositionWorld(i) for i in range(3)]
        
        initial_guesses = [self._get_current_joint_angles(articulated_chain), [0.0] * len(articulated_chain)]
        
        best_result = None
        lowest_cost = float('inf')

        for i, initial_guess in enumerate(initial_guesses):
            logging.info(f"--- Running IK Optimization (Attempt #{i+1}) ---")
            try:
                result = scipy.optimize.least_squares(
                    self._full_chain_ik_error_function, initial_guess, bounds=([math.radians(j["joint_limits"][0]) for j in chain_defs], [math.radians(j["joint_limits"][1]) for j in chain_defs]),
                    args=(articulated_chain, end_target_mri, tf_base_to_world, end_def, apply_correction, elbow_target_mri, elbow_def),
                    method='trf', ftol=1e-6, xtol=1e-6, verbose=0)
                
                if result.success and result.cost < lowest_cost:
                    lowest_cost = result.cost
                    best_result = result
            except Exception as e:
                logging.error(f"Scipy optimization for attempt #{i+1} failed: {e}")
        
        if not best_result:
            logging.error("Full-chain IK failed to converge for all initial guesses.")
            return

        final_angles_rad = best_result.x
        final_angles_deg = [math.degrees(a) for a in final_angles_rad]
        
        self._log_ik_solution_details(final_angles_rad, articulated_chain, tf_base_to_world, end_def, end_target_mri, apply_correction, elbow_def, elbow_target_mri)
        
        for i, name in enumerate(articulated_chain):
            if tf_node := self.jointTransformNodes.get(name):
                tf_node.SetMatrixTransformToParent(self._get_rotation_transform(final_angles_deg[i], self.robot_definition_dict[name].get("articulation_axis")).GetMatrix())
    
    def saveBaseplateTransform(self, current_baseplate_transform_node: vtkMRMLLinearTransformNode):
        """Saves the given transform node to a permanent node in the scene."""
        self._clear_node_by_name(self.SAVED_BASEPLATE_TRANSFORM_NODE_NAME)
        saved_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", self.SAVED_BASEPLATE_TRANSFORM_NODE_NAME)
        world_matrix = vtk.vtkMatrix4x4()
        current_baseplate_transform_node.GetMatrixTransformToWorld(world_matrix)
        saved_node.SetMatrixTransformToParent(world_matrix)
        saved_node.SetSelectable(False)
        self._organize_node_in_subject_hierarchy(saved_node, self.MASTER_FOLDER_NAME, "Saved Transforms")
        
    def _log_ik_solution_details(self, final_angles_rad, articulated_chain, base_transform, end_def, end_target_ras, apply_correction: bool, elbow_def=None, elbow_target_ras=None):
        logging.info("-------------------- IK Solution Details --------------------")
        final_angles_deg = [math.degrees(a) for a in final_angles_rad]
        joint_values_rad = {name: angle for name, angle in zip(articulated_chain, final_angles_rad)}

        logging.info("Final Joint Angles (°):")
        for name, angle_deg in zip(articulated_chain, final_angles_deg):
            logging.info(f"  - {name}: {angle_deg:.2f}°")

        end_local_coords = end_def["local_marker_coords"]
        if apply_correction:
            rotation_transform = vtk.vtkTransform()
            rotation_transform.RotateZ(180)
            end_local_coords = [rotation_transform.TransformPoint(p) for p in end_local_coords]

        tf_end_world = self._get_world_transform_for_joint(joint_values_rad, end_def["name"], base_transform)
        if tf_end_world:
            logging.info("\nEnd-Effector Marker Errors (mm):")
            end_squared_errors = []
            for i, local_p in enumerate(end_local_coords):
                pred_p_h = tf_end_world.MultiplyPoint(list(local_p) + [1.0])
                pred_ras = np.array([c / pred_p_h[3] for c in pred_p_h[:3]])
                target_ras_np = np.array(end_target_ras[i])
                dist_error = np.linalg.norm(pred_ras - target_ras_np)
                end_squared_errors.append(dist_error**2)
                logging.info(f"  - Marker {i+1}: Distance Error = {dist_error:.3f} mm")
            
            if end_squared_errors:
                end_rmse = math.sqrt(sum(end_squared_errors) / len(end_squared_errors))
                logging.info(f"  -> End-Effector RMSE: {end_rmse:.3f} mm")

        if elbow_def and elbow_target_ras:
            tf_elbow_world = self._get_world_transform_for_joint(joint_values_rad, elbow_def["name"], base_transform)
            if tf_elbow_world:
                logging.info("\nElbow Marker Errors (mm):")
                elbow_squared_errors = []
                for i, local_p in enumerate(elbow_def["local_marker_coords"]):
                    pred_p_h = tf_elbow_world.MultiplyPoint(list(local_p) + [1.0])
                    pred_ras = np.array([c / pred_p_h[3] for c in pred_p_h[:3]])
                    target_ras_np = np.array(elbow_target_ras[i])
                    dist_error = np.linalg.norm(pred_ras - target_ras_np)
                    elbow_squared_errors.append(dist_error**2)
                    logging.info(f"  - Marker {i+1}: Distance Error = {dist_error:.3f} mm")
                
                if elbow_squared_errors:
                    elbow_rmse = math.sqrt(sum(elbow_squared_errors) / len(elbow_squared_errors))
                    logging.info(f"  -> Elbow RMSE: {elbow_rmse:.3f} mm")
        logging.info("-----------------------------------------------------------")

    def _log_ik_results(self, angles, targets, locals, tf_parent, prox_def, dist_def, dist_name, apply_correction: bool):
        final_errors = self._generic_two_joint_ik_error_function(angles, targets, locals, tf_parent, prox_def, dist_def, apply_correction)
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
        if not all([joint_def, joint_art_node, local_coords]): return
        tf_model_to_world = vtk.vtkMatrix4x4(); joint_art_node.GetMatrixTransformToWorld(tf_model_to_world)
        
        debug_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", debug_node_name)
        self._organize_node_in_subject_hierarchy(debug_node, self.MASTER_FOLDER_NAME, "Debug Markers")
        
        if disp := debug_node.GetDisplayNode():
            disp.SetVisibility(self.markers_visible)
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
        logging.info("Visualizing all joint marker locations based on forward kinematics.")
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
            logging.error(f"Failed to load STL '{stl}' for {jn}: {e}")
            return None, None
            
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

        folders_to_clear = [
            "Robot Model", "Detected MRI Markers", "Debug Markers",
            "Segmentations", "Trajectory Plan"
        ]
        
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

    def _toggle_robot_markers(self, checked: bool):
        marker_names = set()
        for jc in self.robot_definition:
            if jc.get("has_markers"):
                marker_names.add(f"{jc['name']}Fiducials")
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
