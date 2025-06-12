import logging
import os
import itertools
import math
from typing import Annotated, Optional, Dict, List, Tuple

import vtk
import SimpleITK as sitk
import sitkUtils
import scipy.optimize

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
    intensityThreshold: Annotated[float, WithinRange(0.0, 5000.0)] = 65.0
    minVolumeThreshold: Annotated[float, WithinRange(0.0, 10000.0)] = 150.0
    maxVolumeThreshold: Annotated[float, WithinRange(0.0, 10000.0)] = 1500.0
    distance_tolerance: Annotated[float, WithinRange(0.0, 10.0)] = 3.0

    segmentationNode: vtkMRMLSegmentationNode
    targetFiducialNode: vtkMRMLMarkupsFiducialNode
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
            logging.info("MamriWidget.setup: 'drawFiducialsCheckBox' connected.")
        else:
            logging.warning("MamriWidget.setup: 'drawFiducialsCheckBox' not found in UI. Cannot connect toggle signal.")
        if hasattr(self.ui, "drawModelsCheckBox"):
            self.ui.drawModelsCheckBox.connect("toggled(bool)", self.onDrawModelsCheckBoxToggled)
            logging.info("MamriWidget.setup: 'drawModelsCheckBox' connected.")
        else:
            logging.warning("MamriWidget.setup: 'drawModelsCheckBox' not found in UI. Cannot connect toggle signal.")
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = MamriLogic()
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.initializeParameterNode()

    def cleanup(self) -> None: self.removeObservers()
    def enter(self) -> None: self.initializeParameterNode()
    def exit(self) -> None:
        self.remove_parameter_node_observers()

    def onSceneStartClose(self, caller, event) -> None: self.setParameterNode(None)
    def onSceneEndClose(self, caller, event) -> None:
        if self.parent.isEntered: self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        self.logic = self.logic or MamriLogic()
        self.setParameterNode(self.logic.getParameterNode())
        if self._parameterNode and not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode: self._parameterNode.inputVolume = firstVolumeNode

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
            # New observer connection
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanPlanTrajectory)
        self._checkCanApply()
        # New check function call
        self._checkCanPlanTrajectory()

    def _checkCanApply(self, caller=None, event=None) -> None:
        can_apply = self._parameterNode and self._parameterNode.inputVolume is not None
        if hasattr(self.ui, "applyButton") and self.ui.applyButton:
            self.ui.applyButton.enabled = can_apply
            if can_apply:
                tooltip = _("Run fiducial detection and robot model rendering with IK.") 
            else:
                tooltip = _("Select an input volume node.")
            self.ui.applyButton.toolTip = tooltip

    def _checkCanPlanTrajectory(self, caller=None, event=None) -> None:
        if not hasattr(self.ui, "planTrajectoryButton"):
            return
        
        can_plan = (self._parameterNode and 
                    self._parameterNode.segmentationNode and 
                    self._parameterNode.targetFiducialNode)
        
        self.ui.planTrajectoryButton.enabled = bool(can_plan)
        if can_plan:
            tooltip = _("Calculate the robot joint angles to target the fiducial.")
        else:
            tooltip = _("Select a patient segmentation and a target fiducial to enable planning.")
        self.ui.planTrajectoryButton.toolTip = tooltip

    def onApplyButton(self) -> None:
        if not self._parameterNode:
            slicer.util.errorDisplay("Parameter node is not initialized.")
            return
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            self.logic.process(self._parameterNode)
    
    def onPlanTrajectoryButton(self) -> None:
        if not self._parameterNode:
            slicer.util.errorDisplay("Parameter node is not initialized.")
            return
        # with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
        #     self.logic.planTrajectory(self._parameterNode)
        

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

        self.robot_definition = self._define_robot_structure()
        self.robot_definition_dict = {joint["name"]: joint for joint in self.robot_definition}

        self.ik_chains_config = [
            {"parent_of_proximal": "Baseplate", "proximal": "Shoulder1", "distal_with_markers": "Link1", "log_name": "Shoulder1/Link1"},
            {"parent_of_proximal": "Link1", "proximal": "Shoulder2", "distal_with_markers": "Elbow1", "log_name": "Shoulder2/Elbow1"},
            {"parent_of_proximal": "Link2", "proximal": "Wrist", "distal_with_markers": "End", "log_name": "Wrist/End"},
        ]

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
        """Defines the kinematic structure and properties of the robot."""
        base_path = r"C:\Users\paul\Documents\UTwente\MSc ROB\MSc Thesis\CAD\Joints"
        
        return [
            {
                "name": "Baseplate", "stl_path": os.path.join(base_path, "Baseplate.STL"),
                "parent": None, "fixed_offset_to_parent": None, "has_markers": True,
                "local_marker_coords": [(-10.0, 20.0, 5.0), (10.0, 20.0, 5.0), (-10.0, -20.0, 5.0)],
                "arm_lengths": (40.0, 20.0),
                "color": (1, 0, 0),
                "articulation_axis": None
            },
            {
                "name": "Shoulder1", "stl_path": os.path.join(base_path, "Shoulder1.STL"),
                "parent": "Baseplate", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 20.0)),
                "has_markers": False, "color": (0, 0.5, 0), "articulation_axis": "IS"
            },
            {
                "name": "Link1", "stl_path": os.path.join(base_path, "Link1.STL"),
                "parent": "Shoulder1", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 30)),
                "has_markers": True, "local_marker_coords": [(12.5, 45.0, 110.0), (-12.5, 45.0, 110.0), (12.5, 45.0, 40.0)],
                "arm_lengths": (70.0, 25.0),
                "color": (0, 1, 0),
                "articulation_axis": "PA"
            },
            {
                "name": "Shoulder2", "stl_path": os.path.join(base_path, "Shoulder2.STL"),
                "parent": "Link1", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 150)),
                "has_markers": False, "color": (0, 0.5, 0), "articulation_axis": "PA"
            },
            {
                "name": "Elbow1", "stl_path": os.path.join(base_path, "Elbow1.STL"),
                "parent": "Shoulder2", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, -35)),
                "has_markers": True, "local_marker_coords": [(10, 35.0, 120.0), (-10, 35.0, 120.0), (-10, -35.0, 120.0)],
                "arm_lengths": (70.0, 20.0), 
                "color": (0, 1, 0),
                "articulation_axis": "IS"
            },
            {
                "name": "Link2", "stl_path": os.path.join(base_path, "Link2.STL"),
                "parent": "Elbow1", "fixed_offset_to_parent": self._create_offset_matrix((0, 35, 125)),
                "has_markers": False, "color": (0, 1, 0), "articulation_axis": None
            },
            {
                "name": "Link3", "stl_path": os.path.join(base_path, "Link3.STL"),
                "parent": "Elbow1", "fixed_offset_to_parent": self._create_offset_matrix((0, -45, 125)),
                "has_markers": False, "color": (0, 1, 0), "articulation_axis": None
            },
            {
                "name": "Wrist", "stl_path": os.path.join(base_path, "Wrist.STL"),
                "parent": "Link2", "fixed_offset_to_parent": self._create_offset_matrix((0, 20, 60)),
                "has_markers": False, "color": (0, 0.5, 0), "articulation_axis": "PA"
            },
            {
                "name": "End", "stl_path": os.path.join(base_path, "End.STL"),
                "parent": "Wrist", "fixed_offset_to_parent": self._create_offset_matrix((0, -55, 0)),
                "has_markers": True, "local_marker_coords": [(-10, 22.5, 30), (10, 22.5, 30), (-10, -22.5, 30)],
                "arm_lengths": (45.0, 20.0),
                "color": (1, 0, 0),
                "articulation_axis": "IS"
            },
            {
                "name": "EndEffectorHolder", "stl_path": os.path.join(base_path, "EndEffectorHolder.STL"),
                "parent": "End", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 35)),
                "has_markers": False, "color": (0, 1, 0), "articulation_axis": None
            },
            {
                "name": "NeedleHolder", "stl_path": os.path.join(base_path, "NeedleHolder.STL"),
                "parent": "EndEffectorHolder", "fixed_offset_to_parent": self._create_offset_matrix((0, 0, 26.05)),
                "has_markers": False, "color": (1, 0, 0), "articulation_axis": "TRANS_X"
            }
        ]

    def getParameterNode(self) -> MamriParameterNode:
        return MamriParameterNode(super().getParameterNode())

    def _clear_node_by_name(self, name: str):
        """Removes all nodes with the given name from the scene."""
        while node := slicer.mrmlScene.GetFirstNodeByName(name):
            slicer.mrmlScene.RemoveNode(node)

    def process(self, parameterNode: MamriParameterNode) -> None:
        logging.info("Starting Mamri processing...")
        self._cleanup_module_nodes()

        self.volume_threshold_segmentation(parameterNode)
        identified_joints_data = self.joint_detection(parameterNode)
        self._handle_joint_detection_results(identified_joints_data)

        logging.info("Rendering robot model and performing IK...")
        self._build_robot_model(identified_joints_data)
        self._solve_all_ik_chains(identified_joints_data)

        logging.info("Mamri processing finished.")

    def _build_robot_model(self, identified_joints_data: Dict[str, List[Dict]]):
        """Builds the robot model hierarchy in Slicer."""
        for joint_info in self.robot_definition:
            jn = joint_info["name"]
            stl_path = joint_info["stl_path"]

            if not stl_path or not os.path.exists(stl_path):
                logging.error(f"STL for {jn} not found at {stl_path}. Skipping.")
                continue

            model_matrix = vtk.vtkMatrix4x4()
            model_matrix.Identity()

            if jn == "Baseplate" and joint_info.get("has_markers") and jn in identified_joints_data:
                if alignment := self._calculate_fiducial_alignment_matrix(f"{jn}Fiducials", joint_info["local_marker_coords"]):
                    model_matrix.DeepCopy(alignment)
                else:
                    logging.warning(f"Baseplate alignment failed. Using identity for {jn}.")

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
                else:
                    logging.error(f"Parent '{parent_name}' node not found for '{jn}'.")

    def _solve_all_ik_chains(self, identified_joints_data: Dict[str, List[Dict]]):
        """Iterates through IK chains and solves them."""
        for chain in self.ik_chains_config:
            distal_name = chain["distal_with_markers"]
            distal_def = self.robot_definition_dict.get(distal_name)
            fiducials_node = slicer.mrmlScene.GetFirstNodeByName(f"{distal_name}Fiducials")

            if (distal_name in identified_joints_data and distal_def and distal_def.get("has_markers")
                and fiducials_node and fiducials_node.GetNumberOfControlPoints() == 3):
                logging.info(f"Solving IK for chain: {chain['log_name']}.")
                self._solve_and_apply_generic_two_joint_ik(
                    chain["parent_of_proximal"], chain["proximal"], distal_name, fiducials_node
                )
                self._visualize_joint_local_markers_in_world(distal_name)
            else:
                logging.warning(f"Skipping IK for {chain['log_name']} due to missing data or nodes.")

    def _get_rotation_transform(self, angle_deg: float, axis_str: Optional[str]) -> vtk.vtkTransform:
        """Creates a VTK transform for a given rotation."""
        transform = vtk.vtkTransform()
        if axis_str == "IS": transform.RotateZ(angle_deg)
        elif axis_str == "PA": transform.RotateY(angle_deg)
        elif axis_str == "LR": transform.RotateX(angle_deg)
        return transform

    def _generic_two_joint_ik_error_function(self, angles_rad, target_ras, local_coords, tf_parent_to_world, prox_def, dist_def) -> List[float]:
        """Calculates the error for the IK solver."""
        prox_rad, dist_rad = angles_rad
        
        prox_offset = prox_def.get("fixed_offset_to_parent", vtk.vtkMatrix4x4())
        prox_rot = self._get_rotation_transform(math.degrees(prox_rad), prox_def["articulation_axis"]).GetMatrix()
        
        dist_offset = dist_def.get("fixed_offset_to_parent", vtk.vtkMatrix4x4())
        dist_rot = self._get_rotation_transform(math.degrees(dist_rad), dist_def["articulation_axis"]).GetMatrix()

        tf_prox_fixed = vtk.vtkMatrix4x4(); vtk.vtkMatrix4x4.Multiply4x4(tf_parent_to_world, prox_offset, tf_prox_fixed)
        tf_prox_art = vtk.vtkMatrix4x4(); vtk.vtkMatrix4x4.Multiply4x4(tf_prox_fixed, prox_rot, tf_prox_art)
        tf_dist_fixed = vtk.vtkMatrix4x4(); vtk.vtkMatrix4x4.Multiply4x4(tf_prox_art, dist_offset, tf_dist_fixed)
        tf_dist_model = vtk.vtkMatrix4x4(); vtk.vtkMatrix4x4.Multiply4x4(tf_dist_fixed, dist_rot, tf_dist_model)

        errors = []
        for i, local_p in enumerate(local_coords):
            pred_p_h = tf_dist_model.MultiplyPoint(list(local_p) + [1.0])
            pred_ras = [c / pred_p_h[3] for c in pred_p_h[:3]]
            errors.extend([pred_ras[j] - target_ras[i][j] for j in range(3)])
        return errors

    def _solve_and_apply_generic_two_joint_ik(self, parent_name, prox_name, dist_name, target_node):
        """Solves a 2-joint IK problem and applies the result."""
        ras_p = [0.0, 0.0, 0.0]
        target_mri = []
        for i in range(target_node.GetNumberOfControlPoints()):
            success = target_node.GetNthControlPointPositionWorld(i, ras_p)
            if success:
                target_mri.append(list(ras_p))
            else:
                logging.warning(f"IK: Failed to get world coords for point {i} in {target_node.GetName()}.")

        prox_def = self.robot_definition_dict.get(prox_name)
        dist_def = self.robot_definition_dict.get(dist_name)
        parent_node = self.jointTransformNodes.get(parent_name)
        dist_local = dist_def.get("local_marker_coords")

        if not all([prox_def, dist_def, parent_node, dist_local, len(target_mri) == 3]):
            logging.error(f"IK setup failed for {prox_name}/{dist_name}. Need 3 target points, found {len(target_mri)}. Or other data missing.")
            return

        tf_parent_to_world = vtk.vtkMatrix4x4(); parent_node.GetMatrixTransformToWorld(tf_parent_to_world)
        
        try:
            result = scipy.optimize.least_squares(
                self._generic_two_joint_ik_error_function, [0.0, 0.0],
                bounds=([-math.pi] * 2, [math.pi] * 2),
                args=(target_mri, dist_local, tf_parent_to_world, prox_def, dist_def)
            )
        except Exception as e:
            logging.error(f"Scipy optimization for {prox_name}/{dist_name} failed: {e}")
            return

        if not result.success:
            logging.warning(f"IK failed for {prox_name}/{dist_name}. Status: {result.status}, Cost: {result.cost:.4f}")

        prox_rad, dist_rad = result.x
        logging.info(f"IK Solution {prox_name}/{dist_name}: Prox={math.degrees(prox_rad):.2f}°, Dist={math.degrees(dist_rad):.2f}°, Cost={result.cost:.4f}")

        if prox_node := self.jointTransformNodes.get(prox_name):
            prox_node.SetMatrixTransformToParent(self._get_rotation_transform(math.degrees(prox_rad), prox_def["articulation_axis"]).GetMatrix())
        if dist_node := self.jointTransformNodes.get(dist_name):
            dist_node.SetMatrixTransformToParent(self._get_rotation_transform(math.degrees(dist_rad), dist_def["articulation_axis"]).GetMatrix())

        self._log_ik_results(result.x, target_mri, dist_local, tf_parent_to_world, prox_def, dist_def, dist_name)

    def _log_ik_results(self, angles, targets, locals, tf_parent, prox_def, dist_def, dist_name):
        """Logs the final distances and RMSE after IK."""
        final_errors = self._generic_two_joint_ik_error_function(angles, targets, locals, tf_parent, prox_def, dist_def)
        num_markers = len(locals)
        total_rmse_sq = 0
        logging.info(f"{dist_name} Marker Distances (Post-IK):")
        for i in range(num_markers):
            err = final_errors[i*3 : i*3+3]
            dist_sq = sum(e**2 for e in err)
            total_rmse_sq += dist_sq
            logging.info(f"  Marker {i+1}: {math.sqrt(dist_sq):.2f} mm")
        logging.info(f"  Overall RMSE: {math.sqrt(total_rmse_sq / num_markers):.2f} mm")


    def _visualize_joint_local_markers_in_world(self, joint_name: str):
        """Creates debug markers showing local coordinates in world space."""
        debug_node_name = f"{joint_name}_LocalMarkers_WorldView_DEBUG"
        self._clear_node_by_name(debug_node_name)

        joint_def = self.robot_definition_dict.get(joint_name)
        joint_art_node = self.jointTransformNodes.get(joint_name)
        local_coords = joint_def.get("local_marker_coords") if joint_def else None

        if not all([joint_def, joint_art_node, local_coords]):
            logging.warning(f"Cannot visualize {joint_name} local markers - data missing.")
            return

        tf_model_to_world = vtk.vtkMatrix4x4(); joint_art_node.GetMatrixTransformToWorld(tf_model_to_world)
        debug_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", debug_node_name)
        disp = debug_node.GetDisplayNode()
        if disp:
            disp.SetGlyphScale(3.0); disp.SetTextScale(3.5)
            r, g, b = joint_def.get("color", (0.1, 0.8, 0.8))
            disp.SetSelectedColor(r * 0.7, g * 0.7, b * 0.7); disp.SetColor(r, g, b)
            disp.SetOpacity(0)

        prefix = "".join(w[0] for w in joint_name.split() if w)[:3].upper()
        for i, local_p in enumerate(local_coords):
            world_p_h = tf_model_to_world.MultiplyPoint(list(local_p) + [1.0])
            world_ras = [c / world_p_h[3] for c in world_p_h[:3]]
            idx = debug_node.AddControlPoint(world_ras)
            debug_node.SetNthControlPointLabel(idx, f"{prefix}_Lm{i+1}")
        logging.info(f"Created '{debug_node_name}' with {len(local_coords)} points.")


    def _calculate_fiducial_alignment_matrix(self, node_name: str, local_coords: List[Tuple[float,float,float]]) -> Optional[vtk.vtkMatrix4x4]:
        """Calculates rigid transform from local coords to detected fiducials."""
        fiducials_node = slicer.mrmlScene.GetFirstNodeByName(node_name)
        if not (fiducials_node and fiducials_node.GetNumberOfControlPoints() >= 3 and len(local_coords) >= 3):
            logging.warning(f"Need >= 3 points for alignment of '{node_name}'.")
            return None

        n_pts = min(fiducials_node.GetNumberOfControlPoints(), len(local_coords), 3)
        target = vtk.vtkPoints(); source = vtk.vtkPoints()
        ras_p = [0.0, 0.0, 0.0]
        for i in range(n_pts):
            fiducials_node.GetNthControlPointPositionWorld(i, ras_p)
            target.InsertNextPoint(ras_p)
            source.InsertNextPoint(local_coords[i])

        tf = vtk.vtkLandmarkTransform(); tf.SetSourceLandmarks(source); tf.SetTargetLandmarks(target)
        tf.SetModeToRigidBody(); tf.Update()
        return tf.GetMatrix()

    def _create_model_and_articulation_transform(self, jn: str, stl: str, tf_mat: vtk.vtkMatrix4x4, color) -> Optional[vtkMRMLLinearTransformNode]:
        """Loads STL, creates model and transform nodes, and sets hierarchy/display."""
        try:
            model = loadModel(stl); model.SetName(f"{jn}Model")
            self.jointModelNodes[jn] = model
        except Exception as e:
            logging.error(f"Failed to load STL '{stl}' for {jn}: {e}"); return None

        tf_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", f"{jn}ArticulationTransform")
        tf_node.SetMatrixTransformToParent(tf_mat)
        self.jointTransformNodes[jn] = tf_node
        model.SetAndObserveTransformNodeID(tf_node.GetID())

        if disp := model.GetDisplayNode():
            disp.SetVisibility(True); disp.SetColor(color or (0.7,0.7,0.7)); disp.SetOpacity(0.85)
        return tf_node

    def _handle_joint_detection_results(self, identified_joints_data: Dict[str, List[Dict]]):
        """Creates fiducial nodes for identified joints."""
        if not identified_joints_data:
            logging.info("No joints identified."); return

        for jn, markers in identified_joints_data.items():
            config = self.robot_definition_dict.get(jn)
            if not (config and config.get("has_markers") and len(markers) == 3):
                logging.warning(f"Skipping node creation for '{jn}' (missing config or != 3 markers).")
                continue
            
            # Special Baseplate Y-averaging
            if jn == "Baseplate":
                avg_y = sum(m["ras_coords"][1] for m in markers) / 3.0
                for m in markers: m["ras_coords"][1] = avg_y
                logging.info(f"Adjusted Baseplate fiducial Y-coords to {avg_y:.2f}")

            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"{jn}Fiducials")
            if disp := node.GetDisplayNode():
                disp.SetGlyphScale(2.5); disp.SetTextScale(3.0)
                disp.SetColor(config.get("color", (0.8, 0.8, 0.2))); disp.SetSelectedColor(config.get("color", (0.8, 0.8, 0.2)))

            for i, m in enumerate(markers):
                idx = node.AddControlPoint(m["ras_coords"])
                node.SetNthControlPointLabel(idx, f"{jn}_M{i+1}")
            logging.info(f"Created '{node.GetName()}' with 3 points.")

    def _sort_l_shaped_markers(self, markers: List[Dict], len1: float, len2: float, tol: float) -> Optional[List[Dict]]:
        """Sorts 3 markers based on L-shape arm lengths."""
        if len(markers) != 3: return None
        points = [{'data': m, 'ras': tuple(m["ras_coords"])} for m in markers]
        l_short, l_long = sorted((len1, len2))

        for i in range(3):
            corner, p1, p2 = points[i], points[(i + 1) % 3], points[(i + 2) % 3]
            d1, d2 = math.dist(corner['ras'], p1['ras']), math.dist(corner['ras'], p2['ras'])
            
            if abs(d1 - l_short) <= tol and abs(d2 - l_long) <= tol: return [corner['data'], p1['data'], p2['data']]
            if abs(d1 - l_long) <= tol and abs(d2 - l_short) <= tol: return [corner['data'], p2['data'], p1['data']]
        return None

    def joint_detection(self, pNode: 'MamriParameterNode') -> Dict[str, List[Dict]]:
        """Identifies joint marker sets from all detected fiducials."""
        all_node = slicer.mrmlScene.GetFirstNodeByName("DetectedFiducials")
        if not (all_node and all_node.GetNumberOfControlPoints() >= 3):
            logging.warning("Need >= 3 detected fiducials for joint detection. ⚠️"); return {}

        ras_p = [0.0, 0.0, 0.0]
        all_fiducials = []
        for i in range(all_node.GetNumberOfControlPoints()):
            success = all_node.GetNthControlPointPositionWorld(i, ras_p)
            if success:
                all_fiducials.append({
                    "id": i, 
                    "ras_coords": list(ras_p) 
                })
            else:
                logging.warning(f"Failed to get world coordinates for point {i} in DetectedFiducials.")
        
        identified, used_ids = {}, set()

        for jc in self.robot_definition:
            if not jc.get("has_markers"): continue
            jn = jc["name"]
            
            arm_lengths = jc.get("arm_lengths") 
            if not arm_lengths or len(arm_lengths) != 2: 
                logging.warning(f"'{jn}' missing 'arm_lengths' in its definition or 'arm_lengths' does not contain exactly two values. Skipping.")
                continue
            
            l1, l2 = arm_lengths[0], arm_lengths[1]
            expected_dists = sorted([l1, l2, math.hypot(l1, l2)])
            available = [f for f in all_fiducials if f["id"] not in used_ids]
            if len(available) < 3: continue

            for combo in itertools.combinations(available, 3):
                pts = [c["ras_coords"] for c in combo]
                dists = sorted([math.dist(pts[0], pts[1]), math.dist(pts[0], pts[2]), math.dist(pts[1], pts[2])])

                if all(abs(d - e) <= pNode.distance_tolerance for d, e in zip(dists, expected_dists)):
                    matched_data = [dict(c) for c in combo] # Deep copy
                    sorted_data = self._sort_l_shaped_markers(matched_data, l1, l2, pNode.distance_tolerance)
                    
                    identified[jn] = sorted_data if sorted_data else matched_data
                    used_ids.update(c["id"] for c in combo)
                    logging.info(f"Identified '{jn}' (IDs: {[c['id'] for c in combo]}). Sorted: {bool(sorted_data)}")
                    break # Move to next joint config
            else:
                logging.info(f"No suitable combo found for '{jn}'.")
        return identified

    def volume_threshold_segmentation(self, pNode: 'MamriParameterNode') -> None:
        """Segments potential markers based on intensity and volume."""
        try:
            sitk_img = sitkUtils.PullVolumeFromSlicer(pNode.inputVolume)
        except Exception as e: logging.error(f"Failed to pull volume: {e}"); return

        binary = sitk.BinaryThreshold(sitk_img, pNode.intensityThreshold, 65535)
        closed = sitk.BinaryMorphologicalClosing(binary, [2] * 3, sitk.sitkBall)
        labeled = sitk.ConnectedComponent(closed)
        stats = sitk.LabelShapeStatisticsImageFilter(); stats.Execute(labeled)
        
        fiducials_data = [
            {"vol": stats.GetPhysicalSize(lbl), "centroid": stats.GetCentroid(lbl), "id": lbl}
            for lbl in stats.GetLabels() if pNode.minVolumeThreshold <= stats.GetPhysicalSize(lbl) <= pNode.maxVolumeThreshold
        ]

        self._clear_node_by_name("DetectedFiducials")
        if not fiducials_data: logging.info("No components met volume criteria."); return

        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "DetectedFiducials")
        if disp := node.GetDisplayNode():
            disp.SetGlyphScale(1.0); disp.SetColor(0.9, 0.9, 0.1); disp.SetSelectedColor(1.0, 0.5, 0.0); disp.SetVisibility(0)

        for fd in fiducials_data:
            lps = fd["centroid"]
            idx = node.AddControlPoint([-lps[0], -lps[1], lps[2]]) # LPS to RAS
            node.SetNthControlPointLabel(idx, f"M_{fd['id']}_{fd['vol']:.0f}mm³")
        logging.info(f"Created 'DetectedFiducials' with {len(fiducials_data)} points. 🔬")


    def _cleanup_module_nodes(self):
        """Removes all nodes created by this module."""
        logging.info("Cleaning up module nodes...")
        self._clear_node_by_name("DetectedFiducials")
        
        all_node_names_to_clear = set()
        for jc in self.robot_definition:
            jn = jc["name"]
            all_node_names_to_clear.add(f"{jn}Model")
            all_node_names_to_clear.add(f"{jn}ArticulationTransform")
            if jc.get("parent"): all_node_names_to_clear.add(f"{jc['parent']}To{jn}FixedOffset")
            if jc.get("has_markers"): all_node_names_to_clear.add(f"{jn}Fiducials")
        
        for chain in self.ik_chains_config:
            all_node_names_to_clear.add(f"{chain['distal_with_markers']}_LocalMarkers_WorldView_DEBUG")

        for name in all_node_names_to_clear:
            self._clear_node_by_name(name)

        self.jointModelNodes.clear(); self.jointTransformNodes.clear(); self.jointFixedOffsetTransformNodes.clear()
        logging.info("Cleanup complete.")

    def _toggle_robot_markers(self, checked: bool):
        
        marker_names = set()

        for jc in self.robot_definition:
            if jc.get("has_markers"):
                marker_names.add(f"{jc['name']}Fiducials")
        
        for name in marker_names:
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node and isinstance(node , slicer.vtkMRMLMarkupsFiducialNode):
                disp = node.GetDisplayNode()
                if disp: 
                    disp.SetVisibility(checked)
    
    def _toggle_robot_models(self, checked: bool):
        
        model_names = set()

        for jc in self.robot_definition:
            model_names.add(f"{jc['name']}Model")
        
        for name in model_names:
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node and isinstance(node , slicer.vtkMRMLModelNode):
                disp = node.GetDisplayNode()
                if disp: 
                    disp.SetVisibility(checked)


#
# MamriTest
#
class MamriTest(ScriptedLoadableModuleTest):
    def setUp(self): slicer.mrmlScene.Clear()
    def runTest(self): self.setUp()