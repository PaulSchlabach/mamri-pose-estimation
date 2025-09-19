cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(FiducialDetector)

#-----------------------------------------------------------------------------
# Extension meta-information
slicer_extension_add_meta_data(
  NAME "FiducialDetector"
  CATEGORY "Robotics"  # You can change this to a more appropriate category
  DESCRIPTION "This extension intends to provide tools for automatic fiducial detection for MR-safe biopsy robots of the University of Twente."
  CONTRIBUTORS "Paul Schlabach (University of Twente)"
  HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/FiducialDetector"
  ICON_FILENAME "${CMAKE_CURRENT_SOURCE_DIR}/Resources/Icons/FiducialDetector.png" # Path to an icon within your repo
  SCREENSHOT_FILENAMES "" # Paths to screenshots within your repo
  DEPENDS "" # List of other required extensions, e.g., "Markups;SegmentEditor"
  )
#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(Mamri)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
