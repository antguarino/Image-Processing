rit_add_executable(transform_polar
  SOURCES
    transform_polar.cpp
)

target_link_libraries(transform_polar
  rit::ipcv_geometric_transformation 
  Boost::filesystem
  Boost::program_options
  opencv_core
  opencv_highgui
  opencv_imgcodecs
)
