rit_add_executable(gradient 
  SOURCES
    gradient.cpp
)

target_link_libraries(gradient 
  rit::ipcv_sobel
  Boost::filesystem
  Boost::program_options
  opencv_core
  opencv_highgui
  opencv_imgcodecs
)
