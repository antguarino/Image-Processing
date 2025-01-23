rit_add_executable(visual_cryptography 
  SOURCES
  visual_cryptography.cpp
)

target_link_libraries(visual_cryptography
  rit::ipcv_utils
  Boost::filesystem
  Boost::program_options
  opencv_core
  opencv_highgui
  opencv_imgcodecs
)
