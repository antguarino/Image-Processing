rit_add_executable(fft_display 
  SOURCES
    fft_display.cpp
)

target_link_libraries(fft_display
  rit::ipcv_utils
  Boost::filesystem
  Boost::program_options
  opencv_core
  opencv_highgui
  opencv_imgcodecs
)
