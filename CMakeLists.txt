project( ACA )
cmake_minimum_required(VERSION 2.8)

find_package( OpenCV REQUIRED )

add_subdirectory(src)

SET( CMAKE_CXX_FLAGS_DEBUG "-g -Wall" )
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)
