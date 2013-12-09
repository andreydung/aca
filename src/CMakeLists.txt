add_executable(${CMAKE_PROJECT_NAME} aca.cpp main.cpp aca.h)
target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})
