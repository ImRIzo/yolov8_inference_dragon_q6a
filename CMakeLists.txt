cmake_minimum_required(VERSION 3.14)
project(yolov8_qnn_video CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -Wno-unused-result")

set(CMAKE_BUILD_RPATH "$ORIGIN/lib:")

set(OpenCV_DIR /usr/lib/aarch64-linux-gnu/cmake/opencv4)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn highgui videoio)
set(QNN_INC /home/radxa/qairt/include)

add_executable(yolov8_video
    main.cpp preprocess.cpp inference.cpp postprocess.cpp evaluate.cpp)
target_include_directories(yolov8_video PRIVATE
    ${QNN_INC} ${QNN_INC}/QNN ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolov8_video PRIVATE ${OpenCV_LIBS} dl pthread)

# Release target
add_custom_target(release
    COMMAND bash "${CMAKE_SOURCE_DIR}/mkrelease.sh" "${CMAKE_BINARY_DIR}" "${CMAKE_SOURCE_DIR}"
    COMMENT "Assembling build/release/"
    DEPENDS yolov8_video
)

message(STATUS "  Build:   cmake --build . -j$$(nproc)")
message(STATUS "  Release: cmake --build . --target release")
