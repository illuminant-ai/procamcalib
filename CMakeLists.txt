cmake_minimum_required(VERSION 3.18.0)

option(WITH_SPINNAKER "Use the Spinnaker SDK for increased camera support" OFF)
option(WITH_PHOTONEO "Use the Photoneo API for increased camera support" OFF)
option(WITH_ZIVID "Use the Zivid API for increased camera support" OFF)

set(OpenGL_GL_PREFERENCE "GLVND")

set(PROJECT_NAME "scan3d-capture")

project(${PROJECT_NAME} LANGUAGES C CXX)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOUIC_SEARCH_PATHS forms)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(OpenCV_DIR "C:/dev/opencv-4.7.0/install")
find_package(OpenCV COMPONENTS core imgproc highgui calib3d features2d flann REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Qt5 COMPONENTS OpenGL REQUIRED)

# the options need to be the same as Open3D's default
# if Open3D is configured and built with custom options, you'll also need to
# specify the same custom options.
option(STATIC_WINDOWS_RUNTIME "Use static (MT/MTd) Windows runtime" OFF)
if(STATIC_WINDOWS_RUNTIME)
    message(STATUS "Static (MT/MTd) Windows Runtime detected")
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    message(STATUS "Static (MT/MTd) Windows Runtime NOT detected")
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

add_executable(${PROJECT_NAME}
    src/AboutDialog.cpp
    src/Application.cpp
    src/CalibrationData.cpp
    src/CalibrationDialog.cpp
    src/CaptureDialog.cpp
#    src/cognex_util.cpp
    src/GLWidget.cpp
    src/ImageLabel.cpp
    src/im_util.cpp
    src/io_util.cpp
    src/main.cpp
    src/MainWindow.cpp
    src/ProcessingDialog.cpp
    src/ProjectorWidget.cpp
    src/scan3d.cpp
    src/structured_light.cpp
    src/TreeModel.cpp
    src/VideoInput.cpp
    forms/AboutDialog.ui
    forms/CalibrationDialog.ui
    forms/CaptureDialog.ui
    forms/MainWindow.ui
    forms/ProcessingDialog.ui
    assets/assets.qrc
)

target_include_directories(${PROJECT_NAME} PRIVATE src forms ${OPENCV_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS} ${Qt5OpenGL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${OPENGL_LIBRARIES} Qt5::OpenGL)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

if(WITH_SPINNAKER)
    find_package(Spinnaker REQUIRED)
    
    target_include_directories(${PROJECT_NAME} PRIVATE ${SPINNAKER_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${SPINNAKER_LIBRARIES})
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_SPINNAKER)

    target_sources(${PROJECT_NAME} PRIVATE
        src/CameraConfigurationDialog.cpp
        forms/CameraConfigurationDialog.ui
    )

    MESSAGE(STATUS "Building with the Spinnaker SDK")
endif()

# Photoneo Support
if(WITH_PHOTONEO)
	if (NOT PHO_API_CMAKE_CONFIG_PATH)
		set(PHO_API_CMAKE_CONFIG_PATH "$ENV{PHOXI_CONTROL_PATH}")
	endif()
	string(REPLACE "\\" "/" PHO_API_CMAKE_CONFIG_PATH_ESCAPED ${PHO_API_CMAKE_CONFIG_PATH})
	find_package(PhoXi REQUIRED CONFIG PATHS "${PHO_API_CMAKE_CONFIG_PATH_ESCAPED}")

    target_include_directories(${PROJECT_NAME} PRIVATE ${PHOXI_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${PHOXI_LIBRARY})
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_PHOTONEO)

    MESSAGE(STATUS "Building with the Photoneo API")
endif()

# Zivid Support
if(WITH_ZIVID)
    # use zivid package
    set(ZIVID_VERSION 2.9.0)
    find_package(
        Zivid ${ZIVID_VERSION}
        COMPONENTS Core
        REQUIRED)
    include_directories(${PROJECT_NAME} "C:/Program Files/Zivid/include/")
    target_link_libraries(${PROJECT_NAME} Zivid::Core)
    target_compile_definitions(${PROJECT_NAME} PRIVATE _USE_MATH_DEFINES USE_ZIVID)
    MESSAGE(STATUS "Building with the Zivid API")

endif()

if(WIN32)
    find_package(DirectShow COMPONENTS Strmiids Vfw32 Ole32 OleAut32 REQUIRED)

    target_include_directories(${PROJECT_NAME} PRIVATE ${DirectShow_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} ${DIRECTSHOW_LIBRARIES})
    target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS _SCL_SECURE_NO_WARNINGS _USE_MATH_DEFINES)
    #target_compile_options(${PROJECT_NAME} PRIVATE /MT)
    #target_link_options(${PROJECT_NAME} PRIVATE /INCREMENTAL:NO /NODEFAULTLIB:MSVCRT)
endif(WIN32)

if(WITH_ZIVID)
    target_compile_options(${PROJECT_NAME} PRIVATE /MD)
endif()

install(TARGETS ${PROJECT_NAME} DESTINATION bin)

