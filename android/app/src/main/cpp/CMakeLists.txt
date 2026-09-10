cmake_minimum_required(VERSION 3.22.1)

project("navigation_core")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CORE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../native/navigation_core")

include_directories(
    ${CORE_SRC_DIR}
)

add_library(
    navigation_core
    SHARED
    native-lib.cpp
    ${CORE_SRC_DIR}/strapdown_ins.cpp
    ${CORE_SRC_DIR}/eskf_core.cpp
    ${CORE_SRC_DIR}/outage_detector.cpp
    ${CORE_SRC_DIR}/nav_engine.cpp
    ${CORE_SRC_DIR}/nav_core_c_api.cpp
)

find_library(
    log-lib
    log
)

target_link_libraries(
    navigation_core
    ${log-lib}
)
