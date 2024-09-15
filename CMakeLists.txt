cmake_minimum_required(VERSION 3.18)
project(nerfacc_cpp)

find_package(Torch REQUIRED)

# nerfacc
include_directories(submodules/nerfacc)
file(GLOB_RECURSE NERFACC_SOURCES submodules/nerfacc/nerfacc/*.cpp
     submodules/nerfacc/nerfacc/*.cu)
list(REMOVE_ITEM NERFACC_SOURCES
     "${PROJECT_SOURCE_DIR}/submodules/nerfacc/nerfacc/cuda/csrc/nerfacc.cpp")

add_library(nerfacc ${NERFACC_SOURCES})
target_link_libraries(nerfacc ${catkin_LIBRARIES} ${TORCH_LIBRARIES})
