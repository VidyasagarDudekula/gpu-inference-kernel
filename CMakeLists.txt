cmake_minimum_required(VERSION 3.30.4)
project(MYPROJ LANGUAGES CXX CUDA)

include(FetchContent)

FetchContent_Declare(
    nvbench
    GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
    GIT_TAG main
)

FetchContent_MakeAvailable(nvbench)

add_executable(myproj main.cu)

target_link_libraries(myproj PRIVATE nvbench)
