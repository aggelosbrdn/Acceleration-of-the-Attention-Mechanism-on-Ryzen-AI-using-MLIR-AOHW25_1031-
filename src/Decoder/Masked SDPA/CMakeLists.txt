# cmake needs this line
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_program(WSL NAMES powershell.exe)

if (NOT WSL)
    set(CMAKE_C_COMPILER gcc-13)
    set(CMAKE_CXX_COMPILER g++-13)
    set(XRT_INC_DIR /opt/xilinx/xrt/include CACHE STRING "Path to XRT cloned repo")
    set(XRT_LIB_DIR /opt/xilinx/xrt/lib CACHE STRING "Path to xrt_coreutil.lib")
else()
    set(XRT_INC_DIR C:/Technical/XRT/src/runtime_src/core/include CACHE STRING "Path to XRT cloned repo")
    set(XRT_LIB_DIR C:/Technical/xrtNPUfromDLL CACHE STRING "Path to xrt_coreutil.lib")
endif()

# --- MODIFICATION ---
# Set the target name for SDPA project
set(TARGET_NAME sdpa CACHE STRING "Target to be built")

SET (ProjectName ${TARGET_NAME})
SET (currentTarget ${TARGET_NAME})

set(CMAKE_CXX_STANDARD 23)

if ( WSL )
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
    add_compile_options(/Zc:__cplusplus)
endif ()

project(${ProjectName})

# Add the source files for our 'sdpa' target
add_executable(${currentTarget}
    ${CMAKE_CURRENT_SOURCE_DIR}/../runtime_lib/test_lib/test_utils.cpp
    test.cpp
)

# --- MODIFICATION ---
# Update compiler definitions to pass SDPA dimensions from the Makefile
target_compile_definitions(${currentTarget} PUBLIC 
    DISABLE_ABI_CHECK=1
    M_DIM=${M_DIM}
    K_DIM=${K_DIM}
)

target_include_directories (${currentTarget} PUBLIC
    ${XRT_INC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../runtime_lib/test_lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../runtime_lib
)

target_link_directories(${currentTarget} PUBLIC
    ${XRT_LIB_DIR}
)

target_link_libraries(${currentTarget} PUBLIC
    xrt_coreutil
)

