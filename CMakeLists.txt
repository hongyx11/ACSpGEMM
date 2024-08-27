cmake_minimum_required(VERSION 3.25)

# define project 
project(acspgemm LANGUAGES C CXX CUDA)
set(version 0.0.1)

# load cmake modules
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# set c++ standard and find packages
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80)
find_package(CUDAToolkit REQUIRED)

string(APPEND CMAKE_CUDA_FLAGS "-Xcompiler -Wall -D_FORCE_INLINES -DVERBOSE --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr")

option(CUDA_BUILD_INFO "Build with kernel statistics and line numbers" FALSE)
if (CUDA_BUILD_INFO)
    string(APPEND CMAKE_CUDA_FLAGS " -keep --ptxas-options=-v -lineinfo")
endif ()

option(CUDA_BUILD_DEBUG "Build with kernel debug" FALSE)
if (CUDA_BUILD_DEBUG)
    string(APPEND CMAKE_CUDA_FLAGS " -G")
endif ()

set(HEADERS
    include/acspgemm/common.cuh 
    include/acspgemm/common.h 
    include/acspgemm/Compare.h 
    include/acspgemm/consistent_memory.h 
    include/acspgemm/COO.h 
    include/acspgemm/CSR.h 
    include/acspgemm/CustomExceptions.h 
    include/acspgemm/dCSR.h 
    include/acspgemm/default_scheduling_traits.h 
    include/acspgemm/dVector.h 
    include/acspgemm/execution_stats.h 
    include/acspgemm/memory_space.h 
    include/acspgemm/MergeCaseOffsets.h 
    include/acspgemm/meta_utils.h 
    include/acspgemm/multi_arch_build.h 
    include/acspgemm/Multiply.h 
    include/acspgemm/Transpose.h 
    include/acspgemm/Vector.h
    include/acspgemm/device/acSpGEMM_ChunksToCSR.cuh 
    include/acspgemm/device/acSpGEMM_DetermineBlockStarts.cuh 
    include/acspgemm/device/acSpGEMM_MergeGeneralized.cuh 
    include/acspgemm/device/acSpGEMM_MergeMaxChunks.cuh 
    include/acspgemm/device/acSpGEMM_MergeSimple.cuh 
    include/acspgemm/device/acSpGEMM_SpGEMM.cuh 
    include/acspgemm/device/ARowStorage.cuh 
    include/acspgemm/device/Chunk.cuh 
    include/acspgemm/device/consistent_gpu_memory.h 
    include/acspgemm/device/HelperFunctions.cuh 
    include/acspgemm/device/MultiplyKernels.h
    include/acspgemm/device/SortAndCombine.cuh 
    include/acspgemm/device/WorkDistribution.cuh 
    include/acspgemm/devicetools/consistent_memory.h
    include/acspgemm/devicetools/error.h 
    include/acspgemm/devicetools/event.h
    include/acspgemm/devicetools/memory.h
    include/acspgemm/devicetools/memory_space.h
    include/acspgemm/devicetools/stream.h 
    include/acspgemm/devicetools/unique_handle.h
)

set(SRCS
    source/device/Transpose.cu
    source/device/memory.cpp
    source/device/Compare.cu
    source/COO.cpp
    source/CSR.cpp
    source/dCSR.cpp
    source/device/Multiply.cu
)

# AcSpGEMM Library
add_library(acspgemm ${HEADERS} ${SRCS})
target_include_directories(acspgemm PUBLIC 
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_compile_options(acspgemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-I${CMAKE_CURRENT_SOURCE_DIR}/include>)

add_executable(acspgemm_test source/main.cpp)
target_include_directories(acspgemm_test PUBLIC include)
target_link_libraries(acspgemm_test PUBLIC acspgemm)

# wrap function to install package, don't change it starting from here. 
function(configure_and_install_package PACKAGE_NAME VERSION)
    set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
    set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})

    install(TARGETS ${PACKAGE_NAME}
            EXPORT ${PACKAGE_NAME}Targets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

    install(EXPORT ${PACKAGE_NAME}Targets
            FILE ${PACKAGE_NAME}Targets.cmake
            NAMESPACE ${PACKAGE_NAME}::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME})

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TemplateConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
        PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR
    )

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}ConfigVersion.cmake"
        VERSION ${VERSION}
        COMPATIBILITY AnyNewerVersion)

    install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Config.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}ConfigVersion.cmake"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME})
endfunction()

configure_and_install_package(${PROJECT_NAME} ${version})