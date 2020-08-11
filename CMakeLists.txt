cmake_minimum_required (VERSION 2.8.11)

project(dpt)
# set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

include(CXX14)
check_for_cxx14_compiler(CXX14_COMPILER)

if(CXX14_COMPILER)
    enable_cxx14()
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} seems to have no C++14 support. Please try again with a more recent compiler version.")
endif()

# AVX does not do much benefit at the moment, but breaks compilation on some platforms.
# Disabled for now until AVX becomes important enough to reconsider.
set(XEON_ISA "SSE4.2" CACHE STRING "Selects highest ISA to support.")
set(USE_AVX FALSE CACHE BOOL "Use AVX.")
set(USE_AVX2 FALSE CACHE BOOL "Use AVX2.")

include(OptimizeForArchitecture)
OptimizeForArchitecture()

if (MSVC)
    # Needed by MSVC, but not added by OptimizeForArchitexture()
    add_definitions(-D__SSE__)
endif()

add_definitions(-DINSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")

if (USE_AVX)
    message(STATUS "Compiling with AVX support")
    set(__AVX__ 1)
elseif (USE_SSE4_2)
    message(STATUS "Compiling with SSE4.2 support")
elseif (USE_SSSE3)
    message(STATUS "Compiling with SSE3 support")
else()
    message(FATAL_ERROR "The target machine does not support SSE3. At least SSE3 is required")
endif()

if (MSVC)
    add_definitions(-DCONSTEXPR=const -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS)
else()
    add_definitions(-DCONSTEXPR=constexpr)
endif()

IF(COMMAND cmake_policy)
    if (POLICY CMP0043)
        cmake_policy(SET CMP0043 NEW)
    endif()
ENDIF(COMMAND cmake_policy)

foreach(flag ${Vc_ARCHITECTURE_FLAGS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
endforeach()

find_package(Threads)

# glog
SET(WITH_GFLAGS OFF CACHE BOOL "Use gflags")
SET(BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)
IF(WIN32)
    ADD_DEFINITIONS( -D GOOGLE_GLOG_DLL_DECL= )
ENDIF()
ADD_SUBDIRECTORY(externals/glog)
SET_PROPERTY(TARGET glog logging_unittest demangle_unittest utilities_unittest stl_logging_unittest PROPERTY FOLDER "ext")
INCLUDE_DIRECTORIES (externals/glog/src ${CMAKE_BINARY_DIR}/externals/glog)

#Eigen
set(EIGEN3_INCLUDE_DIR externals/Eigen3)
find_package (Eigen3 REQUIRED)
IF(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Eigen3 not found")
ENDIF()

#PThreads
find_package (Threads REQUIRED)
message(${CMAKE_THREAD_LIBS_INIT})

#OIIO
find_package(OpenImageIO REQUIRED)
IF(NOT OPENIMAGEIO_FOUND)
    message(FATAL_ERROR "OpenImageIO not found")
ENDIF()
include_directories(${OPENIMAGEIO_INCLUDE_DIRS})

#ZLIB
find_package(ZLIB REQUIRED)
if (NOT ZLIB_FOUND)
    message(FATAL_ERROR "ZLib not found")
endif()
include_directories( ${ZLIB_INCLUDE_DIRS} )

#Embree
set(ENABLE_STATIC_LIB ON CACHE BOOL "Build Embree as a static library." FORCE)
set(ENABLE_ISPC_SUPPORT OFF CACHE BOOL "Build Embree with support for ISPC applications." FORCE)
set(ENABLE_TUTORIALS OFF CACHE BOOL "Enable to build Embree tutorials" FORCE)
set(ENABLE_STAT_COUNTERS OFF CACHE BOOL "Enables statistic counters." FORCE)
set(ENABLE_RAY_MASK OFF CACHE BOOL "Enables ray mask support." FORCE)
set(ENABLE_BACKFACE_CULLING OFF CACHE BOOL "Enables backface culling." FORCE)
set(ENABLE_INTERSECTION_FILTER ON CACHE BOOL "Enables intersection filter callback." FORCE)
set(ENABLE_INTERSECTION_FILTER_RESTORE ON CACHE BOOL "Restores previous hit when hit is filtered out." FORCE)
set(ENABLE_TASKING_SYSTEM "INTERNAL" CACHE STRING "Selects tasking system" FORCE)
set(ENABLE_STATIC_RUNTIME OFF CACHE BOOL "Use the static version of the C/C++ runtime library." FORCE)
set(RTCORE_TASKING_SYSTEM OFF CACHE BOOL "Use the static version of the C/C++ runtime library." FORCE)
set(RTCORE_TASKING_SYSTEM "TBB" CACHE STRING "Selects tasking system")
add_subdirectory(externals/embree-2.17.7)

add_definitions(-D ENABLE_ISPC_SUPPORT=OFF)
add_definitions(-D RTCORE_TASKING_SYSTEM=INTERNAL)
add_definitions(-D ENABLE_TUTORIALS=OFF)
add_definitions(-D EMBREE_STATIC_LIB=1)
add_definitions(-D TASKING_INTERNAL)
include_directories(externals/embree-2.17.7/include)
include_directories(${PROJECT_BINARY_DIR}/externals/embree-2.17.7)
link_directories(${PROJECT_BINARY_DIR}/externals/embree-2.17.7)


if (CMAKE_COMPILER_IS_GNUCXX)
    set(CXX_WARNINGS "-Wall -Wextra -Wpointer-arith -Wcast-align -fstrict-aliasing -Wno-unused-local-typedefs -Wno-misleading-indentation -Wno-maybe-uninitialized -Wno-int-in-bool-context -Wno-implicit-fallthrough")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_WARNINGS} -fvisibility-inlines-hidden")
endif()


include_directories("src")
file(GLOB_RECURSE SOURCE_FILES "src/*.cpp" "src/*.h")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSINGLE_PRECISION" )

add_executable(dpt ${SOURCE_FILES})

if(WIN32)
    target_link_libraries(dpt embree glog ${CMAKE_THREAD_LIBS_INIT} ${OPENIMAGEIO_LIBRARIES} ${ZLIB_LIBRARIES} -dl -ldl )
elseif(APPLE)
    target_link_libraries(dpt boost_system embree glog ${CMAKE_THREAD_LIBS_INIT} ${OPENIMAGEIO_LIBRARIES} ${ZLIB_LIBRARIES} -dl -ldl )
else()
    #target_link_libraries(dpt stdc++fs embree glog ${CMAKE_THREAD_LIBS_INIT} ${OPENIMAGEIO_LIBRARIES} ${ZLIB_LIBRARIES} -dl -ldl )
    target_link_libraries(dpt stdc++fs glog embree ${OPENIMAGEIO_LIBRARIES} ${ZLIB_LIBRARIES} -dl -ldl )
endif()

