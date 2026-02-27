cmake_minimum_required(VERSION 3.31)

# -----------------------------------------------------------------------------
# vcpkg library settings
# -----------------------------------------------------------------------------

set(VCPKG_OVERLAY_PORTS "${CMAKE_CURRENT_LIST_DIR}/third_party/SpatialHierarchyLibrary/vcpkg-overlays")
list(APPEND VCPKG_MANIFEST_FEATURES "runtimegpu")

if(WIN32 AND NOT DEFINED VCPKG_LIBRARY_LINKAGE)
    set(VCPKG_HOST_TRIPLET "x64-windows-static-md" CACHE STRING "")
    set(VCPKG_TARGET_TRIPLET "x64-windows-static-md" CACHE STRING "")
    set(VCPKG_LIBRARY_LINKAGE "static" CACHE STRING "")
endif()

# -----------------------------------------------------------------------------
# Project: Spatial Hierarchy Plugin
# -----------------------------------------------------------------------------

set(SPH_PLUGIN "SPHPlugin")
PROJECT(${SPH_PLUGIN}
    VERSION 0.0.1
    DESCRIPTION "A ManiVault plugin that wraps alxvth/SPH"
    LANGUAGES CXX)

# -----------------------------------------------------------------------------
# Set cmake flags
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/EHsc /MP /permissive- /Zc:__cplusplus /Zc:preprocessor )
endif()

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(OpenMP_RUNTIME_MSVC "llvm" FORCE)
endif()
find_package(OpenMP REQUIRED)

find_package(Qt6 COMPONENTS Core Widgets WebEngineWidgets REQUIRED)

find_package(ManiVault COMPONENTS Core PointData ImageData CONFIG)

find_package(unordered_dense CONFIG REQUIRED)

set(SPH_BUILD_VIS OFF CACHE BOOL "" FORCE)
set(SPH_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPH_BUILD_EVAL OFF CACHE BOOL "" FORCE)
set(SPH_BUILD_RUNTIME_GPU ON CACHE BOOL "" FORCE)
set(SPH_PRECOM_HEADER ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/SpatialHierarchyLibrary)

# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------

set(SPH_PLUGIN_SOURCES
    src/SphPlugin.h
    src/SphPlugin.cpp
    src/ComputeEmbeddingWrapper.h
    src/ComputeEmbeddingWrapper.cpp
    src/ComputeHierarchyWrapper.h
    src/ComputeHierarchyWrapper.cpp
    src/RefineAction.h
    src/RefineAction.cpp
    src/RefinedSelectionMapping.h
    src/RefinedSelectionMapping.cpp
)

set(SPH_SETTING_SOURCES
    src/SettingsAction.h
    src/SettingsAction.cpp
    src/SettingsHierarchyAction.h
    src/SettingsHierarchyAction.cpp
    src/SettingsTsneAction.h
    src/SettingsTsneAction.cpp
    src/SettingsAdvancedAction.h
    src/SettingsAdvancedAction.cpp
    src/DimensionSelectionAction.h
    src/DimensionSelectionAction.cpp
    src/TsneComputationAction.h
    src/TsneComputationAction.cpp
)

set(SPH_UTILS_SOURCES
    src/Utils.h
    src/Utils.cpp
)

set(AUX
    PluginInfo.json
)

set(SOURCES ${SPH_PLUGIN_SOURCES} ${SPH_SETTING_SOURCES} ${SPH_UTILS_SOURCES})

source_group(Plugin FILES ${SPH_PLUGIN_SOURCES})
source_group(Settings FILES ${SPH_SETTING_SOURCES})
source_group(Utils FILES ${SPH_UTILS_SOURCES})
source_group(Aux FILES ${AUX})

# -----------------------------------------------------------------------------
# CMake Target
# -----------------------------------------------------------------------------

add_library(${SPH_PLUGIN} SHARED ${SOURCES} ${AUX})

# -----------------------------------------------------------------------------
# Target include directories
# -----------------------------------------------------------------------------

# Include ManiVault core headers
target_include_directories(${SPH_PLUGIN} PRIVATE "${ManiVault_INCLUDE_DIR}")

target_include_directories(${SPH_PLUGIN}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# -----------------------------------------------------------------------------
# Target library linking
# -----------------------------------------------------------------------------

target_link_libraries(${SPH_PLUGIN} PRIVATE Qt6::Widgets)
target_link_libraries(${SPH_PLUGIN} PRIVATE Qt6::WebEngineWidgets)

target_link_libraries(${SPH_PLUGIN} PRIVATE ManiVault::Core)
target_link_libraries(${SPH_PLUGIN} PRIVATE ManiVault::PointData)
target_link_libraries(${SPH_PLUGIN} PRIVATE ManiVault::ImageData)

target_link_libraries(${SPH_PLUGIN} PRIVATE SPHLibrary)
target_link_libraries(${SPH_PLUGIN} PRIVATE hdidimensionalityreduction hdiutils hdidata)
target_link_libraries(${SPH_PLUGIN} PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(${SPH_PLUGIN} PRIVATE unordered_dense::unordered_dense)

# -----------------------------------------------------------------------------
# Target properties and features
# -----------------------------------------------------------------------------

set_target_properties(${SPH_PLUGIN} PROPERTIES AUTOMOC TRUE)

target_compile_features(${SPH_PLUGIN} PRIVATE cxx_std_20)

message(STATUS "Defining symbol __RUNTIME_GPU__ in target ${SPH_PLUGIN}")
target_compile_definitions (${SPH_PLUGIN} PRIVATE __RUNTIME_GPU__)

# Instruction sets and optimization
sph_check_and_set_AVX(${SPH_PLUGIN} ${SPH_USE_AVX})
sph_set_optimization_level(${SPH_PLUGIN} ${SPH_OPTIMIZATION_LEVEL})

# Warning levels
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${SPH_PLUGIN} PRIVATE /W3)
else()
    target_compile_options(${SPH_PLUGIN} PRIVATE -Wall)
endif()

# Suppress warning from biovault_bfloat16 in ManiVault core
target_compile_definitions(${SPH_PLUGIN} PRIVATE _SILENCE_CXX20_IS_POD_DEPRECATION_WARNING)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${SPH_PLUGIN} PRIVATE $<$<CONFIG:DEBUG>:/MDd> $<$<CONFIG:RELWITHDEBINFO>:/MD> $<$<CONFIG:RELEASE>:/MD>)
endif()

# -----------------------------------------------------------------------------
# Target installation
# -----------------------------------------------------------------------------

install(TARGETS ${SPH_PLUGIN}
    RUNTIME DESTINATION Plugins COMPONENT PLUGIN # Windows .dll
    LIBRARY DESTINATION Plugins COMPONENT PLUGIN # Linux/Mac .so
)

add_custom_command(TARGET ${SPH_PLUGIN} POST_BUILD
    COMMAND "${CMAKE_COMMAND}"
    --install ${CMAKE_CURRENT_BINARY_DIR}
    --component PLUGIN
    --config $<CONFIGURATION>
    --prefix ${ManiVault_INSTALL_DIR}/$<CONFIGURATION>
)

mv_handle_plugin_config(${SPH_PLUGIN} 0)

# -----------------------------------------------------------------------------
# Dependency installation
# -----------------------------------------------------------------------------

if(WIN32)
    # Automatically available when using find_package(ManiVault)
    # faiss links dynamically to openblas on windows but statically on linux
    mv_install_dependencies(${PROJECT_NAME})
endif()


# -----------------------------------------------------------------------------
# Miscellaneous
# -----------------------------------------------------------------------------

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Automatically set the debug environment (command + working directory) for MSVC 
    set_property(TARGET ${SPH_PLUGIN} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<IF:$<CONFIG:DEBUG>,${ManiVault_INSTALL_DIR}/Debug,$<IF:$<CONFIG:RELWITHDEBINFO>,${ManiVault_INSTALL_DIR}/RelWithDebInfo,${ManiVault_INSTALL_DIR}/Release>>)
    set_property(TARGET ${SPH_PLUGIN} PROPERTY VS_DEBUGGER_COMMAND $<IF:$<CONFIG:DEBUG>,"${ManiVault_INSTALL_DIR}/Debug/ManiVault Studio.exe",$<IF:$<CONFIG:RELWITHDEBINFO>,"${ManiVault_INSTALL_DIR}/RelWithDebInfo/ManiVault Studio.exe","${ManiVault_INSTALL_DIR}/Release/ManiVault Studio.exe">>)
endif()
