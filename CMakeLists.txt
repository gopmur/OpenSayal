cmake_minimum_required(VERSION 4.0)

if (EXISTS "${CMAKE_SOURCE_DIR}/cmake/compiler_settings.cmake")
  include("${CMAKE_SOURCE_DIR}/cmake/compiler_settings.cmake")
else()
  include("${CMAKE_SOURCE_DIR}/cmake/compiler_settings.default.cmake")
  message(WARNING "No compiler settings file found, using default settings.")
endif()

project(OpenSayal CXX CUDA)

# set c++ version
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "/O2")
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math --ptxas-options=-O3 -rdc=true")

  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi")
  set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -G -rdc=true")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native ")
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math --ptxas-options=-O3 -rdc=true")

  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ")
  set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -G -rdc=true")
endif()

# store source files
file(GLOB_RECURSE PROJECT_SOURCE_FILES
  "${PROJECT_SOURCE_DIR}/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/*.cu"
)

# store include directories
set(PROJECT_INCLUDE_DIRECTORIES
  "inc"
  "lib/sdl/include"
  "lib/json/include/nlohmann"
  "lib/fmt/include"
)

# add source files
add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add include directories`
target_include_directories(${PROJECT_NAME} PUBLIC
  ${PROJECT_INCLUDE_DIRECTORIES}
)

add_subdirectory("lib/sdl")
add_subdirectory("lib/json")
add_subdirectory("lib/fmt")

target_link_directories(${PROJECT_NAME} PRIVATE
  "lib/sdl"
  "lib/json"
  "lib/fmt"
)

target_link_libraries(${PROJECT_NAME}
  SDL2
  SDL2main
  nlohmann_json
  fmt
)
