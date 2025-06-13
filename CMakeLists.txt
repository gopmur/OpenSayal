cmake_minimum_required(VERSION 4.0)
project(OpenSayal CXX CUDA)

# set c++ version
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /openmp")
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -Xcompiler=/openmp --use_fast_math --ptxas-options=-O3 -rdc=true")

  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /openmp /Od /Zi")
  set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -Xcompiler=/openmp -G -rdc=true")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -fopenmp")
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -Xcompiler=-fopenmp --use_fast_math --ptxas-options=-O3 -rdc=true")

  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fopenmp")
  set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -Xcompiler=-fopenmp -G -rdc=true")
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

target_link_directories(${PROJECT_NAME} PRIVATE
  "lib/sdl"
  "lib/json"
)

target_link_libraries(${PROJECT_NAME}
  SDL2
  SDL2main
  nlohmann_json
)
