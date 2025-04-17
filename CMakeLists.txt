cmake_minimum_required(VERSION 3.31)
project(fluid-simulator)

# set c/c++ version
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# store source files
file(GLOB_RECURSE PROJECT_SOURCE_FILES
  "${PROJECT_SOURCE_DIR}/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/*.c"
)

# store include directories
set(PROJECT_INCLUDE_DIRECTORIES
  "inc"
  "lib/sdl/include"
)

# add source files
add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_FILES})

# add include directories`
target_include_directories(${PROJECT_NAME} PUBLIC
  ${PROJECT_INCLUDE_DIRECTORIES}
)

# link sdl
add_subdirectory("lib/sdl")
target_link_directories(${PROJECT_NAME} PRIVATE
  "lib/sdl"
)
target_link_libraries(${PROJECT_NAME}
  SDL2
)
