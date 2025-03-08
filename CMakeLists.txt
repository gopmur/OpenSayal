cmake_minimum_required(VERSION 3.31)
project(fluid-simulator)

set(CMAKE_CXX_STANDARD 20)

file(GLOB_RECURSE PROJECT_SOURCE_FILES 
  "${PROJECT_SOURCE_DIR}/src/*.cpp" 
  "${PROJECT_SOURCE_DIR}/src/*.c"
)
add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC "inc" "lib/sdl")

add_subdirectory("lib/sdl")
target_link_directories(${PROJECT_NAME} PRIVATE "lib/sdl")
target_link_libraries(${PROJECT_NAME} SDL2)