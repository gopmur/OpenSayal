#pragma once

#include <string>

#include "json.hpp"

using json = nlohmann::json;

struct ColorConfig {
  int r;
  int g;
  int b;
  int a;
};

struct OpenMPConfig {
  int thread_count;
};

struct CudaConfig {
  int block_size_x;
  int block_size_y;
};

struct ThreadConfig {
  OpenMPConfig openMP;
  CudaConfig cuda;
};

struct ProjectionConfig {
  int n;
  float o;
};

struct PhysicsConfig {
  float g;
};

struct WindTunnelConfig {
  int pipe_height;
  int pipe_length;
  int smoke_length;
  int smoke_height;
  int smoke_count;
  float speed;
  float smoke;
};

struct ObstacleConfig {
  bool enable;
  int center_x;
  int center_y;
  float radius;
};

struct TimeConfig {
  float d_t;
  bool enable_real_time;
  float real_time_multiplier;
};

struct SmokeConfig {
  bool enable_decay;
  float decay_rate;
};

struct PathLineConfig {
  bool enable;
  int length;
  ColorConfig color;
  int distance;
};

struct FluidConfig {
  float density;
  float drag_coeff;
  float viscosity;
};

struct SimConfig {
  int height;
  int width;
  int cell_pixel_size;
  float cell_size;
  bool enable_drain;
  bool enable_pressure;
  bool enable_smoke;
  bool enable_interactive;
  ProjectionConfig projection;
  WindTunnelConfig wind_tunnel;
  PhysicsConfig physics;
  TimeConfig time;
  SmokeConfig smoke;
  ObstacleConfig obstacle;
};

struct ArrowConfig {
  ColorConfig color;
  bool enable;
  int distance;
  float length_multiplier;
  float disable_threshold;
  int head_length;
};

struct VisualConfig {
  ArrowConfig arrows;
  PathLineConfig path_line;
};

struct Config {
  ThreadConfig thread;
  SimConfig sim;
  FluidConfig fluid;
  VisualConfig visual;
};

class ConfigParser {
 private:
  const std::string config_file_name;

  template <typename T>
  static T get_or(json config_json, std::string key, T default_value);

 public:
  ConfigParser(std::string config_file_name);
  ConfigParser();
  Config parse() const;
};

template <typename T>
T ConfigParser::get_or(json config_json, std::string key, T default_value) {
  if (config_json.contains(key)) {
    return config_json.at(key);
  }
  if (key == "") {
    return default_value;
  }
  int first_dot_pos = key.find('.');
  if (first_dot_pos == std::string::npos) {
    return default_value;
  }
  auto parent_key = key.substr(0, first_dot_pos);
  auto child_key = key.substr(first_dot_pos + 1);
  try {
    json child_json = config_json.at(parent_key);
    return get_or(child_json, child_key, default_value);
  } catch (...) {
    return default_value;
  }
}