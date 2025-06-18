#include <fstream>
#include <iostream>
#include <omp.h>

#include "json.hpp"

#include "config_parser.hpp"

using json = nlohmann::json;

ConfigParser::ConfigParser(std::string config_file_name)
    : config_file_name(config_file_name) {}
ConfigParser::ConfigParser() : config_file_name("OpenSayal.conf.json") {}

Config ConfigParser::parse() const {
  Config config;
  std::ifstream config_file(this->config_file_name);
  json config_json = json::parse(config_file);
  config_file.close();

  int sim_height = ConfigParser::get_or(config_json, "sim.height", 1080);
  int sim_width = ConfigParser::get_or(config_json, "sim.width", 1920);

  config = {
      .thread =
          {
              .openMP = {.thread_count = ConfigParser::get_or(
                             config_json, "thread.openMP.thread_count",
                             omp_get_thread_num() / 2)},
              .cuda = {.block_size_x = ConfigParser::get_or(
                           config_json, "thread.cuda.block_size_x", 64),
                       .block_size_y = ConfigParser::get_or(
                           config_json, "thread.cuda.block_size_y", 1)},
          },
      .sim =
          {
              .height = sim_height,
              .width = sim_width,
              .cell_pixel_size =
                  ConfigParser::get_or(config_json, "sim.cell_pixel_size", 1),
              .cell_size =
                  ConfigParser::get_or(config_json, "sim.cell_size", 1.0f),
              .enable_drain =
                  ConfigParser::get_or(config_json, "sim.enable_drain", true),
              .enable_pressure = ConfigParser::get_or(
                  config_json, "sim.enable_pressure", false),
              .enable_smoke =
                  ConfigParser::get_or(config_json, "sim.enable_smoke", true),
              .enable_interactive = ConfigParser::get_or(
                  config_json, "sim.enable_interactive", false),
              .projection =
                  {
                      .n = ConfigParser::get_or(config_json, "sim.projection.n",
                                                50),
                      .o = ConfigParser::get_or(config_json, "sim.projection.o",
                                                1.9f),
                  },
              .wind_tunnel =
                  {

                      .pipe_height = ConfigParser::get_or(
                          config_json, "sim.wind_tunnel.pipe_height",
                          sim_height / 4),
                      .pipe_length = 0,
                      .smoke_length = ConfigParser::get_or(
                          config_json, "sim.wind_tunnel.smoke_length", 1),
                      .smoke_height = ConfigParser::get_or(
                          config_json, "sim.wind_tunnel.smoke_height",
                          sim_height / 4),
                      .smoke_count = ConfigParser::get_or(
                          config_json, "sim.wind_tunnel.smoke_count", 1),
                      .speed = ConfigParser::get_or(
                          config_json, "sim.wind_tunnel.speed", 0.0f),
                      .smoke = ConfigParser::get_or(
                          config_json, "sim.wind_tunnel.smoke", 1.0f),
                  },
              .physics =
                  {
                      .g = ConfigParser::get_or(config_json, "sim.physics.g",
                                                0.0f),
                  },
              .time =
                  {
                      .d_t = ConfigParser::get_or(config_json, "sim.time.d_t",
                                                  0.05f),
                      .enable_real_time = ConfigParser::get_or(
                          config_json, "sim.time.enable_real_time", false),
                      .real_time_multiplier = ConfigParser::get_or(
                          config_json, "sim.time.real_time_multiplier", 1.0f),
                  },
              .smoke =
                  {
                      .enable_decay = ConfigParser::get_or(
                          config_json, "sim.smoke.enable_decay", false),
                      .decay_rate = ConfigParser::get_or(
                          config_json, "sim.smoke.decay_rate", 0.05f),
                  },
              .obstacle =
                  {
                      .enable = ConfigParser::get_or(
                          config_json, "sim.obstacle.enable", true),
                      .center_x = ConfigParser::get_or(
                          config_json, "sim.obstacle.center_x", sim_width / 2),
                      .center_y = ConfigParser::get_or(
                          config_json, "sim.obstacle.center_y", sim_height / 2),
                      .radius = ConfigParser::get_or(
                          config_json, "sim.obstacle.radius",
                          std::min({sim_height, sim_width}) / 30.0f),
                  },
          },
      .fluid =
          {
              .density =
                  ConfigParser::get_or(config_json, "fluid.density", 1.0f),
              .drag_coeff =
                  ConfigParser::get_or(config_json, "fluid.drag_coeff", 0.0f),
              .viscosity =
                  ConfigParser::get_or(config_json, "fluid.viscosity", 0.001f),
          },
      .visual =
          {
              .arrows =
                  {
                      .color =
                          {
                              .r = ConfigParser::get_or(
                                  config_json, "visual.arrows.color.r", 0),
                              .g = ConfigParser::get_or(
                                  config_json, "visual.arrows.color.g", 0),
                              .b = ConfigParser::get_or(
                                  config_json, "visual.arrows.color.b", 0),
                              .a = ConfigParser::get_or(
                                  config_json, "visual.arrows.color.a", 255),
                          },
                      .enable = ConfigParser::get_or(
                          config_json, "visual.arrows.enable", false),
                      .distance = ConfigParser::get_or(
                          config_json, "visual.arrows.distance", 20),
                      .length_multiplier = ConfigParser::get_or(
                          config_json, "visual.arrows.length_multiplier", 0.1f),
                      .disable_threshold =
                          ConfigParser::get_or(
                              config_json,
                              "visual.arrows.disable_threshold", 0.0f),
                      .head_length =
                          ConfigParser::get_or(config_json,
                                               "visual.arrows.head_length", 5),
                  },
              .path_line =
                  {
                      .enable =
                          ConfigParser::get_or(config_json,
                                               "visual.path_line.enable",
                                               false),
                      .length =
                          ConfigParser::get_or(config_json,
                                               "visual.path_line.length", 20),
                      .color =
                          {
                              .r =
                                  ConfigParser::get_or(
                                      config_json,
                                      "visual.path_line.color.r", 0),
                              .g =
                                  ConfigParser::
                                      get_or(config_json,
                                             "visual.path_line.color.g", 0),
                              .b =
                                  ConfigParser::
                                      get_or(config_json,
                                             "visual.path_line.color.b", 0),
                              .a =
                                  ConfigParser::
                                      get_or(config_json,
                                             "visual.path_line.color.a", 255),
                          },
                      .distance =
                          ConfigParser::get_or(config_json,
                                               "visual.path_line.distance", 20),
                  },
          },
  };

  return config;
}