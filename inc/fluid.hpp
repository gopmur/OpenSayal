#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>

#include "SDL_rect.h"
#include "config.hpp"
#include "helper.hpp"

class FluidCell {
 private:
  bool is_solid;
  Vector2d<float> velocity;
  float pressure;

 public:
  inline FluidCell();
  inline FluidCell(bool is_solid);

  // getters
  inline bool get_is_solid() const;
  inline uint8_t get_s() const;
  inline Vector2d<float> get_velocity() const;
  inline float get_pressure() const;

  // setters
  inline void set_velocity_x(float x);
  inline void set_velocity_y(float y);
  inline void set_velocity(float x, float y);
  inline void set_pressure(float pressure);
};

inline void FluidCell::set_pressure(float pressure) {
  this->pressure = pressure;
}

inline FluidCell::FluidCell() : velocity(0, 0), is_solid(0), pressure(0) {}

inline FluidCell::FluidCell(bool is_solid)
    : velocity(0, 0), is_solid(is_solid), pressure(0) {}

inline bool FluidCell::get_is_solid() const {
  return is_solid;
}

inline Vector2d<float> FluidCell::get_velocity() const {
  return velocity;
}

inline float FluidCell::get_pressure() const {
  return pressure;
}

inline uint8_t FluidCell::get_s() const {
  return !is_solid;
}

inline void FluidCell::set_velocity_x(float x) {
  this->velocity.set_x(x);
}

inline void FluidCell::set_velocity_y(float y) {
  this->velocity.set_y(y);
}

inline void FluidCell::set_velocity(float x, float y) {
  this->set_velocity_x(x);
  this->set_velocity_y(y);
}

class SmokeCell {
 private:
  // This value should be between 0 and 1
  float smoke;

 public:
  inline SmokeCell();
  inline float get_smoke() const;
  inline void set_smoke(float smoke);
};

inline SmokeCell::SmokeCell() : smoke(0) {}

inline float SmokeCell::get_smoke() const {
  return this->smoke;
}

inline void SmokeCell::set_smoke(float smoke) {
  this->smoke = smoke;
}

class Cell {
  FluidCell fluid;
  SmokeCell smoke;

 public:
  inline Cell();
  inline Cell(bool is_solid);

  // getters
  inline const Vector2d<float> get_velocity() const;
  inline const float get_smoke() const;
  inline const bool is_solid() const;
  inline const uint8_t get_s() const;
  inline float get_pressure() const;

  // setters
  inline void set_velocity_x(float x);
  inline void set_velocity_y(float y);
  inline void set_velocity(float x, float y);
  inline void set_smoke(float smoke);
  inline void set_pressure(float pressure);
};

inline float Cell::get_pressure() const {
  return this->fluid.get_pressure();
}

inline void Cell::set_pressure(float pressure) {
  this->fluid.set_pressure(pressure);
}

inline Cell::Cell(bool is_solid) : smoke(), fluid(is_solid) {}
inline Cell::Cell() : smoke(), fluid() {}

inline void Cell::set_smoke(float smoke) {
  this->smoke.set_smoke(smoke);
}

inline const Vector2d<float> Cell::get_velocity() const {
  return this->fluid.get_velocity();
}

inline const float Cell::get_smoke() const {
  return this->smoke.get_smoke();
}

inline const bool Cell::is_solid() const {
  return this->fluid.get_is_solid();
}

inline const uint8_t Cell::get_s() const {
  return this->fluid.get_s();
}

inline void Cell::set_velocity_x(float x) {
  this->fluid.set_velocity_x(x);
}

inline void Cell::set_velocity_y(float y) {
  this->fluid.set_velocity_y(y);
}

inline void Cell::set_velocity(float x, float y) {
  this->set_velocity_x(x);
  this->set_velocity_y(y);
}

template <int H, int W>
class Fluid {
 private:
  Cell (*grid)[H];
  Vector2d<float> (*velocity_buffer)[H];
  float (*smoke_buffer)[H];
  uint8_t (*total_s)[H];

  inline Cell& get_mut_cell(int i, int j);
  inline Vector2d<float>& get_mut_velocity_buffer(int i, int j);
  inline void set_smoke_buffer(int i, int j, float smoke);
  inline float get_smoke_buffer(int i, int j);
  inline void step_projection(int i, int j, float d_t);
  inline float interpolate_smoke(float x, float y) const;
  inline float get_general_velocity_y(float x, float y) const;
  inline float get_general_velocity_x(float x, float y) const;
  inline bool index_is_valid(int i, int j) const;
  inline bool is_valid_fluid(int i, int j) const;
  inline Vector2d<float> get_center_position(int i, int j) const;
  inline Vector2d<float> get_u_position(int i, int j) const;
  inline Vector2d<float> get_v_position(int, int j) const;
  inline void set_pressure(int i, int j, float pressure);

  inline void zero_pressure();
  inline void update_pressure(int i, int j, float velocity_diff, float d_t);
  inline void apply_external_forces(float d_t);
  inline void apply_projection(float d_t);
  inline void apply_smoke_advection(float d_t);
  inline void apply_velocity_advection(float d_t);
  inline void extrapolate();
  inline void decay_smoke(float d_t);

 public:
  const float g = PHYSICS_G;
  const float o;
  const int cell_size;
  const int n;

  Fluid(float o, int n, int cell_size);
  ~Fluid();

  // getters
  inline const Cell& get_cell(int i, int j) const;
  float get_divergence(int i, int j) const;
  inline float get_pressure(int i, int j) const;
  uint8_t get_s(int i, int j) const;

  inline bool is_edge(int i, int j) const;

  inline Vector2d<float> get_general_velocity(float x, float y) const;
  inline Vector2d<float> get_vertical_edge_velocity(int i, int j) const;
  inline Vector2d<float> get_horizontal_edge_velocity(int i, int j) const;
  inline std::array<SDL_Point, TRACE_LENGTH> trace(int i,
                                                   int j,
                                                   float d_t) const;

  inline void update(float d_t);
};

template <int H, int W>
inline float Fluid<H, W>::get_pressure(int i, int j) const {
  return this->get_cell(i, j).get_pressure();
}

template <int H, int W>
inline void Fluid<H, W>::set_pressure(int i, int j, float pressure) {
  Cell& cell = this->get_mut_cell(i, j);
  cell.set_pressure(pressure);
}

template <int H, int W>
inline std::array<SDL_Point, TRACE_LENGTH> Fluid<H, W>::trace(int i,
                                                              int j,
                                                              float d_t) const {
  Vector2d<float> position = this->get_center_position(i, j);
  std::array<SDL_Point, TRACE_LENGTH> trace_points;
  trace_points[0] = {static_cast<int>(position.get_x()),
                     H - 1 - static_cast<int>(position.get_y())};
  for (int k = 1; k < TRACE_LENGTH; k++) {
    auto x = position.get_x();
    auto y = position.get_y();
    Vector2d<float> velocity = this->get_general_velocity(x, y);
    position = position + velocity * d_t;
    trace_points[k] = {static_cast<int>(position.get_x()),
                       H - 1 - static_cast<int>(position.get_y())};
  }
  return trace_points;
}

template <int H, int W>
inline bool Fluid<H, W>::is_edge(int i, int j) const {
  return i == 0 || j == 0 || i == W - 1 || j == H - 1;
}
template <int H, int W>
Fluid<H, W>::~Fluid() {
  delete[] this->grid;
  delete[] this->smoke_buffer;
  delete[] this->velocity_buffer;
  delete[] this->total_s;
}

template <int H, int W>
Fluid<H, W>::Fluid(float o, int n, int cell_size)
    : o(o), n(n), cell_size(cell_size) {
  this->grid = new Cell[W][H];
  this->smoke_buffer = new float[W][H];
  this->velocity_buffer = new Vector2d<float>[W][H];
  this->total_s = new uint8_t[W][H];
  for (auto i = 0; i < W; i++) {
    for (auto j = 0; j < H; j++) {
      Cell& cell = this->get_mut_cell(i, j);
      cell = Cell(
          i == 0 or j == 0 or j == H - 1 or
#if ENABLE_RIGHT_WALL
          i == W - 1 or
#endif
          (ENABLE_CIRCLE and std::sqrt(std::pow((i - CIRCLE_POSITION_X), 2) +
                                       std::pow((j - CIRCLE_POSITION_Y), 2)) <
                                 CIRCLE_RADIUS or
           (i < PIPE_LENGTH && (j == H / 2 - PIPE_HEIGHT / 2 - 1 or
                                j == H / 2 + PIPE_HEIGHT / 2 + 1))));
      this->total_s[i][j] = UINT8_MAX;
    }
  }
}

template <int H, int W>
inline const Cell& Fluid<H, W>::get_cell(int i, int j) const {
  return grid[i][j];
};

template <int H, int W>
inline void Fluid<H, W>::set_smoke_buffer(int i, int j, float smoke) {
  smoke_buffer[i][j] = smoke;
}

template <int H, int W>
inline float Fluid<H, W>::get_smoke_buffer(int i, int j) {
  return this->smoke_buffer[i][j];
}

template <int H, int W>
inline Vector2d<float>& Fluid<H, W>::get_mut_velocity_buffer(int i, int j) {
  return this->velocity_buffer[i][j];
}

template <int H, int W>
inline Cell& Fluid<H, W>::get_mut_cell(int i, int j) {
  return grid[i][j];
};

template <int H, int W>
float Fluid<H, W>::get_divergence(int i, int j) const {
  const Cell& cell = get_cell(i, j);
  const Cell& top_cell = get_cell(i, j + 1);
  const Cell& right_cell = get_cell(i + 1, j);

  auto u = cell.get_velocity().get_x();
  auto v = cell.get_velocity().get_y();
  auto top_v = top_cell.get_velocity().get_y();
  auto right_u = right_cell.get_velocity().get_x();

  auto divergence = right_u - u + top_v - v;

  return divergence;
}

template <int H, int W>
uint8_t Fluid<H, W>::get_s(int i, int j) const {
  if (total_s[i][j] != UINT8_MAX) {
    return total_s[i][j];
  }

  const Cell& top_cell = get_cell(i, j + 1);
  const Cell& bottom_cell = get_cell(i, j - 1);
  const Cell& right_cell = get_cell(i + 1, j);
  const Cell& left_cell = get_cell(i - 1, j);

  auto top_s = top_cell.get_s();
  auto bottom_s = bottom_cell.get_s();
  auto right_s = right_cell.get_s();
  auto left_s = left_cell.get_s();

  auto s = top_s + bottom_s + right_s + left_s;
  this->total_s[i][j] = s;

  return s;
}

template <int H, int W>
inline void Fluid<H, W>::zero_pressure() {
  for (int i = 0; i < W; i++) {
    for (int j = 0; j < H; j++) {
      this->get_mut_cell(i, j).set_pressure(0);
    }
  }
}
template <int H, int W>
inline void Fluid<H, W>::update_pressure(int i,
                                         int j,
                                         float velocity_diff,
                                         float d_t) {
  float pressure = this->get_pressure(i, j);
  pressure += velocity_diff * FLUID_DENSITY * CELL_SIZE / d_t;
  this->set_pressure(i, j, pressure);
}

template <int H, int W>
inline void Fluid<H, W>::step_projection(int i, int j, float d_t) {
  Cell& cell = get_mut_cell(i, j);
  if (cell.is_solid()) {
    return;
  }

  Cell& left_cell = get_mut_cell(i - 1, j);
  Cell& right_cell = get_mut_cell(i + 1, j);
  Cell& bottom_cell = get_mut_cell(i, j - 1);
  Cell& top_cell = get_mut_cell(i, j + 1);

  auto u = cell.get_velocity().get_x();
  auto v = cell.get_velocity().get_y();
  auto top_v = top_cell.get_velocity().get_y();
  auto right_u = right_cell.get_velocity().get_x();

  auto divergence = get_divergence(i, j);
  auto s = get_s(i, j);
  auto velocity_diff = this->o * (divergence / s);

#if ENABLE_PRESSURE
  if (i >= SMOKE_LENGTH + 1 or j >= H / 2 + PIPE_HEIGHT / 2 or
      j <= H / 2 - PIPE_HEIGHT / 2)
    this->update_pressure(i, j, velocity_diff, d_t);
#endif

  if (left_cell.get_s()) {
    u += velocity_diff;
    cell.set_velocity_x(u);
  }

  if (right_cell.get_s()) {
    right_u -= velocity_diff;
    right_cell.set_velocity_x(right_u);
  }

  if (bottom_cell.get_s()) {
    v += velocity_diff;
    cell.set_velocity_y(v);
  }

  if (top_cell.get_s()) {
    top_v -= velocity_diff;
    top_cell.set_velocity_y(top_v);
  }
}

template <int H, int W>
inline void Fluid<H, W>::apply_projection(float d_t) {
#if ENABLE_PRESSURE
  this->zero_pressure();
#endif
  for (int _ = 0; _ < n; _++) {
    for (int i = 1; i < W - 1; i++) {
      for (int j = i % 2 + 1; j < H - 1; j += 2) {
        this->step_projection(i, j, d_t);
      }
    }
    for (int i = 1; i < W - 1; i++) {
      for (int j = (i + 1) % 2 + 1; j < H - 1; j += 2) {
        this->step_projection(i, j, d_t);
      }
    }
  }
}

template <int H, int W>
inline void Fluid<H, W>::apply_external_forces(float d_t) {
  for (int i = 1; i < W - 1; i++) {
    for (int j = 1; j < H - 1; j++) {
      Cell& cell = get_mut_cell(i, j);
      if (i <= SMOKE_LENGTH and i != 0 && j >= H / 2 - PIPE_HEIGHT / 2 &&
          j <= H / 2 + PIPE_HEIGHT / 2) {
        cell.set_smoke(WIND_SMOKE);
        cell.set_velocity_x(WIND_SPEED);
      }
      auto vel_y = cell.get_velocity().get_y();
      cell.set_velocity_y(vel_y + PHYSICS_G * d_t);
    }
  }
}

template <int H, int W>
inline bool Fluid<H, W>::index_is_valid(int i, int j) const {
  return i < W and j < H and i >= 0 and j >= 0;
}

template <int H, int W>
inline bool Fluid<H, W>::is_valid_fluid(int i, int j) const {
  return index_is_valid(i, j) and not this->get_cell(i, j).is_solid();
}

template <int H, int W>
inline Vector2d<float> Fluid<H, W>::get_vertical_edge_velocity(int i,
                                                               int j) const {
  const Cell& cell = this->get_cell(i, j);
  auto u = cell.get_velocity().get_x();

  auto avg_v = cell.get_velocity().get_y();
  int count = 1;

  if (is_valid_fluid(i - 1, j + 1)) {
    const Cell& top_left_cell = this->get_cell(i - 1, j + 1);
    avg_v += top_left_cell.get_velocity().get_y();
    count++;
  }

  if (is_valid_fluid(i, j + 1)) {
    const Cell& top_right_cell = this->get_cell(i, j + 1);
    avg_v += top_right_cell.get_velocity().get_y();
    count++;
  }

  if (is_valid_fluid(i - 1, j)) {
    const Cell& bottom_left_cell = this->get_cell(i - 1, j);
    avg_v += bottom_left_cell.get_velocity().get_y();
    count++;
  }

  avg_v /= count;

  return Vector2d<float>(u, avg_v);
}

template <int H, int W>
inline Vector2d<float> Fluid<H, W>::get_horizontal_edge_velocity(int i,
                                                                 int j) const {
  const Cell& cell = this->get_cell(i, j);
  auto v = cell.get_velocity().get_y();

  float avg_u = cell.get_velocity().get_x();
  int count = 1;

  if (is_valid_fluid(i + 1, j)) {
    const Cell& top_right_cell = this->get_cell(i + 1, j);
    avg_u += top_right_cell.get_velocity().get_x();
    count++;
  }

  if (is_valid_fluid(i, j - 1)) {
    const Cell& bottom_left_cell = this->get_cell(i, j - 1);
    avg_u += bottom_left_cell.get_velocity().get_x();
    count++;
  }

  if (is_valid_fluid(i + 1, j - 1)) {
    const Cell& bottom_right_cell = this->get_cell(i + 1, j - 1);
    avg_u += bottom_right_cell.get_velocity().get_x();
    count++;
  }

  avg_u /= count;

  return Vector2d<float>(avg_u, v);
}

template <int H, int W>
inline float Fluid<H, W>::get_general_velocity_y(float x, float y) const {
  int i = x / this->cell_size;
  int j = y / this->cell_size;

  if (not this->is_valid_fluid(i, j)) {
    return 0;
  }

  float in_x = x - i * this->cell_size;
  float in_y = y - j * this->cell_size;

  float avg_v = 0;

  // take average with the left cell
  if (in_x < this->cell_size / 2.0) {
    float d_x = this->cell_size / 2.0 - in_x;
    float w_x = 1 - d_x / this->cell_size;
    float w_y = 1 - in_y / this->cell_size;

    if (this->is_valid_fluid(i, j)) {
      const Cell& bottom_right_cell = this->get_cell(i, j);
      avg_v += w_y * w_x * bottom_right_cell.get_velocity().get_y();
    }

    if (this->is_valid_fluid(i - 1, j)) {
      const Cell& bottom_left_cell = this->get_cell(i - 1, j);
      avg_v += w_y * (1 - w_x) * bottom_left_cell.get_velocity().get_y();
    }

    if (this->is_valid_fluid(i - 1, j + 1)) {
      const Cell& top_left_cell = this->get_cell(i - 1, j + 1);
      avg_v += (1 - w_y) * (1 - w_x) * top_left_cell.get_velocity().get_y();
    }

    if (this->is_valid_fluid(i, j + 1)) {
      const Cell& top_right_cell = this->get_cell(i, j + 1);
      avg_v += (1 - w_y) * w_x * top_right_cell.get_velocity().get_y();
    }
  }
  // take average with the right cell
  else {
    float d_x = in_x - this->cell_size / 2.0;
    float w_x = 1 - d_x / this->cell_size;
    float w_y = 1 - in_y / this->cell_size;

    if (this->is_valid_fluid(i, j)) {
      const Cell& bottom_left_cell = this->get_cell(i, j);
      avg_v += w_y * w_x * bottom_left_cell.get_velocity().get_y();
    }

    if (this->is_valid_fluid(i, j + 1)) {
      const Cell& top_left_cell = this->get_cell(i, j + 1);
      avg_v += (1 - w_y) * w_x * top_left_cell.get_velocity().get_y();
    }

    if (this->is_valid_fluid(i + 1, j + 1)) {
      const Cell& top_right_cell = this->get_cell(i + 1, j + 1);
      avg_v += (1 - w_y) * (1 - w_x) * top_right_cell.get_velocity().get_y();
    }

    if (this->is_valid_fluid(i + 1, j)) {
      const Cell& bottom_right_cell = this->get_cell(i + 1, j);
      avg_v += w_y * (1 - w_x) * bottom_right_cell.get_velocity().get_y();
    }
  }

  return avg_v;
}

template <int H, int W>
inline float Fluid<H, W>::get_general_velocity_x(float x, float y) const {
  int i = x / this->cell_size;
  int j = y / this->cell_size;

  if (not this->is_valid_fluid(i, j)) {
    return 0;
  }

  float in_x = x - i * this->cell_size;
  float in_y = y - j * this->cell_size;

  float avg_u = 0;

  // take average with the bottom cell
  if (in_y <= this->cell_size / 2.0) {
    float d_y = this->cell_size / 2.0 - in_y;
    float w_x = 1 - in_x / this->cell_size;
    float w_y = 1 - d_y / this->cell_size;

    if (this->is_valid_fluid(i, j)) {
      const Cell& top_left_cell = this->get_cell(i, j);
      avg_u += w_y * w_x * top_left_cell.get_velocity().get_x();
    }

    if (this->is_valid_fluid(i + 1, j)) {
      const Cell& top_right_cell = this->get_cell(i + 1, j);
      avg_u += w_y * (1 - w_x) * top_right_cell.get_velocity().get_x();
    }

    if (this->is_valid_fluid(i, j - 1)) {
      const Cell& bottom_left_cell = this->get_cell(i, j - 1);
      avg_u += (1 - w_y) * w_x * bottom_left_cell.get_velocity().get_x();
    }

    if (this->is_valid_fluid(i + 1, j - 1)) {
      const Cell& bottom_right_cell = this->get_cell(i + 1, j - 1);
      avg_u += (1 - w_y) * (1 - w_x) * bottom_right_cell.get_velocity().get_x();
    }
  }

  // take average with the top cell
  else {
    float d_y = in_y - this->cell_size / 2.0;
    float w_x = 1 - in_x / this->cell_size;
    float w_y = 1 - d_y / this->cell_size;

    if (this->is_valid_fluid(i, j)) {
      const Cell& bottom_left_cell = this->get_cell(i, j);
      avg_u += w_y * w_x * bottom_left_cell.get_velocity().get_x();
    }

    if (this->is_valid_fluid(i, j + 1)) {
      const Cell& top_left_cell = this->get_cell(i, j + 1);
      avg_u += (1 - w_y) * w_x * top_left_cell.get_velocity().get_x();
    }

    if (this->is_valid_fluid(i + 1, j)) {
      const Cell& bottom_right_cell = this->get_cell(i + 1, j);
      avg_u += w_y * (1 - w_x) * bottom_right_cell.get_velocity().get_x();
    }

    if (this->is_valid_fluid(i + 1, j + 1)) {
      const Cell& top_right_cell = this->get_cell(i + 1, j + 1);
      avg_u += (1 - w_y) * (1 - w_x) * top_right_cell.get_velocity().get_x();
    }
  }

  return avg_u;
}

template <int H, int W>
inline Vector2d<float> Fluid<H, W>::get_general_velocity(float x,
                                                         float y) const {
  float u = this->get_general_velocity_x(x, y);
  float v = this->get_general_velocity_y(x, y);
  return Vector2d<float>(u, v);
}

template <int H, int W>
inline Vector2d<float> Fluid<H, W>::get_center_position(int i, int j) const {
  return Vector2d<float>((i + 0.5) * this->cell_size,
                         (j + 0.5) * this->cell_size);
}

template <int H, int W>
inline Vector2d<float> Fluid<H, W>::get_u_position(int i, int j) const {
  return Vector2d<float>(i * this->cell_size, (j + 0.5) * this->cell_size);
}

template <int H, int W>
inline Vector2d<float> Fluid<H, W>::get_v_position(int i, int j) const {
  return Vector2d<float>((i + 0.5) * this->cell_size, j * this->cell_size);
}

template <int H, int W>
inline void Fluid<H, W>::apply_smoke_advection(float d_t) {
  for (int i = 1; i < W - 1; i++) {
    for (int j = 1; j < H - 1; j++) {
      Vector2d<float> current_pos = this->get_center_position(i, j);
      Vector2d<float> current_velocity =
          this->get_general_velocity(current_pos.get_x(), current_pos.get_y());
      auto prev_pos = current_pos - current_velocity * d_t;
      float new_smoke = interpolate_smoke(prev_pos.get_x(), prev_pos.get_y());
      this->set_smoke_buffer(i, j, new_smoke);
    }
  }

  for (int i = 1; i < W - 1; i++) {
    for (int j = 1; j < H - 1; j++) {
      float new_smoke = this->get_smoke_buffer(i, j);
      this->get_mut_cell(i, j).set_smoke(new_smoke);
    }
  }
}

template <int H, int W>
inline void Fluid<H, W>::apply_velocity_advection(float d_t) {
  for (int i = 1; i < W - 1; i++) {
    for (int j = 1; j < H - 1; j++) {
      Vector2d<float> current_pos = this->get_u_position(i, j);
      Vector2d<float> current_velocity = this->get_vertical_edge_velocity(i, j);
      auto prev_pos = current_pos - current_velocity * d_t;
      float new_velocity =
          this->get_general_velocity_x(prev_pos.get_x(), prev_pos.get_y());
      this->get_mut_velocity_buffer(i, j).set_x(new_velocity);

      current_pos = this->get_v_position(i, j);
      current_velocity = this->get_horizontal_edge_velocity(i, j);
      prev_pos = current_pos - current_velocity * d_t;
      new_velocity =
          this->get_general_velocity_y(prev_pos.get_x(), prev_pos.get_y());
      this->get_mut_velocity_buffer(i, j).set_y(new_velocity);
    }
  }

  for (int i = 1; i < W - 1; i++) {
    for (int j = 1; j < H - 1; j++) {
      Vector2d<float> new_velocity = this->get_mut_velocity_buffer(i, j);
      this->get_mut_cell(i, j).set_velocity(new_velocity.get_x(),
                                            new_velocity.get_y());
    }
  }
}

template <int H, int W>
inline float Fluid<H, W>::interpolate_smoke(float x, float y) const {
  int i = x / this->cell_size;
  int j = y / this->cell_size;

  float in_x = x - i * this->cell_size;
  float in_y = y - j * this->cell_size;

  Vector2d<int> indices_1(i, j);
  Vector2d<int> indices_2;
  Vector2d<int> indices_3;
  Vector2d<int> indices_4;

  float distance_sum = 0;
  float avg_smoke = 0;

  if (in_x < this->cell_size / 2.0 && in_y < this->cell_size / 2.0) {
    indices_2 = Vector2d<int>(i - 1, j);
    indices_3 = Vector2d<int>(i, j - 1);
    indices_4 = Vector2d<int>(i - 1, j - 1);
  } else if (in_x < this->cell_size / 2.0) {
    indices_2 = Vector2d<int>(i - 1, j);
    indices_3 = Vector2d<int>(i, j + 1);
    indices_4 = Vector2d<int>(i - 1, j + 1);
  } else if (in_y < this->cell_size / 2.0) {
    indices_2 = Vector2d<int>(i + 1, j);
    indices_3 = Vector2d<int>(i, j - 1);
    indices_4 = Vector2d<int>(i + 1, j - 1);
  } else {
    indices_2 = Vector2d<int>(i + 1, j);
    indices_3 = Vector2d<int>(i, j + 1);
    indices_4 = Vector2d<int>(i + 1, j + 1);
  }

  Vector2d<float> pos_1 =
      get_center_position(indices_1.get_x(), indices_1.get_y());
  Vector2d<float> pos_2 =
      get_center_position(indices_2.get_x(), indices_2.get_y());
  Vector2d<float> pos_3 =
      get_center_position(indices_3.get_x(), indices_3.get_y());
  Vector2d<float> pos_4 =
      get_center_position(indices_4.get_x(), indices_4.get_y());

  auto distance_1 = get_distance(Vector2d<float>(x, y), pos_1);
  auto distance_2 = get_distance(Vector2d<float>(x, y), pos_2);
  auto distance_3 = get_distance(Vector2d<float>(x, y), pos_3);
  auto distance_4 = get_distance(Vector2d<float>(x, y), pos_4);

  distance_sum = distance_1 + distance_2 + distance_3 + distance_4;

  float inv1 = 1.0 / (distance_1 + 1e-6);
  float inv2 = 1.0 / (distance_2 + 1e-6);
  float inv3 = 1.0 / (distance_3 + 1e-6);
  float inv4 = 1.0 / (distance_4 + 1e-6);

  float sum_inv = inv1 + inv2 + inv3 + inv4;

  float w1 = inv1 / sum_inv;
  float w2 = inv2 / sum_inv;
  float w3 = inv3 / sum_inv;
  float w4 = inv4 / sum_inv;

  if (is_valid_fluid(indices_1.get_x(), indices_1.get_y())) {
    const Cell& cell = this->get_cell(indices_1.get_x(), indices_1.get_y());
    avg_smoke += w1 * cell.get_smoke();
  }
  if (is_valid_fluid(indices_2.get_x(), indices_2.get_y())) {
    const Cell& cell = this->get_cell(indices_2.get_x(), indices_2.get_y());
    avg_smoke += w2 * cell.get_smoke();
  }
  if (is_valid_fluid(indices_3.get_x(), indices_3.get_y())) {
    const Cell& cell = this->get_cell(indices_3.get_x(), indices_3.get_y());
    avg_smoke += w3 * cell.get_smoke();
  }
  if (is_valid_fluid(indices_4.get_x(), indices_4.get_y())) {
    const Cell& cell = this->get_cell(indices_4.get_x(), indices_4.get_y());
    avg_smoke += w4 * cell.get_smoke();
  }

  return avg_smoke;
}

template <int H, int W>
inline void Fluid<H, W>::extrapolate() {
  for (int i = 0; i < W; i++) {
    {
      Cell& bottom_cell = this->get_mut_cell(i, 0);
      Cell& top_cell = this->get_mut_cell(i, 1);
      bottom_cell.set_velocity_x(top_cell.get_velocity().get_x());
      top_cell.set_velocity_y(0);
    }

    {
      Cell& bottom_cell = this->get_mut_cell(i, H - 2);
      Cell& top_cell = this->get_mut_cell(i, H - 1);
      top_cell.set_velocity_x(bottom_cell.get_velocity().get_x());
    }
  }
  for (int j = 0; j < H; j++) {
    {
      Cell& right_cell = this->get_mut_cell(W - 1, j);
      Cell& left_cell = this->get_mut_cell(W - 2, j);
      right_cell.set_velocity_y(left_cell.get_velocity().get_y());
    }
    {
      Cell& right_cell = this->get_mut_cell(1, j);
      Cell& left_cell = this->get_mut_cell(0, j);
      left_cell.set_velocity_y(right_cell.get_velocity().get_y());
    }
  }
}

template <int H, int W>
inline void Fluid<H, W>::decay_smoke(float d_t) {
  for (int i = 0; i < W; i++) {
    for (int j = 0; j < H; j++) {
      Cell& cell = this->get_mut_cell(i, j);
      float smoke = cell.get_smoke();
#if SMOKE_DECAY == 0
      cell.set_smoke(smoke);
#else
      cell.set_smoke(std::max({smoke - SMOKE_DECAY_RATE * d_t, 0.0}));
#endif
    }
  }
}

template <int H, int W>
inline void Fluid<H, W>::update(float d_t) {
  this->apply_external_forces(d_t);
  this->apply_projection(d_t);
  this->extrapolate();
  this->apply_velocity_advection(d_t);
#if ENABLE_SMOKE
  if (WIND_SMOKE != 0) {
    this->apply_smoke_advection(d_t);
    this->decay_smoke(d_t);
  }
#endif
}