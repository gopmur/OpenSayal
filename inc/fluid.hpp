#pragma once

#include <array>
#include <cstdint>
#include <tuple>

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
};

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
  float density;

 public:
  inline SmokeCell();
  inline float get_density() const;
  inline void set_density(float density);
};

inline SmokeCell::SmokeCell() : density(0) {}

inline float SmokeCell::get_density() const {
  return this->density;
}

inline void SmokeCell::set_density(float density) {
  this->density = density;
}

class Cell {
  FluidCell fluid;
  SmokeCell smoke;

 public:
  inline Cell();
  inline Cell(bool is_solid);

  // getters
  inline const Vector2d<float> get_velocity() const;
  inline const float get_density() const;
  inline const bool is_solid() const;
  inline const uint8_t get_s() const;

  // setters
  inline void set_velocity_x(float x);
  inline void set_velocity_y(float y);
  inline void set_velocity(float x, float y);
  inline void set_density(float density);
};

inline Cell::Cell(bool is_solid) : smoke(), fluid(is_solid) {}
inline Cell::Cell() : smoke(), fluid() {}

inline void Cell::set_density(float density) {
  this->smoke.set_density(density);
}

inline const Vector2d<float> Cell::get_velocity() const {
  return this->fluid.get_velocity();
}

inline const float Cell::get_density() const {
  return this->smoke.get_density();
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
  std::array<std::array<Cell, H>, W> grid;
  std::array<std::array<Vector2d<float>, H>, W> velocity_buffer;
  std::array<std::array<float, H>, W> smoke_buffer;

  inline Cell& get_mut_cell(int i, int j);
  inline Vector2d<float>& get_mut_velocity_buffer(int i, int j);
  inline void set_smoke_buffer(int i, int j, float density);
  inline float get_smoke_buffer(int i, int j);
  void step_projection(int i, int j);
  Vector2d<float> get_vertical_edge_velocity(int i, int j) const;
  Vector2d<float> get_horizontal_edge_velocity(int i, int j) const;
  float interpolate_smoke(float x, float y) const;
  float get_general_velocity_y(float x, float y) const;
  float get_general_velocity_x(float x, float y) const;
  inline bool index_is_valid(int i, int j) const;
  inline bool is_valid_fluid(int i, int j) const;
  inline Vector2d<float> get_center_position(int i, int j) const;
  inline Vector2d<float> get_u_position(int i, int j) const;
  inline Vector2d<float> get_v_position(int, int j) const;

  void apply_external_forces(float d_t);
  void apply_projection();
  void apply_advection(float d_t);
  void extrapolate();

 public:
  const float g = PHYSICS_G;
  const float o;
  const int cell_size;
  const int n;

  Fluid(float o, int n, int cell_size);

  // getters
  inline const Cell& get_cell(int i, int j) const;
  float get_divergence(int i, int j) const;
  uint8_t get_s(int i, int j) const;

  inline bool is_edge(int i, int j) const;

  inline Vector2d<float> get_general_velocity(float x, float y) const;

  void update(float d_t);
};

template <int H, int W>
inline bool Fluid<H, W>::is_edge(int i, int j) const {
  return i == 0 || j == 0 || i == W - 1 || j == H - 1;
}

template <int H, int W>
Fluid<H, W>::Fluid(float o, int n, int cell_size)
    : o(o), n(n), cell_size(cell_size) {
  for (auto i = 0; i < W; i++) {
    for (auto j = 0; j < H; j++) {
      Cell& cell = this->get_mut_cell(i, j);
      if (std::sqrt(std::pow((i - CIRCLE_POSITION_X), 2) +
                    std::pow((j - CIRCLE_POSITION_Y), 2)) < CIRCLE_RADIUS ||
          (i < PIPE_LENGTH && (j == H / 2 - PIPE_HEIGHT / 2 - 1 ||
                               j == H / 2 + PIPE_HEIGHT / 2 + 1))) {
        cell = Cell(true);
      } else {
        cell = Cell(i == 0 || j == 0 || j == H - 1);
      }
    }
  }
}

template <int H, int W>
inline const Cell& Fluid<H, W>::get_cell(int i, int j) const {
  return grid.at(i).at(j);
};
template <int H, int W>
inline void Fluid<H, W>::set_smoke_buffer(int i, int j, float density) {
  smoke_buffer.at(i).at(j) = density;
}

template <int H, int W>
inline float Fluid<H, W>::get_smoke_buffer(int i, int j) {
  return smoke_buffer.at(i).at(j);
}

template <int H, int W>
inline Vector2d<float>& Fluid<H, W>::get_mut_velocity_buffer(int i, int j) {
  return this->velocity_buffer.at(i).at(j);
}

template <int H, int W>
inline Cell& Fluid<H, W>::get_mut_cell(int i, int j) {
  return grid.at(i).at(j);
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

// ! Possible performance gain by memoizing s for each cell
template <int H, int W>
uint8_t Fluid<H, W>::get_s(int i, int j) const {
  const Cell& top_cell = get_cell(i, j + 1);
  const Cell& bottom_cell = get_cell(i, j - 1);
  const Cell& right_cell = get_cell(i + 1, j);
  const Cell& left_cell = get_cell(i - 1, j);

  auto top_s = top_cell.get_s();
  auto bottom_s = bottom_cell.get_s();
  auto right_s = right_cell.get_s();
  auto left_s = left_cell.get_s();

  auto s = top_s + bottom_s + right_s + left_s;

  return s;
}

template <int H, int W>
void Fluid<H, W>::step_projection(int i, int j) {
  Cell& cell = get_mut_cell(i, j);
  // if (cell.is_solid()) {
  //   return;
  // }

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
void Fluid<H, W>::apply_projection() {
  for (int _ = 0; _ < n; _++) {
    for (int i = 1; i < W - 1; i++) {
      for (int j = 1; j < H - 1; j++) {
        step_projection(i, j);
      }
    }
  }
}

template <int H, int W>
void Fluid<H, W>::apply_external_forces(float d_t) {
  for (int i = 0; i < W; i++) {
    for (int j = 0; j < H; j++) {
      Cell& cell = get_mut_cell(i, j);
      if (i == 1 && j >= H / 2 - PIPE_HEIGHT / 2 &&
          j <= H / 2 + PIPE_HEIGHT / 2) {
        cell.set_density(1);
        cell.set_velocity_x(WIND_SPEED);
      }
    }
  }
}

template <int H, int W>
inline bool Fluid<H, W>::index_is_valid(int i, int j) const {
  return i <= W && j <= H;
}

template <int H, int W>
inline bool Fluid<H, W>::is_valid_fluid(int i, int j) const {
  return index_is_valid(i, j) and not this->get_cell(i, j).is_solid();
}

template <int H, int W>
Vector2d<float> Fluid<H, W>::get_vertical_edge_velocity(int i, int j) const {
  const Cell& cell = this->get_cell(i, j);
  auto u = cell.get_velocity().get_x();

  auto avg_v = cell.get_velocity().get_y();

  if (is_valid_fluid(i - 1, j + 1)) {
    const Cell& top_left_cell = this->get_cell(i - 1, j + 1);
    avg_v += top_left_cell.get_velocity().get_y();
  }

  if (is_valid_fluid(i, j + 1)) {
    const Cell& top_right_cell = this->get_cell(i, j + 1);
    avg_v += top_right_cell.get_velocity().get_y();
  }

  if (is_valid_fluid(i - 1, j)) {
    const Cell& bottom_left_cell = this->get_cell(i - 1, j);
    avg_v += bottom_left_cell.get_velocity().get_y();
  }

  avg_v /= 4;

  return Vector2d<float>(u, avg_v);
}

template <int H, int W>
Vector2d<float> Fluid<H, W>::get_horizontal_edge_velocity(int i, int j) const {
  const Cell& cell = this->get_cell(i, j);
  auto v = cell.get_velocity().get_y();

  float avg_u = cell.get_velocity().get_x();

  if (is_valid_fluid(i + 1, j)) {
    const Cell& top_right_cell = this->get_cell(i + 1, j);
    avg_u += top_right_cell.get_velocity().get_x();
  }

  if (is_valid_fluid(i, j - 1)) {
    const Cell& bottom_left_cell = this->get_cell(i, j - 1);
    avg_u += bottom_left_cell.get_velocity().get_x();
  }

  if (is_valid_fluid(i + 1, j - 1)) {
    const Cell& bottom_right_cell = this->get_cell(i + 1, j - 1);
    avg_u += bottom_right_cell.get_velocity().get_x();
  }

  avg_u /= 4;

  return Vector2d<float>(avg_u, v);
}

template <int H, int W>
float Fluid<H, W>::get_general_velocity_y(float x, float y) const {
  int i = x / this->cell_size;
  int j = y / this->cell_size;

  float in_x = x - i * this->cell_size;
  float in_y = y - j * this->cell_size;

  float avg_v = 0;

  // take average with the left cell
  if (in_x < this->cell_size / 2.0) {
    float d_x = this->cell_size / 2.0 - in_x;
    float w_x = d_x / this->cell_size;
    float w_y = in_y / this->cell_size;

    if (this->is_valid_fluid(i, j)) {
      const Cell& bottom_right_cell = this->get_cell(i, j);
      avg_v += w_y * w_x * bottom_right_cell.get_velocity().get_y();
    }

    if (this->is_valid_fluid(i - 1, j)) {
      const Cell& bottom_left_cell = this->get_cell(i - 1, j);
      avg_v = w_y * (1 - w_x) * bottom_left_cell.get_velocity().get_y();
    }

    if (this->is_valid_fluid(i - 1, j + 1)) {
      const Cell& top_left_cell = this->get_cell(i - 1, j + 1);
      avg_v = (1 - w_y) * (1 - w_x) * top_left_cell.get_velocity().get_y();
    }

    if (this->is_valid_fluid(i, j + 1)) {
      const Cell& top_right_cell = this->get_cell(i, j + 1);
      avg_v = (1 - w_y) * w_x * top_right_cell.get_velocity().get_y();
    }
  }
  // take average with the right cell
  else {
    float d_x = in_x - this->cell_size / 2.0;
    float w_x = d_x / this->cell_size;
    float w_y = in_y / this->cell_size;

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
float Fluid<H, W>::get_general_velocity_x(float x, float y) const {
  int i = x / this->cell_size;
  int j = y / this->cell_size;

  float in_x = x - i * this->cell_size;
  float in_y = y - j * this->cell_size;

  float avg_u = 0;

  // take average with the bottom cell
  if (in_y < this->cell_size / 2.0) {
    float d_y = this->cell_size / 2.0 - in_y;
    float w_x = in_x / this->cell_size;
    float w_y = d_y / this->cell_size;

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
    float w_x = in_x / this->cell_size;
    float w_y = d_y / this->cell_size;

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
      avg_u = (1 - w_y) * (1 - w_x) * top_right_cell.get_velocity().get_x();
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
void Fluid<H, W>::apply_advection(float d_t) {
  for (int i = 1; i < W - 1; i++) {
    for (int j = 1; j < H - 1; j++) {
      // update u
      Vector2d<float> current_pos = this->get_u_position(i, j);
      Vector2d<float> current_velocity = this->get_vertical_edge_velocity(i, j);
      auto prev_pos = current_pos - current_velocity * d_t;
      float new_velocity =
          this->get_general_velocity_x(prev_pos.get_x(), prev_pos.get_y());
      this->get_mut_velocity_buffer(i, j).set_x(new_velocity);

      // update v
      current_pos = this->get_v_position(i, j);
      current_velocity = this->get_horizontal_edge_velocity(i, j);
      prev_pos = current_pos - current_velocity * d_t;
      new_velocity =
          this->get_general_velocity_y(prev_pos.get_x(), prev_pos.get_y());
      this->get_mut_velocity_buffer(i, j).set_y(new_velocity);

      // update smoke
      current_pos = this->get_center_position(i, j);
      current_velocity =
          this->get_general_velocity(current_pos.get_x(), current_pos.get_y());
      prev_pos = current_pos - current_velocity * d_t;
      float new_density = interpolate_smoke(prev_pos.get_x(), prev_pos.get_y());
      this->set_smoke_buffer(i, j, new_density);
    }
  }

  // move velocities
  for (int i = 1; i < W - 1; i++) {
    for (int j = 1; j < H - 1; j++) {
      // Vector2d<float> new_velocity = this->get_mut_velocity_buffer(i, j);
      // this->get_mut_cell(i, j).set_velocity(new_velocity.get_x(),
      //                                       new_velocity.get_y());
      float new_density = this->get_smoke_buffer(i, j);
      this->get_mut_cell(i, j).set_density(new_density);
    }
  }
}

template <int H, int W>
float Fluid<H, W>::interpolate_smoke(float x, float y) const {
  int i = x / this->cell_size;
  int j = y / this->cell_size;

  float in_x = x - i * this->cell_size;
  float in_y = y - j * this->cell_size;

  Vector2d<int> indices_1(i, j);
  Vector2d<int> indices_2;
  Vector2d<int> indices_3;
  Vector2d<int> indices_4;

  float distance_sum = 0;
  float avg_density = 0;

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

  auto w1 = distance_1 / distance_sum;
  auto w2 = distance_2 / distance_sum;
  auto w3 = distance_3 / distance_sum;
  auto w4 = distance_4 / distance_sum;

  if (is_valid_fluid(indices_1.get_x(), indices_1.get_y())) {
    const Cell& cell = this->get_cell(indices_1.get_x(), indices_1.get_y());
    avg_density += w1 * cell.get_density();
  }
  if (is_valid_fluid(indices_2.get_x(), indices_2.get_y())) {
    const Cell& cell = this->get_cell(indices_2.get_x(), indices_2.get_y());
    avg_density += w2 * cell.get_density();
  }
  if (is_valid_fluid(indices_3.get_x(), indices_1.get_y())) {
    const Cell& cell = this->get_cell(indices_3.get_x(), indices_3.get_y());
    avg_density += w3 * cell.get_density();
  }
  if (is_valid_fluid(indices_4.get_x(), indices_1.get_y())) {
    const Cell& cell = this->get_cell(indices_4.get_x(), indices_4.get_y());
    avg_density += w4 * cell.get_density();
  }

  return avg_density;
}

template <int H, int W>
void Fluid<H, W>::extrapolate() {
  for (int i = 0; i < W; i++) {
    {
      Cell& bottom_cell = this->get_mut_cell(i, 0);
      Cell& top_cell = this->get_mut_cell(i, 1);
      bottom_cell.set_velocity_x(top_cell.get_velocity().get_x());
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
void Fluid<H, W>::update(float d_t) {
  this->apply_external_forces(d_t);
  this->apply_projection();
  this->extrapolate();

  // this->apply_advection(d_t);
}