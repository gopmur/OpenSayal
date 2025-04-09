#pragma once

#include <array>
#include <cstdint>
#include <format>
#include <stdexcept>

#include "config.hpp"
#include "helper.hpp"

template <typename T>
  requires arithmetic_concept<T>
class Vector2d {
 private:
  // the same as i
  T x;
  // the same as j
  T y;

 public:
  inline Vector2d(T x, T y);
  inline Vector2d();

  // getters
  inline T get_x() const;
  inline T get_y() const;

  // setters
  inline void set_x(T x);
  inline void set_y(T y);
};

template <typename T>
  requires arithmetic_concept<T>
inline Vector2d<T>::Vector2d(T x, T y) : x(x), y(x) {}

template <typename T>
  requires arithmetic_concept<T>
inline Vector2d<T>::Vector2d() : x(0), y(0) {}

template <typename T>
  requires arithmetic_concept<T>
inline T Vector2d<T>::get_x() const {
  return x;
}

template <typename T>
  requires arithmetic_concept<T>
inline T Vector2d<T>::get_y() const {
  return y;
}

template <typename T>
  requires arithmetic_concept<T>
inline void Vector2d<T>::set_x(T x) {
  this->x = x;
}

template <typename T>
  requires arithmetic_concept<T>
inline void Vector2d<T>::set_y(T y) {
  this->y = y;
}

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

template <uint32_t H, uint32_t W>
class Fluid {
 private:
  std::array<std::array<Cell, W>, H> grid;
  std::array<std::array<Vector2d<float>, W>, H> velocity_buffer;

  Cell& get_mut_cell(uint32_t i, uint32_t j);
  void step_projection(uint32_t i, uint32_t j);
  Vector2d<float> get_vertical_edge_velocity(uint32_t i, uint32_t j) const;
  Vector2d<float> get_horizontal_edge_velocity(uint32_t i, uint32_t j) const;
  inline Vector2d<float> get_general_velocity(float x, float y) const;
  float get_general_velocity_y(float x, float y) const;
  float get_general_velocity_x(float x, float y) const;
  inline bool index_is_valid(uint32_t i, uint32_t j) const;
  inline bool is_valid_fluid(uint32_t i, uint32_t j) const;
  inline Vector2d<float> get_position(uint32_t i, uint32_t j) const;

 public:
  const float g = PHYSICS_G;
  const float o;
  const uint32_t cell_size;
  const uint32_t n;

  Fluid(float o, uint32_t n, uint32_t cell_size);

  // getters
  inline const Cell& get_cell(uint32_t i, uint32_t j) const;
  float get_divergence(uint32_t i, uint32_t j) const;
  uint8_t get_s(uint32_t i, uint32_t j) const;

  inline bool is_edge(uint32_t i, uint32_t j) const;

  void apply_external_forces(float d_t);
  void apply_projection();
  void apply_advection(float d_t);
};

template <uint32_t H, uint32_t W>
inline bool Fluid<H, W>::is_edge(uint32_t i, uint32_t j) const {
  return i == 0 || j == 0 || i == W - 1 || j == H - 1;
}

template <uint32_t H, uint32_t W>
Fluid<H, W>::Fluid(float o, uint32_t n, uint32_t cell_size)
    : o(o), n(n), cell_size(cell_size) {
  for (auto i = 0; i < W; i++) {
    for (auto j = 0; j < H; j++) {
      Cell& cell = this->get_mut_cell(i, j);
      if (j >= 11 && j <= 16 && i == 13) {
        cell = Cell(true);
      } else {
        cell = Cell(is_edge(i, j));
      }
    }
  }
}

template <uint32_t H, uint32_t W>
inline const Cell& Fluid<H, W>::get_cell(uint32_t i, uint32_t j) const {
  return grid.at(H - j - 1).at(i);
};

template <uint32_t H, uint32_t W>
Cell& Fluid<H, W>::get_mut_cell(uint32_t i, uint32_t j) {
  return grid.at(H - j - 1).at(i);
};

template <uint32_t H, uint32_t W>
float Fluid<H, W>::get_divergence(uint32_t i, uint32_t j) const {
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
template <uint32_t H, uint32_t W>
uint8_t Fluid<H, W>::get_s(uint32_t i, uint32_t j) const {
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

template <uint32_t H, uint32_t W>
void Fluid<H, W>::step_projection(uint32_t i, uint32_t j) {
  Cell& cell = get_mut_cell(i, j);
  if (cell.is_solid()) {
    return;
  }

  Cell& top_cell = get_mut_cell(i, j + 1);
  Cell& bottom_cell = get_mut_cell(i, j - 1);
  Cell& right_cell = get_mut_cell(i + 1, j);
  Cell& left_cell = get_mut_cell(i - 1, j);

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

template <uint32_t H, uint32_t W>
void Fluid<H, W>::apply_projection() {
  for (uint32_t _ = 0; _ < n; _++) {
    for (uint32_t i = 1; i < W - 1; i++) {
      for (uint32_t j = 1; j < H - 1; j++) {
        step_projection(i, j);
      }
    }
  }
}

template <uint32_t H, uint32_t W>
void Fluid<H, W>::apply_external_forces(float d_t) {
  for (uint32_t i = 0; i < W; i++) {
    for (uint32_t j = 0; j < H; j++) {
      Cell& cell = get_mut_cell(i, j);
      if (cell.is_solid()) {
        continue;
      }

      if (j == 10 && i > 5 && i < 11) {
        cell.set_velocity(20, 20);
        cell.set_density(1);
      }
      auto v = cell.get_velocity().get_y();
      v += d_t * g;
      cell.set_velocity_y(v);
    }
  }
}

template <uint32_t H, uint32_t W>
inline bool Fluid<H, W>::index_is_valid(uint32_t i, uint32_t j) const {
  return i <= W && j <= H;
}

template <uint32_t H, uint32_t W>
inline bool Fluid<H, W>::is_valid_fluid(uint32_t i, uint32_t j) const {
  return index_is_valid(i, j) and not this->get_cell(i, j).is_solid();
}

template <uint32_t H, uint32_t W>
Vector2d<float> Fluid<H, W>::get_vertical_edge_velocity(uint32_t i,
                                                        uint32_t j) const {
  Cell& cell = this->get_cell(i, j);
  auto u = cell.get_velocity().get_x();

  auto avg_v = cell.get_velocity().get_y();

  if (is_valid_fluid(i - 1, j + 1)) {
    Cell& top_left_cell = this->get_cell(i - 1, j + 1);
    avg_v += top_left_cell.get_velocity().get_y();
  }

  if (is_valid_fluid(i, j + 1)) {
    Cell& top_right_cell = this->get_cell(i, j + 1);
    avg_v += top_right_cell.get_velocity().get_y();
  }

  if (is_valid_fluid(i - 1, j)) {
    Cell& bottom_left_cell = this->get_cell(i - 1, j);
    avg_v += bottom_left_cell.get_velocity().get_y();
  }

  avg_v /= 4;

  return Vector2d<float>(u, avg_v);
}

template <uint32_t H, uint32_t W>
Vector2d<float> Fluid<H, W>::get_horizontal_edge_velocity(uint32_t i,
                                                          uint32_t j) const {
  Cell& cell = this->get_cell(i, j);
  auto v = cell.get_velocity().get_y();

  float avg_u = cell.get_velocity().get_x();

  if (is_valid_fluid(i + 1, j)) {
    Cell& top_right_cell = this->get_cell(i + 1, j);
    avg_u += top_right_cell.get_velocity().get_x();
  }

  if (is_valid_fluid(i, j - 1)) {
    Cell& bottom_left_cell = this->get_cell(i, j - 1);
    avg_u += bottom_left_cell.get_velocity().get_x();
  }

  if (is_valid_fluid(i + 1, j - 1)) {
    Cell& bottom_right_cell = this->get_cell(i + 1, j - 1);
    avg_u += bottom_right_cell.get_velocity().get_x();
  }

  avg_u /= 4;

  return Vector2d<float>(avg_u, v);
}

template <uint32_t H, uint32_t W>
float Fluid<H, W>::get_general_velocity_y(float x, float y) const {
  uint32_t i = x / this->cell_size;
  uint32_t j = y / this->cell_size;

  float in_x = x - i * this->cell_size;
  float in_y = y - j * this->cell_size;

  float avg_v = 0;

  // take average with the left cell
  if (in_x < this->cell_size / 2.0) {
    float d_x = this->cell_size / 2.0 - in_x;
    float w_x = d_x / this->cell_size;
    float w_y = in_y / this->cell_size;

    if (this->is_valid_fluid(i, j)) {
      Cell& bottom_right_cell = this->get_cell(i, j);
      avg_v += w_y * w_x * bottom_right_cell.get_velocity().get_y();
    }

    if (this->is_valid_fluid(i - 1, j)) {
      Cell& bottom_left_cell = this->get_cell(i - 1, j);
      avg_v = w_y * (1 - w_x) * bottom_left_cell.get_velocity().get_y();
    }

    if (this->is_valid_fluid(i - 1, j + 1)) {
      Cell& top_left_cell = this->get_cell(i - 1, j + 1);
      avg_v = (1 - w_y) * (1 - w_x) * top_left_cell.get_velocity().get_y();
    }

    if (this->is_valid_fluid(i, j + 1)) {
      Cell& top_right_cell = this->get_cell(i, j + 1);
      avg_v = (1 - w_y) * w_x * top_right_cell.get_velocity().get_y();
    }
  }
  // take average with the right cell
  else {
    float d_x = in_x - this->cell_size / 2.0;
    float w_x = d_x / this->cell_size;
    float w_y = in_y / this->cell_size;

    if (this->is_valid_fluid(i, j)) {
      Cell& bottom_left_cell = this->get_cell(i, j);
      avg_v += w_y * w_x * bottom_left_cell.get_velocity().get_y();
    }

    if (this->is_valid_fluid(i, j + 1)) {
      Cell& top_left_cell = this->get_cell(i, j + 1);
      avg_v += (1 - w_y) * w_x * top_left_cell.get_velocity().get_y();
    }

    if (this->is_valid_fluid(i + 1, j + 1)) {
      Cell& top_right_cell = this->get_cell(i + 1, j + 1);
      avg_v += (1 - w_y) * (1 - w_x) * top_right_cell.get_velocity().get_y();
    }

    if (this->is_valid_fluid(i + 1, j)) {
      Cell& bottom_right_cell = this->get_cell(i + 1, j);
      avg_v += w_y * (1 - w_x) * bottom_right_cell.get_velocity().get_y();
    }
  }

  return avg_v / 4;
}

template <uint32_t H, uint32_t W>
float Fluid<H, W>::get_general_velocity_x(float x, float y) const {
  uint32_t i = x / this->cell_size;
  uint32_t j = y / this->cell_size;

  float in_x = x - i * this->cell_size;
  float in_y = y - j * this->cell_size;

  float avg_u = 0;

  // take average with the bottom cell
  if (in_y < this->cell_size / 2.0) {
    float d_y = this->cell_size / 2.0 - in_y;
    float w_x = in_x / this->cell_size;
    float w_y = d_y / this->cell_size;

    if (this->is_valid_fluid(i, j)) {
      Cell& top_left_cell = this->get_cell(i, j);
      avg_u += w_y * w_x * top_left_cell.get_velocity().get_x();
    }

    if (this->is_valid_fluid(i + 1, j)) {
      Cell& top_right_cell = this->get_cell(i + 1, j);
      avg_u += w_y * (1 - w_x) * top_right_cell.get_velocity().get_x();
    }

    if (this->is_valid_fluid(i, j - 1)) {
      Cell& bottom_left_cell = this->get_cell(i, j - 1);
      avg_u += (1 - w_y) * w_x * bottom_left_cell.get_velocity().get_x();
    }

    if (this->is_valid_fluid(i + 1, j - 1)) {
      Cell& bottom_right_cell = this->get_cell(i + 1, j - 1);
      avg_u += (1 - w_y) * (1 - w_x) * bottom_right_cell.get_velocity().get_x();
    }
  }

  // take average with the top cell
  else {
    float d_y = in_y - this->cell_size / 2.0;
    float w_x = in_x / this->cell_size;
    float w_y = d_y / this->cell_size;

    if (this->is_valid_fluid(i, j)) {
      Cell& bottom_left_cell = this->get_cell(i, j);
      avg_u += w_y * w_x * bottom_left_cell.get_velocity().get_x();
    }

    if (this->is_valid_fluid(i, j + 1)) {
      Cell& top_left_cell = this->get_cell(i, j + 1);
      avg_u += (1 - w_y) * w_x * top_left_cell.get_velocity().get_x();
    }

    if (this->is_valid_fluid(i + 1, j)) {
      Cell& bottom_right_cell = this->get_cell(i + 1, j);
      avg_u += w_y * (1 - w_x) * bottom_right_cell.get_velocity().get_x();
    }

    if (this->is_valid_fluid(i + 1, j + 1)) {
      Cell& top_right_cell = this->get_cell(i + 1, j + 1);
      avg_u = (1 - w_y) * (1 - w_x) * top_right_cell.get_velocity().get_x();
    }
  }

  return avg_u / 4;
}

template <uint32_t H, uint32_t W>
inline Vector2d<float> Fluid<H, W>::get_general_velocity(float x, float y) const {
  auto u = this->get_general_velocity_x(x, y);
  auto v = this->get_general_velocity_y(x, y);
  return Vector2d<float>(u, v);
}

template <uint32_t H, uint32_t W>
inline Vector2d<float> Fluid<H, W>::get_position(uint32_t i, uint32_t j) const {
  return Vector2d<float>(i * this->cell_size, j * this->cell_size);
}

template <uint32_t H, uint32_t W>
void Fluid<H, W>::apply_advection(float d_t) {
  
}