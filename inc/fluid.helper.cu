#pragma once

#include "helper.cu"
#include <stdint.h>

class FluidCell {
private:
  bool is_solid;
  Vector2d<float> velocity;
  float pressure;

public:
  __device__ __host__ inline FluidCell();
  __device__ __host__ inline FluidCell(bool is_solid);

  // getters
  __device__ __host__ inline bool get_is_solid() const;
  __device__ __host__ inline uint8_t get_s() const;
  __device__ __host__ inline Vector2d<float> get_velocity() const;
  __device__ __host__ inline float get_pressure() const;

  // setters
  __device__ __host__ inline void set_velocity_x(float x);
  __device__ __host__ inline void set_velocity_y(float y);
  __device__ __host__ inline void set_velocity(float x, float y);
  __device__ __host__ inline void set_pressure(float pressure);
};

__device__ __host__ inline void FluidCell::set_pressure(float pressure) {
  this->pressure = pressure;
}

__device__ __host__ inline FluidCell::FluidCell()
    : velocity(0, 0), is_solid(0), pressure(0) {}

__device__ __host__ inline FluidCell::FluidCell(bool is_solid)
    : velocity(0, 0), is_solid(is_solid), pressure(0) {}

__device__ __host__ inline bool FluidCell::get_is_solid() const { return is_solid; }

__device__ __host__ inline Vector2d<float> FluidCell::get_velocity() const {
  return velocity;
}

__device__ __host__ inline float FluidCell::get_pressure() const { return pressure; }

__device__ __host__ inline uint8_t FluidCell::get_s() const { return !is_solid; }

__device__ __host__ inline void FluidCell::set_velocity_x(float x) {
  this->velocity.set_x(x);
}

__device__ __host__ inline void FluidCell::set_velocity_y(float y) {
  this->velocity.set_y(y);
}

__device__ __host__ inline void FluidCell::set_velocity(float x, float y) {
  this->set_velocity_x(x);
  this->set_velocity_y(y);
}

class SmokeCell {
private:
  // This value should be between 0 and 1
  float smoke;

public:
  __device__ __host__ inline SmokeCell();
  __device__ __host__ inline float get_smoke() const;
  __device__ __host__ inline void set_smoke(float smoke);
};

__device__ __host__ inline SmokeCell::SmokeCell() : smoke(0) {}

__device__ __host__ inline float SmokeCell::get_smoke() const { return this->smoke; }

__device__ __host__ inline void SmokeCell::set_smoke(float smoke) {
  this->smoke = smoke;
}

class Cell {
  FluidCell fluid;
  SmokeCell smoke;
public:
  __device__ __host__ inline Cell();
  __device__ __host__ inline Cell(bool is_solid);

  // getters
  __device__ __host__ inline Vector2d<float> get_velocity() const;
  __device__ __host__ inline float get_smoke() const;
  __device__ __host__ inline bool is_solid() const;
  __device__ __host__ inline uint8_t get_s() const;
  __device__ __host__ inline float get_pressure() const;

  // setters
  __device__ __host__ inline void set_velocity_x(float x);
  __device__ __host__ inline void set_velocity_y(float y);
  __device__ __host__ inline void set_velocity(float x, float y);
  __device__ __host__ inline void set_smoke(float smoke);
  __device__ __host__ inline void set_pressure(float pressure);
};

__device__ __host__ inline float Cell::get_pressure() const {
  return this->fluid.get_pressure();
}

__device__ __host__ inline void Cell::set_pressure(float pressure) {
  this->fluid.set_pressure(pressure);
}

__device__ __host__ inline Cell::Cell(bool is_solid) : smoke(), fluid(is_solid) {}
__device__ __host__ inline Cell::Cell() : smoke(), fluid() {}

__device__ __host__ inline void Cell::set_smoke(float smoke) {
  this->smoke.set_smoke(smoke);
}

__device__ __host__ inline Vector2d<float> Cell::get_velocity() const {
  return this->fluid.get_velocity();
}

__device__ __host__ inline float Cell::get_smoke() const {
  return this->smoke.get_smoke();
}

__device__ __host__ inline bool Cell::is_solid() const {
  return this->fluid.get_is_solid();
}

__device__ __host__ inline uint8_t Cell::get_s() const { return this->fluid.get_s(); }

__device__ __host__ inline void Cell::set_velocity_x(float x) {
  this->fluid.set_velocity_x(x);
}

__device__ __host__ inline void Cell::set_velocity_y(float y) {
  this->fluid.set_velocity_y(y);
}

__device__ __host__ inline void Cell::set_velocity(float x, float y) {
  this->set_velocity_x(x);
  this->set_velocity_y(y);
}
