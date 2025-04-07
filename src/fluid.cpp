#include "fluid.hpp"
#include <stdint.h>

// -------------------------------- FluidCell ------------------------------- //

FluidCell::FluidCell() : velocity(0, 0), is_solid(0), pressure(0) {}

FluidCell::FluidCell(bool is_solid)
    : velocity(0, 0), is_solid(is_solid), pressure(0) {}

bool FluidCell::get_is_solid() const {
  return is_solid;
}

Vector2d<float> FluidCell::get_velocity() const {
  return velocity;
}

float FluidCell::get_pressure() const {
  return pressure;
}

uint8_t FluidCell::get_s() const {
  return !is_solid;
}

void FluidCell::set_velocity_x(float x) {
  this->velocity.set_x(x);
}

void FluidCell::set_velocity_y(float y) {
  this->velocity.set_y(y);
}

void FluidCell::set_velocity(float x, float y) {
  this->set_velocity_x(x);
  this->set_velocity_y(y);
}

// -------------------------------- SmokeCell ------------------------------- //

SmokeCell::SmokeCell() : density(0) {}

float SmokeCell::get_density() const {
  return this->density;
}

// ---------------------------------- Cell ---------------------------------- //

Cell::Cell(bool is_solid) : smoke(), fluid(is_solid) {}
Cell::Cell() : smoke(), fluid() {}

const Vector2d<float> Cell::get_velocity() const {
  return this->fluid.get_velocity();
}

const float Cell::get_density() const {
  return this->smoke.get_density();
}

const bool Cell::is_solid() const {
  return this->fluid.get_is_solid();
}

const uint8_t Cell::get_s() const {
  return this->fluid.get_s();
}

void Cell::set_velocity_x(float x) {
  this->fluid.set_velocity_x(x);
}

void Cell::set_velocity_y(float y) {
  this->fluid.set_velocity_y(y);
}

void Cell::set_velocity(float x, float y) {
  this->set_velocity_x(x);
  this->set_velocity_y(y);
}
