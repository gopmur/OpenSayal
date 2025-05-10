#pragma once

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
 