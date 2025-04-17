#pragma once

#include <cmath>
#include <concepts>
#include <cstdio>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include "fluid.hpp"

template <typename T>
concept arithmetic_concept = std::integral<T> or std::floating_point<T>;

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

  inline Vector2d<T> operator+(const Vector2d<T>& other);
  inline Vector2d<T> operator-(const Vector2d<T>& other);

  template <typename G>
    requires arithmetic_concept<G>
  friend Vector2d<T> operator*(const Vector2d<T>& vector, G scalar) {
    return Vector2d<T>(vector.x * scalar, vector.y * scalar);
  }
};

template <typename T>
  requires arithmetic_concept<T>
inline Vector2d<T> Vector2d<T>::operator+(const Vector2d<T>& other) {
  return Vector2d<T>(this->x + other.x, this->y + other.y);
}

template <typename T>
  requires arithmetic_concept<T>
inline Vector2d<T> Vector2d<T>::operator-(const Vector2d<T>& other) {
  return Vector2d<T>(this->x - other.x, this->y - other.y);
}

template <typename T>
  requires arithmetic_concept<T>
inline Vector2d<T>::Vector2d(T x, T y) : x(x), y(y) {}

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

template <typename T>
inline float get_distance(Vector2d<T> a, Vector2d<T> b) {
  return std::sqrt((a.get_x() - b.get_x()) * (a.get_x() - b.get_x()) +
                   (a.get_y() - b.get_y()) * (a.get_y() - b.get_y()));
}

void hsv_to_rgb(float h, float s, float v, uint8_t& r, uint8_t& g, uint8_t& b) {
  float c = v * s;
  float x = c * (1 - std::fabs(fmod(h / 60.0f, 2) - 1));
  float m = v - c;

  float r_, g_, b_;
  if (h < 60)      { r_ = c; g_ = x; b_ = 0; }
  else if (h < 120){ r_ = x; g_ = c; b_ = 0; }
  else if (h < 180){ r_ = 0; g_ = c; b_ = x; }
  else if (h < 240){ r_ = 0; g_ = x; b_ = c; }
  else if (h < 300){ r_ = x; g_ = 0; b_ = c; }
  else             { r_ = c; g_ = 0; b_ = x; }

  r = static_cast<uint8_t>((r_ + m) * 255);
  g = static_cast<uint8_t>((g_ + m) * 255);
  b = static_cast<uint8_t>((b_ + m) * 255);
}

uint64_t start_perf_counter() {
  struct perf_event_attr pe {};
  pe.type = PERF_TYPE_HARDWARE;
  pe.size = sizeof(pe);
  pe.config = PERF_COUNT_HW_INSTRUCTIONS;
  pe.disabled = 1;
  pe.exclude_kernel = 1;
  pe.exclude_hv = 1;

  // pid = 0 (this process), cpu = -1 (all CPUs), group = -1, flags = 0
  int fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
  if (fd == -1) {
      perror("perf_event_open");
      return -1;
  }

  ioctl(fd, PERF_EVENT_IOC_RESET, 0);
  ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
  return fd;
}

uint64_t stop_perf_counter(int fd) {
  ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
  uint64_t count = 0;
  read(fd, &count, sizeof(uint64_t));
  close(fd);
  return count;
}