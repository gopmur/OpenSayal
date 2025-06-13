#include "helper.cuh"

__device__ __host__ void hsv_to_rgb(float h,
                                    float s,
                                    float v,
                                    uint8_t& r,
                                    uint8_t& g,
                                    uint8_t& b) {
  float c = v * s;
  float x = c * (1 - abs(fmodf(h / 60.0f, 2) - 1));
  float m = v - c;

  float r_, g_, b_;
  if (h < 60) {
    r_ = c;
    g_ = x;
    b_ = 0;
  } else if (h < 120) {
    r_ = x;
    g_ = c;
    b_ = 0;
  } else if (h < 180) {
    r_ = 0;
    g_ = c;
    b_ = x;
  } else if (h < 240) {
    r_ = 0;
    g_ = x;
    b_ = c;
  } else if (h < 300) {
    r_ = x;
    g_ = 0;
    b_ = c;
  } else {
    r_ = c;
    g_ = 0;
    b_ = x;
  }

  r = static_cast<uint8_t>((r_ + m) * 255);
  g = static_cast<uint8_t>((g_ + m) * 255);
  b = static_cast<uint8_t>((b_ + m) * 255);
}

__device__ __host__ int map_rgba(int r, int g, int b, int a) {
  return r << 24 | g << 16 | b << 8 | a;
}

__host__ __device__ float clamp(float x, float lower, float upper) {
  return (x < lower) ? lower : (x > upper) ? upper : x;
}