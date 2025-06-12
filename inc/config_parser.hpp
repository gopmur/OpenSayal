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
  bool enable_read_time;
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
};

struct SimConfig {
  ProjectionConfig projection;
  WindTunnelConfig wind_tunnel;
  PhysicsConfig physics;
  TimeConfig time;
  SmokeConfig smoke;
  ObstacleConfig obstacle;
  int height;
  int width;
  int cell_pixel_size;
  float cell_size;
  bool enable_drain;
  bool enable_pressure;
  bool enable_smoke;
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