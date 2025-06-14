
#include "graphics_handler.cuh"

__device__ __host__ int GraphicsHandler::indx(int x, int y) {
  return y * this->width + x;
}
__device__ __host__ int GraphicsHandler::indx_traces(int i, int j, int k) {
  return (this->traces_height - j - 1) * this->traces_width *
             this->trace_length +
         i * this->trace_length + k;
}
__device__ __host__ int GraphicsHandler::indx_arrow_data(int i, int j) {
  return (this->arrow_data_height - j - 1) * this->arrow_data_width + i;
}

void GraphicsHandler::alloc_device_memory() {
  cudaMalloc(&d_this, sizeof(GraphicsHandler));
  cudaMalloc(
      &this->d_arrow_data,
      this->arrow_data_height * this->arrow_data_width * sizeof(ArrowData));
  cudaMalloc(&this->d_traces_x, this->traces_width * this->traces_height *
                                    this->trace_length * sizeof(int));
  cudaMalloc(&this->d_traces_y, this->traces_width * this->traces_height *
                                    this->trace_length * sizeof(int));
  cudaMalloc(&this->d_fluid_pixels, this->width * this->height * sizeof(int));
}

void GraphicsHandler::init_device_memory() {
  cudaMemcpy(d_this, this, sizeof(GraphicsHandler), cudaMemcpyHostToDevice);
}

void GraphicsHandler::alloc_host_memory() {
  this->fluid_pixels =
      static_cast<int*>(std::malloc(this->height * this->width * sizeof(int)));
  this->arrow_data = static_cast<ArrowData*>(std::malloc(
      this->arrow_data_height * this->arrow_data_width * sizeof(ArrowData)));
  this->traces_x =
      static_cast<int*>(std::malloc(this->traces_height * this->traces_width *
                                    this->trace_length * sizeof(int)));
  this->traces_y =
      static_cast<int*>(std::malloc(this->traces_height * this->traces_width *
                                    this->trace_length * sizeof(int)));
}

void GraphicsHandler::init_sdl() {
  this->window = nullptr;
  this->renderer = nullptr;
  this->fluid_texture = nullptr;
  this->format = nullptr;

  int window_height = cell_size * height;
  int window_width = cell_size * width;

  auto sdl_status = SDL_Init(SDL_INIT_VIDEO);
  if (sdl_status < 0) {
    auto sdl_error_message = SDL_GetError();
    Logger::error(
        std::format("video initialization failed: ", sdl_error_message));
    exit(EXIT_FAILURE);
  }

  this->window = SDL_CreateWindow("Fluid simulation", SDL_WINDOWPOS_CENTERED,
                                  SDL_WINDOWPOS_CENTERED, window_width,
                                  window_height, SDL_WINDOW_SHOWN);
  if (this->window == nullptr) {
    auto sdl_error_message = SDL_GetError();
    Logger::error(std::format("window creation failed: ", sdl_error_message));
    this->cleanup();
    exit(EXIT_FAILURE);
  }

  this->renderer =
      SDL_CreateRenderer(this->window, -1, SDL_RENDERER_ACCELERATED);
  if (this->renderer == nullptr) {
    auto sdl_error_message = SDL_GetError();
    Logger::error(std::format("renderer creation failed: ", sdl_error_message));
    this->cleanup();
    exit(EXIT_FAILURE);
  }

  this->fluid_texture =
      SDL_CreateTexture(this->renderer, SDL_PIXELFORMAT_RGBA8888,
                        SDL_TEXTUREACCESS_STREAMING, this->width, this->height);
  if (this->fluid_texture == nullptr) {
    auto sdl_error_message = SDL_GetError();
    Logger::error(std::format("texture creation failed: ", sdl_error_message));
    this->cleanup();
    exit(EXIT_FAILURE);
  }

  this->format = SDL_AllocFormat(SDL_PIXELFORMAT_RGBA8888);
  if (this->format == nullptr) {
    auto sdl_error_message = SDL_GetError();
    Logger::error(std::format("format allocation failed: ", sdl_error_message));
    this->cleanup();
    exit(EXIT_FAILURE);
  }
}

GraphicsHandler::GraphicsHandler(Config config)
    : width(config.sim.width),
      height(config.sim.height),
      cell_size(config.sim.cell_pixel_size),
      arrow_head_angle(ARROW_HEAD_ANGLE),
      arrow_head_length(config.visual.arrows.head_length),
      arrow_disable_thresh_hold(config.visual.arrows.disable_threshold),
      arrow_data_width(config.sim.width / config.visual.arrows.distance),
      arrow_data_height(config.sim.height / config.visual.arrows.distance),
      traces_width(config.sim.width / config.visual.path_line.distance),
      traces_height(config.sim.height / config.visual.path_line.distance),
      arrow_length_multiplier(config.visual.arrows.length_multiplier),
      block_size_x(config.thread.cuda.block_size_x),
      block_size_y(config.thread.cuda.block_size_y),
      arrow_distance(config.visual.arrows.distance),
      trace_distance(config.visual.path_line.distance),
      trace_length(config.visual.path_line.length),
      trace_color(config.visual.path_line.color),
      arrow_color(config.visual.arrows.color),
      enable_smoke(config.sim.enable_smoke),
      enable_pressure(config.sim.enable_pressure),
      enable_arrows(config.visual.arrows.enable),
      enable_traces(config.visual.path_line.enable) {
  this->alloc_host_memory();
  this->alloc_device_memory();
  this->init_device_memory();
  this->init_sdl();
  Logger::static_debug("graphics initialized successfully");
}

GraphicsHandler::~GraphicsHandler() {
  this->cleanup();
}

void GraphicsHandler::free_host_memory() {
  std::free(this->traces_x);
  std::free(this->traces_y);
  std::free(this->arrow_data);
  std::free(this->fluid_pixels);
}

void GraphicsHandler::free_device_memory() {
  cudaFree(d_this);
  cudaFree(d_traces_x);
  cudaFree(d_traces_y);
  cudaFree(d_arrow_data);
  cudaFree(d_fluid_pixels);
}

void GraphicsHandler::cleanup() {
  this->free_device_memory();
  this->free_host_memory();
  Logger::static_debug("cleaning up graphics");
  if (this->window != nullptr) {
    SDL_DestroyWindow(this->window);
  }
  if (this->renderer != nullptr) {
    SDL_DestroyRenderer(this->renderer);
  }
  if (this->fluid_texture != nullptr) {
    SDL_DestroyTexture(this->fluid_texture);
  }
  if (this->format != nullptr) {
    SDL_FreeFormat(format);
  }
  SDL_Quit();
}

__device__ ArrowData GraphicsHandler::make_arrow_data(int x,
                                                      int y,
                                                      float length,
                                                      float angle) {
  ArrowData arrow_data;

  if (length < this->arrow_disable_thresh_hold) {
    arrow_data.valid = false;
    return arrow_data;
  }

  arrow_data.valid = true;

  arrow_data.start_x = x;
  arrow_data.start_y = y;
  length *= this->arrow_length_multiplier;
  int x_offset = length * cos(angle);
  int y_offset = -length * sin(angle);
  arrow_data.end_x = x + x_offset;
  arrow_data.end_y = y + y_offset;

  int arrow_x_offset = -arrow_head_length * cos(angle + arrow_head_angle);
  int arrow_y_offset = arrow_head_length * sin(angle + arrow_head_angle);
  arrow_data.left_head_end_x = arrow_data.end_x + arrow_x_offset;
  arrow_data.left_head_end_y = arrow_data.end_y + arrow_y_offset;

  arrow_x_offset = -arrow_head_length * cos(arrow_head_angle - angle);
  arrow_y_offset = -arrow_head_length * sin(arrow_head_angle - angle);
  arrow_data.right_head_end_x = arrow_data.end_x + arrow_x_offset;
  arrow_data.right_head_end_y = arrow_data.end_y + arrow_y_offset;

  return arrow_data;
};

void GraphicsHandler::draw_arrow(const ArrowData& arrow_data) {
  if (not arrow_data.valid) {
    return;
  }
  SDL_RenderDrawLine(renderer, arrow_data.start_x, arrow_data.start_y,
                     arrow_data.end_x, arrow_data.end_y);
  SDL_RenderDrawLine(renderer, arrow_data.end_x, arrow_data.end_y,
                     arrow_data.left_head_end_x, arrow_data.left_head_end_y);
  SDL_RenderDrawLine(renderer, arrow_data.end_x, arrow_data.end_y,
                     arrow_data.right_head_end_x, arrow_data.right_head_end_y);
}

__device__ void GraphicsHandler::update_smoke_pixels(float smoke,
                                                     int x,
                                                     int y) {
  uint8_t color = 255 - static_cast<uint8_t>(smoke * 255);
  this->d_fluid_pixels[indx(x, y)] = map_rgba(255, color, color, 255);
}

__device__ void GraphicsHandler::update_smoke_and_pressure(float smoke,
                                                           float pressure,
                                                           int x,
                                                           int y,
                                                           float min_pressure,
                                                           float max_pressure) {
  float norm_p = 0;
  if (pressure < 0 and min_pressure != 0) {
    norm_p = -pressure / min_pressure;
  } else if (max_pressure != 0) {
    norm_p = pressure / max_pressure;
  }
  norm_p = clamp(norm_p, -1.0f, 1.0f);
  float hue = (1.0f - norm_p) * 120.0f;
  uint8_t r, g, b;
  hsv_to_rgb(hue, 1.0f, smoke, r, g, b);
  this->d_fluid_pixels[indx(x, y)] = map_rgba(r, g, b, 255);
}

__device__ void GraphicsHandler::update_pressure_pixel(float pressure,
                                                       int x,
                                                       int y,
                                                       float min_pressure,
                                                       float max_pressure) {
  float norm_p;
  if (pressure < 0) {
    norm_p = -pressure / min_pressure;
  } else {
    norm_p = pressure / max_pressure;
  }
  norm_p = clamp(norm_p, -1.0f, 1.0f);
  float hue = (1.0f - norm_p) * 120.0f;
  uint8_t r, g, b;
  hsv_to_rgb(hue, 1.0f, 1.0f, r, g, b);
  this->d_fluid_pixels[indx(x, y)] = map_rgba(r, g, b, 255);
}

__global__ void update_smoke_and_pressure_pixel_kernel(
    Fluid* d_fluid,
    GraphicsHandler* d_graphics_handler,
    float min_pressure,
    float max_pressure) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= d_fluid->width or j >= d_fluid->height) {
    return;
  }
  int x = i;
  int y = d_fluid->height - j - 1;
  if (d_fluid->d_is_solid[d_fluid->indx(i, j)]) {
    d_graphics_handler->d_fluid_pixels[d_graphics_handler->indx(x, y)] =
        map_rgba(80, 80, 80, 255);
  } else {
    d_graphics_handler->update_smoke_and_pressure(
        d_fluid->d_smoke[d_fluid->indx(i, j)],
        d_fluid->d_pressure[d_fluid->indx(i, j)], x, y, min_pressure,
        max_pressure);
  }
}

__global__ void update_smoke_pixel_kernel(Fluid* d_fluid,
                                          GraphicsHandler* d_graphics_handler) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= d_fluid->width or j >= d_fluid->height) {
    return;
  }
  int x = i;
  int y = d_fluid->height - j - 1;
  if (d_fluid->d_is_solid[d_fluid->indx(i, j)]) {
    d_graphics_handler->d_fluid_pixels[d_graphics_handler->indx(x, y)] =
        map_rgba(80, 80, 80, 255);
  } else {
    d_graphics_handler->update_smoke_pixels(
        d_fluid->d_smoke[d_fluid->indx(i, j)], x, y);
  }
}

__global__ void update_pressure_pixel_kernel(
    Fluid* d_fluid,
    GraphicsHandler* d_graphics_handler,
    float min_pressure,
    float max_pressure) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= d_fluid->width or j >= d_fluid->height) {
    return;
  }
  int x = i;
  int y = d_fluid->height - j - 1;
  if (d_fluid->d_is_solid[d_fluid->indx(i, j)]) {
    d_graphics_handler->d_fluid_pixels[d_graphics_handler->indx(x, y)] =
        map_rgba(80, 80, 80, 255);
  } else {
    d_graphics_handler->update_pressure_pixel(
        d_fluid->d_pressure[d_fluid->indx(i, j)], x, y, min_pressure,
        max_pressure);
  }
}

void GraphicsHandler::update_fluid_pixels(const Fluid& fluid) {
  float min_pressure = fluid.min_pressure;
  float max_pressure = fluid.max_pressure;
  int block_dim_x = this->block_size_x;
  int block_dim_y = this->block_size_y;
  int grid_dim_x = std::ceil(static_cast<float>(fluid.width) / block_dim_x);
  int grid_dim_y = std::ceil(static_cast<float>(fluid.height) / block_dim_y);
  auto block_dim = dim3(block_dim_x, block_dim_y, 1);
  auto grid_dim = dim3(grid_dim_x, grid_dim_y, 1);
  if (this->enable_pressure && this->enable_smoke) {
    update_smoke_and_pressure_pixel_kernel<<<grid_dim, block_dim>>>(
        fluid.d_this, d_this, min_pressure, max_pressure);
  } else if (this->enable_pressure) {
    update_pressure_pixel_kernel<<<grid_dim, block_dim>>>(
        fluid.d_this, d_this, min_pressure, max_pressure);
  } else if (this->enable_smoke) {
    update_smoke_pixel_kernel<<<grid_dim, block_dim>>>(fluid.d_this, d_this);
  }

  cudaMemcpyAsync(this->fluid_pixels, this->d_fluid_pixels,
                  sizeof(int) * this->height * this->width,
                  cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}

__global__ void update_center_velocity_arrow_kernel(
    GraphicsHandler* d_graphics_handler,
    Fluid* d_fluid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  i *= d_graphics_handler->arrow_distance;
  j *= d_graphics_handler->arrow_distance;
  if (i >= d_fluid->width or j >= d_fluid->height) {
    return;
  }
  d_graphics_handler->update_center_velocity_arrow_at(d_fluid, i, j);
}

__device__ void GraphicsHandler::update_center_velocity_arrow_at(
    const Fluid* d_fluid,
    int i,
    int j) {
  if (d_fluid->d_is_solid[d_fluid->indx(i, j)]) {
    return;
  }
  float x = (i + 0.5) * this->cell_size;
  float y = (this->height - j - 1 + 0.5) * this->cell_size;
  Vector2d<float> velocity =
      d_fluid->get_general_velocity(x, this->height * this->cell_size - y);
  auto vel_x = velocity.get_x();
  auto vel_y = velocity.get_y();
  auto angle = atan2(vel_y, vel_x);
  auto length = sqrt(vel_x * vel_x + vel_y * vel_y);
  this->d_arrow_data[this->indx_arrow_data(i / this->arrow_distance,
                                           j / this->arrow_distance)] =
      this->make_arrow_data(x, y, length, angle);
}

void GraphicsHandler::update_center_velocity_arrow(const Fluid& fluid) {
  int arrow_x = this->width / this->arrow_distance;
  int arrow_y = this->height / this->arrow_distance;
  int block_x = this->block_size_x;
  int block_y = this->block_size_y;
  int grid_x = std::ceil(arrow_x / static_cast<float>(block_x));
  int grid_y = std::ceil(arrow_y / static_cast<float>(block_y));
  auto block_dim = dim3(block_x, block_y, 1);
  auto gird_dim = dim3(grid_x, grid_y, 1);
  update_center_velocity_arrow_kernel<<<gird_dim, block_dim>>>(d_this,
                                                               fluid.d_this);
  cudaMemcpyAsync(this->arrow_data, this->d_arrow_data,
                  sizeof(ArrowData) * arrow_x * arrow_y,
                  cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int j = 0; j < this->height / this->arrow_distance; j++) {
    for (int i = 0; i < this->width / this->arrow_distance; i++) {
      this->draw_arrow(this->arrow_data[indx_arrow_data(i, j)]);
    }
  }
}

void GraphicsHandler::update_velocity_arrows(const Fluid& fluid) {
  if (this->enable_arrows) {
    SDL_SetRenderDrawColor(renderer, this->arrow_color.r, this->arrow_color.g,
                           this->arrow_color.b, this->arrow_color.a);
    this->update_center_velocity_arrow(fluid);
  }
}

__device__ void GraphicsHandler::update_trace_at(const Fluid* d_fluid,
                                                 float d_t,
                                                 int i,
                                                 int j) {
  const int trace_i = i / this->trace_distance;
  const int trace_j = j / this->trace_distance;

  if (d_fluid->d_is_solid[d_fluid->indx(i, j)]) {
    this->d_traces_x[indx_traces(trace_i, trace_j, 0)] = -1;
    this->d_traces_y[indx_traces(trace_i, trace_j, 0)] = -1;
  } else {
    d_fluid->trace(i, j, d_t,
                   &this->d_traces_x[indx_traces(trace_i, trace_j, 0)],
                   &this->d_traces_y[indx_traces(trace_i, trace_j, 0)],
                   this->trace_length);
  }
}
__global__ void update_traces_kernel(GraphicsHandler* d_graphics_handler,
                                     Fluid* d_fluid,
                                     float d_t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  i *= d_graphics_handler->trace_distance;
  j *= d_graphics_handler->trace_distance;

  if (i >= d_fluid->width || j >= d_fluid->height)
    return;

  d_graphics_handler->update_trace_at(d_fluid, d_t, i, j);
}

void GraphicsHandler::update_traces(const Fluid& fluid, float d_t) {
  const int trace_cols = this->width / this->trace_distance;
  const int trace_rows = this->height / this->trace_distance;

  dim3 block_dim(this->block_size_x, this->block_size_y);
  dim3 grid_dim((trace_cols + this->block_size_x - 1) / this->block_size_x,
                (trace_rows + this->block_size_y - 1) / this->block_size_y);
  update_traces_kernel<<<grid_dim, block_dim>>>(d_this, fluid.d_this, d_t);
  cudaMemcpyAsync(this->traces_x, this->d_traces_x,
                  trace_cols * trace_rows * this->trace_length * sizeof(int),
                  cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(this->traces_y, this->d_traces_y,
                  trace_cols * trace_rows * this->trace_length * sizeof(int),
                  cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  SDL_SetRenderDrawColor(this->renderer, this->trace_color.r,
                         this->trace_color.g, this->trace_color.b,
                         this->trace_color.a);

  SDL_Point* traces = static_cast<SDL_Point*>(
      std::malloc(this->trace_length * sizeof(SDL_Point)));
  for (int j = 0; j < trace_rows; j++) {
    for (int i = 0; i < trace_cols; i++) {
      for (int k = 0; k < this->trace_length; k++) {
        traces[k].x = this->traces_x[indx_traces(i, j, k)];
        traces[k].y = this->traces_y[indx_traces(i, j, k)];
      }
      if (traces_x[indx_traces(i, j, 0)] < 0)
        continue;
      SDL_RenderDrawLines(this->renderer, traces, this->trace_length);
    }
  }
}

void GraphicsHandler::update(const Fluid& fluid, float d_t) {
  this->update_fluid_pixels(fluid);
  SDL_UpdateTexture(this->fluid_texture, NULL, this->fluid_pixels,
                    this->width * sizeof(int));

  SDL_RenderClear(renderer);
  SDL_RenderCopy(renderer, this->fluid_texture, NULL, NULL);
  if (this->enable_traces) {
    this->update_traces(fluid, d_t);
  }

  this->update_velocity_arrows(fluid);

  SDL_RenderPresent(renderer);
}