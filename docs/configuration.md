# Configuration Documentation

This document outlines the configurable parameters for the simulation system. Parameters are grouped by category for clarity.

## Thread Configuration

### `thread.cuda.block_size_x`

- **Type:** `int`  
- **Default:** `64`  
- **Description:** Number of threads along the X-axis per CUDA block. It is recommended not to modify this value.

### `thread.cuda.block_size_y`

- **Type:** `int`  
- **Default:** `1`  
- **Description:** Number of threads along the Y-axis per CUDA block. It is recommended not to modify this value.

## Simulation Parameters (`sim`)

### `sim.height`

- **Type:** `int`  
- **Default:** `1080`  
- **Description:** Simulation grid height in number of cells.

### `sim.width`

- **Type:** `int`  
- **Default:** `1920`  
- **Description:** Simulation grid width in number of cells.

### `sim.cell_pixel_size`

- **Type:** `int`  
- **Default:** `1`  
- **Description:** Size of each cell in pixels, used for rendering.

### `sim.cell_size`

- **Type:** `float`  
- **Default:** `1`  
- **Description:** Physical size of each simulation cell. Affects simulation scale.

### `sim.enable_drain`

- **Type:** `bool`  
- **Default:** `true`  
- **Description:** When disabled, adds a solid wall to the right boundary of the simulation.

### `sim.enable_pressure`

- **Type:** `bool`  
- **Default:** `false`  
- **Description:** Enables pressure calculation and visualization.

### `sim.enable_smoke`

- **Type:** `bool`  
- **Default:** `true`  
- **Description:** Enables smoke advection and visualization.

### `sim.enable_interactive`

- **Type:** `bool`  
- **Default:** `false`  
- **Description:** Allows mouse interaction with the simulation (e.g., applying forces).

#### Mouse Controls

- **Left click:** Apply outward force  
- **Right click:** Add smoke and outward force  
- **Middle click:** Apply inward force  
- **Scroll:** Adjust force magnitude

### Projection Settings

---

#### `sim.projection.n`

- **Type:** `int`  
- **Default:** `50`  
- **Description:** Number of iterations in the projection solver. Higher values yield more accurate results but reduce performance.

#### `sim.projection.o`

- **Type:** `float`  
- **Default:** `1.9`  
- **Description:** Over-relaxation coefficient for the projection step. Must be within `[0, 2]`.

### Wind Tunnel Settings

---

#### `sim.wind_tunnel.pipe_height`

- **Type:** `int`  
- **Default:** `sim.height / 4`  
- **Description:** Height of the inlet pipe for wind tunnel mode.

#### `sim.wind_tunnel.smoke_length`

- **Type:** `int`  
- **Default:** `1`  
- **Description:** Length along the X-axis where smoke is introduced. Larger values improve distribution uniformity at high speeds.

#### `sim.wind_tunnel.speed`

- **Type:** `float`  
- **Default:** `0`  
- **Description:** Inlet velocity of the wind tunnel.

#### `sim.wind_tunnel.smoke`

- **Type:** `float`  
- **Default:** `1`  
- **Description:** Smoke density introduced by the wind tunnel.

### Physics

---

#### `sim.physics.g`

- **Type:** `float`  
- **Default:** `0`  
- **Description:** Gravitational acceleration applied in the simulation.

### Time Control

---

#### `sim.time.d_t`

- **Type:** `float`  
- **Default:** `0.05`  
- **Description:** Simulation timestep. Ignored if real-time simulation is enabled.

#### `sim.time.enable_real_time`

- **Type:** `bool`  
- **Default:** `false`  
- **Description:** If enabled, timestep is derived from system clock, which may introduce instability.

#### `sim.time.real_time_multiplier`

- **Type:** `float`  
- **Default:** `1`  
- **Description:** Scales real time when `enable_real_time` is true.

### Smoke Settings

---

#### `sim.smoke.enable_decay`

- **Type:** `bool`  
- **Default:** `false`  
- **Description:** Enables smoke dissipation over time.

#### `sim.smoke.decay_rate`

- **Type:** `float`  
- **Default:** `0.05`  
- **Description:** Rate at which smoke fades. Ignored if decay is disabled.

### Obstacle

---

#### `sim.obstacle.enable`

- **Type:** `bool`  
- **Default:** `true`  
- **Description:** When disabled, removes the circular obstacle from the simulation.

#### `sim.obstacle.center_x`

- **Type:** `int`  
- **Default:** `sim.width / 2`  
- **Description:** X-coordinate of the obstacle’s center.

#### `sim.obstacle.center_y`

- **Type:** `int`  
- **Default:** `sim.height / 2`  
- **Description:** Y-coordinate of the obstacle’s center.

#### `sim.obstacle.radius`

- **Type:** `float`  
- **Default:** `min(sim.height, sim.width) / 30`  
- **Description:** Radius of the obstacle.

## Fluid Properties

### `fluid.density`

- **Type:** `float`  
- **Default:** `1`  
- **Description:** Fluid density (informational only; has no effect if pressure visualization is off).

### `fluid.drag_coeff`

- **Type:** `float`  
- **Default:** `0`  
- **Description:** Drag coefficient applied to fluid motion.

## Visualization Options

### Velocity Arrows (`visual.arrows`)

#### `visual.arrows.enable`

- **Type:** `bool`  
- **Default:** `false`  
- **Description:** Enables rendering of velocity arrows.

#### `visual.arrows.color.r/g/b`

- **Type:** `int`  
- **Default:** `0`  
- **Description:** RGB components for arrow color.

#### `visual.arrows.distance`

- **Type:** `int`  
- **Default:** `20`  
- **Description:** Cell spacing between arrows.

#### `visual.arrows.length_multiplier`

- **Type:** `float`  
- **Default:** `0.1`  
- **Description:** Multiplier applied to velocity magnitude to compute arrow length.

#### `visual.arrows.disable_threshold`

- **Type:** `float`  
- **Default:** `0`  
- **Description:** Arrows are not drawn for cells with velocity below this threshold.

#### `visual.arrows.head_length`

- **Type:** `int`  
- **Default:** `5`  
- **Description:** Arrowhead size in pixels.

### Path Lines (`visual.path_line`)

---

#### `visual.path_line.enable`

- **Type:** `bool`  
- **Default:** `false`  
- **Description:** Enables rendering of path lines for flow visualization.

#### `visual.path_line.length`

- **Type:** `int`  
- **Default:** `20`  
- **Description:** Number of iterations used to compute each path.

#### `visual.path_line.color.r/g/b`

- **Type:** `int`  
- **Default:** `0`  
- **Description:** RGB components for path line color.

#### `visual.path_line.distance`

- **Type:** `int`  
- **Default:** `20`  
- **Description:** Cell spacing between path lines.
