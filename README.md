# Crowdy: Crowd Analysis, Simulation, and Risk Monitoring Pipeline

## Overview

Crowdy is a modular pipeline for advanced crowd analysis, simulation, and real-time risk alerting. It processes video to detect and track individuals, visualizes crowd flow, simulates panic/stampede scenarios, computes density/velocity/pressure fields, and generates real-time alerts for high-risk conditions. The system is designed for pedestrian/pilgrim monitoring, event safety, and research.

---

## Workflow Summary

### 1. Detection & Tracking

- **Script:** `run_yolov8x.py`
- **Input:** Video file (`--video`) and YOLOv8 model weights (`--model`)
- **Output:** 
  - Per-frame checkpoint CSVs (`checkpoints/`)
  - Periodic trajectory parquet files (`trajectories/`)
  - 3D point clouds (optional)
- **Description:** Detects people in each frame, tracks them using DeepSORT, and logs their positions and trajectories. Optionally, extracts 3D head positions using monocular depth estimation and homography.

**Example:**
```bash
python run_yolov8x.py --video video2.mp4 --model yolov8x.pt --checkpoints_dir checkpoints
```

---

### 2. Top-View Visualization

- **Scripts:** 
  - `src/animate_topview_tracks.py`
  - `src/animate_topview_tracks_with_heatmap.py`
  - `src/animate_topview.py`
  - `src/animate_tracks_on_video.py`
- **Input:** Trajectory parquet/CSV or checkpoints directory
- **Output:** GIFs (e.g., `topview_tracks.gif`, `topview_tracks_heatmap.gif`)
- **Description:** Visualizes tracks in a 2D top-view, optionally overlaying Gaussian heatmaps or showing tracks over the original video.

**Example:**
```bash
python src/animate_topview_tracks.py --traj_file trajectories/trajectories_100.parquet --output_gif topview_tracks.gif
```

---

### 3. Density, Velocity, and Pressure Field Computation

- **Scripts:** 
  - `src/generate_density_field.py`
  - `src/generate_vector_field.py`
  - `src/panic_monitor_pipeline.py`
- **Input:** Trajectory CSV (e.g., from panic simulation)
- **Output:** PNG overlays (`density_field.png`, `vector_field.png`), per-step overlays, and alert CSVs
- **Description:** Computes per-cell density, velocity, and pressure fields. Monitors for high-pressure (stampede risk) regions and generates real-time alerts.

**Example:**
```bash
python src/panic_monitor_pipeline.py --traj_csv panic_sfm_trajectories.csv --output_dir panic_monitor_frames
```

---

### 4. Forecasting (Future Position Prediction)

- **Scripts:** 
  - `src/sph_forecast.py`
  - `src/predict_future_topview.py`
  - `src/animate_future_topview_tracks.py`
- **Input:** Trajectory parquet/CSV
- **Output:** Forecasted positions (CSV), GIFs (e.g., `future_topview_tracks.gif`)
- **Description:** Predicts future positions using linear or SPH (Smoothed Particle Hydrodynamics) methods and animates the forecast.

---

### 5. Panic Simulation

- **Scripts:** 
  - `src/panic_simulate.py`
  - `src/panic_sfm_simulate.py`
- **Input:** Initial agent positions (`panic_agents_init.csv`), config (`pipeline_config.yaml`)
- **Output:** Simulated trajectories with panic state (`panic_sfm_trajectories.csv`)
- **Description:** Simulates panic propagation using a Social Force Model (SFM), with panic injected at a specified step and propagated to neighbors.

---

### 6. Panic Animation

- **Scripts:** 
  - `src/animate_panic_sfm_topview.py`
  - `src/animate_all_topview_frames.py`
  - `src/animate_key_frames.py`
- **Input:** Simulated SFM trajectory CSV
- **Output:** GIFs (e.g., `panic_sfm_topview.gif`, `all_topview_frames.gif`, `key_frames_animation.gif`)
- **Description:** Animates the simulated panic flow, coloring agents by panic state and highlighting key events (panic trigger, first stampede).

---

### 7. Monitoring & Real-Time Alerts

- **Script:** `src/panic_monitor_pipeline.py`
- **Input:** SFM trajectory CSV
- **Output:** 
  - Overlays for density, velocity, and pressure
  - Real-time alert CSVs (`pressure_alerts.csv`)
  - Predicted crowd goal (`predicted_goal.txt`)
- **Description:** Monitors simulation for high-risk regions, generates alerts, and predicts likely crowd goals based on flow direction.

---

### 8. Full Pipeline Automation

- **Scripts:** 
  - `run_pipeline_topview_heatmap.py`
  - `src/run_full_pipeline.py`
- **Description:** Automates the full workflow: detection → tracking → simulation → visualization → monitoring.

---

## Configuration

- **File:** `pipeline_config.yaml`
- **Sections:**
  - `panic_simulation`: Controls SFM simulation parameters (input/output files, steps, panic trigger, goal location)
  - `stampede_risk`: Controls monitoring parameters (grid size, pressure threshold, alert duration, output directory)

---

## Outputs

- **GIFs:** Visualizations of tracks, heatmaps, panic propagation, and forecasts
- **CSVs:** Trajectories, checkpoints, alerts, and forecasts
- **PNGs:** Density, velocity, and pressure field overlays
- **Text:** Predicted crowd goal, alert summaries

---

## Example End-to-End Usage

```bash
# 1. Run detection and tracking
python run_yolov8x.py --video video2.mp4 --model yolov8x.pt --checkpoints_dir checkpoints

# 2. Simulate panic scenario
python src/panic_sfm_simulate.py --config pipeline_config.yaml

# 3. Monitor for stampede risk and generate overlays/alerts
python src/panic_monitor_pipeline.py --traj_csv panic_sfm_trajectories.csv --output_dir panic_monitor_frames

# 4. Visualize results
python src/animate_all_topview_frames.py --traj_csv panic_sfm_trajectories.csv --output_gif all_topview_frames.gif
```

---

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies (PyTorch, OpenCV, pandas, matplotlib, imageio, etc.)

---

## Notes

- All scripts are modular and can be run independently or as part of the full pipeline.
- Outputs are organized by stage for easy analysis and visualization.
- The system is extensible for new simulation models, alerting logic, or visualization styles.

---

## Directory Structure

- `src/`: All core scripts and modules
- `checkpoints/`: Per-frame detection/tracking outputs
- `trajectories/`: Aggregated trajectory files
- `panic_monitor_frames/`: Monitoring overlays and alerts
- `*.gif`, `*.png`, `*.csv`: Visual and tabular outputs

---

## Contact

For questions, issues, or contributions, please open an issue or contact the maintainer.
