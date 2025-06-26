import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
from scipy.ndimage import uniform_filter1d, gaussian_filter
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde
from utils import get_grid_edges, get_cell_size, step_to_seconds, plot_density_field, plot_pressure_field

# --- Parameters ---
DEFAULT_GRID_SIZE = 100
DEFAULT_CONSECUTIVE_STEPS = 3
DEFAULT_PRESSURE_PERCENTILE = 95

# Compute density, velocity, and pressure fields for each step
# Now with moving average velocity, dynamic threshold, and temporal context

def monitor_panic_simulation(traj_csv, grid_size=DEFAULT_GRID_SIZE, pressure_percentile=DEFAULT_PRESSURE_PERCENTILE, consecutive_steps=DEFAULT_CONSECUTIVE_STEPS, output_dir='panic_monitor_frames'):
    df = pd.read_csv(traj_csv)
    steps = df['step'].max() + 1
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    x_edges, y_edges = get_grid_edges(x_min, x_max, y_min, y_max, grid_size)
    import os
    os.makedirs(output_dir, exist_ok=True)
    # For temporal context
    cell_alert_history = defaultdict(list)  # (cell_x, cell_y) -> list of bools per step
    all_pressures = []
    alerts = []
    # Precompute moving average velocities for each agent
    df = df.sort_values(['id', 'step'])
    df['vx'] = df.groupby('id')['x'].diff().fillna(0)
    df['vy'] = df.groupby('id')['y'].diff().fillna(0)
    # Apply moving average (window=3)
    df['vx_ma'] = df.groupby('id')['vx'].transform(lambda x: uniform_filter1d(x, size=3, mode='nearest'))
    df['vy_ma'] = df.groupby('id')['vy'].transform(lambda x: uniform_filter1d(x, size=3, mode='nearest'))
    for t in range(steps):
        frame_df = df[df['step'] == t]
        points = frame_df[['x', 'y']].values
        # Density field
        density, _, _ = np.histogram2d(points[:,0], points[:,1], bins=[x_edges, y_edges])
        # Velocity field (moving average)
        vx = np.zeros_like(density)
        vy = np.zeros_like(density)
        for _, row in frame_df.iterrows():
            ix = np.searchsorted(x_edges, row['x'], side='right') - 1
            iy = np.searchsorted(y_edges, row['y'], side='right') - 1
            if 0 <= ix < grid_size and 0 <= iy < grid_size:
                vx[ix, iy] += row['vx_ma']
                vy[ix, iy] += row['vy_ma']
        # Average velocity per cell
        mask = density > 0
        vx[mask] /= density[mask]
        vy[mask] /= density[mask]
        # Pressure field
        grad_vx = np.gradient(vx)[0]  # x-gradient
        grad_vy = np.gradient(vy)[1]  # y-gradient
        velocity_grad = np.sqrt(grad_vx**2 + grad_vy**2)
        pressure = density * velocity_grad
        all_pressures.append(pressure)
    # Compute dynamic threshold per step
    all_pressures = np.array(all_pressures)  # shape: (steps, grid, grid)
    dynamic_thresholds = np.percentile(all_pressures[all_pressures > 0], pressure_percentile)
    # Second pass: temporal context for alerts
    for t in range(steps):
        pressure = all_pressures[t]
        alert_cells = np.argwhere(pressure > dynamic_thresholds)
        for cell in alert_cells:
            cell_tuple = (cell[0], cell[1])
            cell_alert_history[cell_tuple].append(t)
    # Only alert if cell is above threshold for N consecutive steps
    final_alerts = []
    for cell, times in cell_alert_history.items():
        times = sorted(times)
        # Find runs of consecutive steps
        run = []
        for i, step in enumerate(times):
            if i == 0 or step == times[i-1] + 1:
                run.append(step)
            else:
                if len(run) >= consecutive_steps:
                    final_alerts.append({'cell_x': cell[0], 'cell_y': cell[1], 'start_step': run[0], 'end_step': run[-1]})
                run = [step]
        if len(run) >= consecutive_steps:
            final_alerts.append({'cell_x': cell[0], 'cell_y': cell[1], 'start_step': run[0], 'end_step': run[-1]})
    # Save overlays for visualization (last step)
    pressure = all_pressures[-1]
    plot_pressure_field(pressure, x_edges, y_edges, f'{output_dir}/pressure_last.png', title=f'Pressure Field (Last Step)')
    # Save alerts summary
    if final_alerts:
        pd.DataFrame(final_alerts).to_csv(f'{output_dir}/pressure_alerts.csv', index=False)
        print(f"Alerts summary saved to {output_dir}/pressure_alerts.csv")
    else:
        print("No high-pressure (stampede risk) detected.")
    # --- Enhanced Goal Prediction: Use last N steps and velocity direction ---
    GOAL_WINDOW = 5  # Number of recent steps to use for goal estimation
    recent_steps = range(max(0, steps - GOAL_WINDOW), steps)
    recent_frames = df[df['step'].isin(recent_steps)]
    # Use DBSCAN clustering on recent positions
    recent_positions = recent_frames[['x', 'y']].values
    if len(recent_positions) > 1:
        clustering = DBSCAN(eps=40, min_samples=5).fit(recent_positions)
        labels, counts = np.unique(clustering.labels_, return_counts=True)
        valid = labels[labels != -1]
        if len(valid) > 0:
            main_label = valid[np.argmax(counts[labels != -1])]
            goal_points = recent_positions[clustering.labels_ == main_label]
            goal_x, goal_y = np.mean(goal_points[:,0]), np.mean(goal_points[:,1])
        else:
            goal_x, goal_y = np.mean(recent_positions[:,0]), np.mean(recent_positions[:,1])
    else:
        goal_x, goal_y = np.mean(recent_positions[:,0]), np.mean(recent_positions[:,1])
    # Further refine using velocity direction
    recent_frames = recent_frames.copy()
    recent_frames['speed'] = np.hypot(recent_frames['vx_ma'], recent_frames['vy_ma'])
    moving = recent_frames[recent_frames['speed'] > 0.1]
    if not moving.empty:
        avg_dir = np.mean(moving[['vx_ma', 'vy_ma']].values, axis=0)
        norm = np.linalg.norm(avg_dir)
        if norm > 0:
            avg_dir = avg_dir / norm
            # Project from mean position in direction of avg_dir
            mean_pos = np.mean(moving[['x', 'y']].values, axis=0)
            goal_x, goal_y = mean_pos + avg_dir * 100  # Project 100 units ahead
    # Print and save predicted goal
    print(f"Predicted crowd goal: ({goal_x:.2f}, {goal_y:.2f})")
    with open(f"{output_dir}/predicted_goal.txt", "w") as f:
        f.write(f"Predicted crowd goal: ({goal_x:.2f}, {goal_y:.2f})\n")
    # --- Predict time of stampede risk (pre-threshold analysis) ---
    # For each alert cell, look at pressure trend before alert
    predicted_stampede_times = []
    for alert in final_alerts:
        cx, cy = int(alert['cell_x']), int(alert['cell_y'])
        start = alert['start_step']
        # Look at pressure in this cell for 5 steps before alert
        pre_steps = range(max(0, start-5), start)
        pre_pressures = [all_pressures[t][cx, cy] for t in pre_steps]
        # If pressure is rising rapidly, estimate when it will cross threshold
        if len(pre_pressures) >= 2:
            diffs = np.diff(pre_pressures)
            avg_rise = np.mean(diffs[-3:]) if len(diffs) >= 3 else np.mean(diffs)
            if avg_rise > 0:
                last_val = pre_pressures[-1]
                steps_to_threshold = (dynamic_thresholds - last_val) / avg_rise if avg_rise > 0 else 0
                predicted_time = start - 1 + steps_to_threshold
                predicted_stampede_times.append({'cell_x': cx, 'cell_y': cy, 'predicted_time': predicted_time})
    # Print and save predicted stampede times
    if predicted_stampede_times:
        print("Predicted stampede times (before threshold):")
        for pred in predicted_stampede_times:
            print(f"  Cell ({pred['cell_x']}, {pred['cell_y']}): step ~{pred['predicted_time']:.2f}")
        pd.DataFrame(predicted_stampede_times).to_csv(f"{output_dir}/predicted_stampede_times.csv", index=False)
    # --- Multi-feature, temporal, and clustering-based first stampede detection ---
    # Compute density and pressure for each step
    densities = []
    for t in range(steps):
        frame_df = df[df['step'] == t]
        points = frame_df[['x', 'y']].values
        density, _, _ = np.histogram2d(points[:,0], points[:,1], bins=[x_edges, y_edges])
        densities.append(density)
    densities = np.array(densities)  # shape: (steps, grid, grid)
    # Compute rate of change (delta) for density and pressure
    delta_density = np.diff(densities, axis=0, prepend=densities[[0]])
    delta_pressure = np.diff(all_pressures, axis=0, prepend=all_pressures[[0]])
    # Thresholds
    DENSITY_THRESH = 8
    PRESSURE_THRESH = np.percentile(all_pressures[all_pressures > 0], 80)
    DELTA_THRESH = 0.2
    # Find first cell(s) where all criteria are met
    first_stampede = None
    for t in range(steps):
        risk_mask = (
            (densities[t] >= DENSITY_THRESH) &
            (all_pressures[t] >= PRESSURE_THRESH) &
            (delta_density[t] > DELTA_THRESH)
        )
        if np.any(risk_mask):
            # Cluster adjacent high-risk cells
            coords = np.argwhere(risk_mask)
            if len(coords) > 1:
                clustering = DBSCAN(eps=1.5, min_samples=1).fit(coords)
                labels = clustering.labels_
                # Find largest cluster
                main_label = np.argmax(np.bincount(labels))
                cluster_coords = coords[labels == main_label]
                centroid = cluster_coords.mean(axis=0)
                cx, cy = centroid
            else:
                cx, cy = coords[0]
            # Map to real-world coordinates
            px = (x_edges[int(cx)] + x_edges[int(cx)+1]) / 2
            py = (y_edges[int(cy)] + y_edges[int(cy)+1]) / 2
            first_stampede = dict(
                step=t,
                px=px,
                py=py,
                duration=1  # Can be extended to track how long the risk lasts
            )
            break
    # --- Harmonized Visualization ---
    # Use first stampede step for all summary images, else last step
    summary_step = first_stampede['step'] if first_stampede else steps-1
    summary_title = f"Step {summary_step} (t={step_to_seconds(summary_step):.1f}s), Density ≥ {DENSITY_THRESH}, Pressure ≥ {PRESSURE_THRESH:.2f}, ΔDensity ≥ {DELTA_THRESH}"
    summary_frame = df[df['step'] == summary_step]
    # Density summary
    H, _, _ = np.histogram2d(summary_frame['x'], summary_frame['y'], bins=[x_edges, y_edges])
    plot_density_field(H, x_edges, y_edges, f'{output_dir}/density_summary.png', title=f'Density Field - {summary_title}')
    # Pressure summary
    pressure = all_pressures[summary_step]
    plot_pressure_field(pressure, x_edges, y_edges, f'{output_dir}/pressure_summary.png', title=f'Pressure Field - {summary_title}')
    # Gaussian heatmap overlay for summary step
    if len(summary_frame) > 1:
        sigma = 10
        heatmap, xedges, yedges = np.histogram2d(
            summary_frame['x'], summary_frame['y'],
            bins=200, range=[[x_min, x_max], [y_min, y_max]]
        )
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        extent = [x_min, x_max, y_min, y_max]
        plt.figure(figsize=(8, 8))
        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='jet', alpha=0.5, aspect='auto', zorder=1)
        plt.scatter(summary_frame['x'], summary_frame['y'], color='gray', s=40, edgecolor='white', zorder=2, label='Agents')
        if first_stampede:
            plt.scatter([first_stampede['px']], [first_stampede['py']], color='magenta', s=180, marker='*', label='First Stampede Spot', zorder=3)
            plt.text(first_stampede['px'], first_stampede['py'], f"t={summary_step}", color='white', fontsize=12, ha='center', va='center', bbox=dict(facecolor='magenta', alpha=0.7, edgecolor='none'))
        plt.title(f'Gaussian Heatmap - {summary_title}')
        plt.xlabel('x (meters)')
        plt.ylabel('y (meters)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/first_stampede_gaussian_heatmap.png')
        plt.close()
        print(f"Gaussian heatmap saved as {output_dir}/first_stampede_gaussian_heatmap.png")
    # First stampede summary (multi-feature)
    if first_stampede:
        plt.figure(figsize=(8,8))
        plt.scatter(summary_frame['x'], summary_frame['y'], color='gray', s=10, alpha=0.5, label='Agents')
        plt.scatter([first_stampede['px']], [first_stampede['py']], color='magenta', s=180, marker='*', label='First Stampede Spot')
        plt.text(first_stampede['px'], first_stampede['py'], f"t={summary_step}", color='white', fontsize=12, ha='center', va='center', bbox=dict(facecolor='magenta', alpha=0.7, edgecolor='none'))
        plt.title(f'First Stampede Spot Detected (Multi-Feature) - {summary_title}')
        plt.xlabel('x (meters)')
        plt.ylabel('y (meters)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/first_stampede_summary.png')
        plt.close()
        print(f"First stampede summary image saved as {output_dir}/first_stampede_summary.png")
    # --- Annotate plot with panic injection and first stampede times ---
    # Read panic_trigger_step from config if available
    panic_trigger_step = 0
    try:
        import yaml
        with open('pipeline_config.yaml') as f:
            cfg = yaml.safe_load(f)
        panic_trigger_step = cfg.get('panic_simulation', {}).get('panic_trigger_step', 0)
    except Exception as e:
        print(f"Could not read panic_trigger_step from config: {e}")
    # Find panic injection region (mean position of panicked agents at injection step)
    panic_injection_frame = df[df['step'] == panic_trigger_step]
    panic_agents = panic_injection_frame[panic_injection_frame['panic'] == True]
    if not panic_agents.empty:
        panic_x = panic_agents['x'].mean()
        panic_y = panic_agents['y'].mean()
    else:
        panic_x, panic_y = None, None
    # Only consider stampede risk after panic injection
    valid_alerts = [a for a in final_alerts if a['start_step'] >= panic_trigger_step]
    min_start = min(a['start_step'] for a in valid_alerts) if valid_alerts else None
    # Plot as before, but annotate both events
    if final_alerts:
        # ... (existing plot code up to plt.scatter/plt.text for risk zones) ...
        for alert in final_alerts:
            start, end = alert['start_step'], alert['end_step']
            duration = end - start + 1
            frame = df[df['step'] == start]
            cx, cy = int(alert['cell_x']), int(alert['cell_y'])
            x_edges = np.linspace(x_min, x_max, grid_size+1)
            y_edges = np.linspace(y_min, y_max, grid_size+1)
            in_cell = frame[(frame['x'] >= x_edges[cx]) & (frame['x'] < x_edges[cx+1]) &
                            (frame['y'] >= y_edges[cy]) & (frame['y'] < y_edges[cy+1])]
            if not in_cell.empty:
                px, py = in_cell['x'].mean(), in_cell['y'].mean()
                if start == min_start:
                    plt.scatter([px], [py], color='magenta', s=160, marker='s', label='First Stampede')
                    plt.text(px, py, f"First Stampede\nt={start}\n({duration})", color='white', fontsize=10, ha='center', va='center',
                             bbox=dict(facecolor='magenta', alpha=0.7, edgecolor='none'))
                else:
                    plt.scatter([px], [py], color='red', s=120, marker='s', label=None)
                    plt.text(px, py, f"t={start}-{end}\n({duration})", color='white', fontsize=8, ha='center', va='center',
                             bbox=dict(facecolor='red', alpha=0.5, edgecolor='none'))
        # Annotate panic injection
        if panic_x is not None and panic_y is not None:
            plt.scatter([panic_x], [panic_y], color='blue', s=180, marker='*', label='Panic Injection')
            plt.text(panic_x, panic_y, f"Panic Injected\nt={panic_trigger_step}", color='white', fontsize=10, ha='center', va='center',
                     bbox=dict(facecolor='blue', alpha=0.7, edgecolor='none'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monitor panic simulation for density, pressure, and stampede risk.')
    parser.add_argument('--traj_csv', type=str, default='panic_sfm_trajectories.csv', help='CSV of simulated SFM trajectories')
    parser.add_argument('--grid_size', type=int, default=DEFAULT_GRID_SIZE, help='Number of grid cells per axis')
    parser.add_argument('--pressure_percentile', type=float, default=DEFAULT_PRESSURE_PERCENTILE, help='Percentile for dynamic pressure threshold')
    parser.add_argument('--consecutive_steps', type=int, default=DEFAULT_CONSECUTIVE_STEPS, help='Consecutive steps above threshold for alert')
    parser.add_argument('--output_dir', type=str, default='panic_monitor_frames', help='Directory to save overlays and alerts')
    args = parser.parse_args()
    monitor_panic_simulation(args.traj_csv, grid_size=args.grid_size, pressure_percentile=args.pressure_percentile, consecutive_steps=args.consecutive_steps, output_dir=args.output_dir) 