import numpy as np
import pandas as pd
import argparse
import os
from queue import PriorityQueue
import yaml

def astar(grid, start, goal, cost_map):
    """A* pathfinding on a 2D grid."""
    rows, cols = grid.shape
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: np.linalg.norm(np.array(start) - np.array(goal))}
    while not open_set.empty():
        _, current = open_set.get()
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, g_score[goal]
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            neighbor = (current[0]+dx, current[1]+dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                tentative_g = g_score[current] + cost_map[neighbor]
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + np.linalg.norm(np.array(neighbor) - np.array(goal))
                    open_set.put((f_score[neighbor], neighbor))
    return None, np.inf  # No path found

def main(traj_csv, pressure_alerts_csv, config_yaml, grid_size=30, output_dir='astar_results'):
    os.makedirs(output_dir, exist_ok=True)
    # Load alerts
    alerts = pd.read_csv(pressure_alerts_csv)
    # Load trajectories to get bounds
    df = pd.read_csv(traj_csv)
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    x_edges = np.linspace(x_min, x_max, grid_size+1)
    y_edges = np.linspace(y_min, y_max, grid_size+1)
    # Load candidate goals from config
    with open(config_yaml) as f:
        cfg = yaml.safe_load(f)
    candidate_goals = cfg.get('panic_simulation', {}).get('candidate_goals', [])
    if not candidate_goals:
        # fallback to single goal_x, goal_y
        candidate_goals = [[cfg['panic_simulation']['goal_x'], cfg['panic_simulation']['goal_y']]]
    # Map candidate goals to grid
    grid_goals = [(
        np.searchsorted(x_edges, gx, side='right') - 1,
        np.searchsorted(y_edges, gy, side='right') - 1
    ) for gx, gy in candidate_goals]
    # For each alert, run A*
    results = []
    for _, alert in alerts.iterrows():
        cx, cy = int(alert['cell_x']), int(alert['cell_y'])
        # Build cost map: high cost for alert cells, low cost otherwise
        cost_map = np.ones((grid_size, grid_size))
        for _, a in alerts.iterrows():
            cost_map[int(a['cell_x']), int(a['cell_y'])] = 1000  # High cost for alert cells
        best_goal = None
        best_cost = np.inf
        best_path = None
        best_goal_xy = None
        for (gx, gy), (goal_x, goal_y) in zip(grid_goals, candidate_goals):
            path, cost = astar(cost_map, (cx, cy), (gx, gy), cost_map)
            if cost < best_cost:
                best_cost = cost
                best_path = path
                best_goal = (gx, gy)
                best_goal_xy = (goal_x, goal_y)
        if best_goal is not None and best_goal_xy is not None:
            results.append({
                'start_cell_x': cx,
                'start_cell_y': cy,
                'goal_cell_x': best_goal[0],
                'goal_cell_y': best_goal[1],
                'goal_x': best_goal_xy[0],
                'goal_y': best_goal_xy[1],
                'path_length': len(best_path) if best_path else 0,
                'path_cost': best_cost,
                'path': best_path
            })
        else:
            # No valid path found to any goal
            results.append({
                'start_cell_x': cx,
                'start_cell_y': cy,
                'goal_cell_x': np.nan,
                'goal_cell_y': np.nan,
                'goal_x': np.nan,
                'goal_y': np.nan,
                'path_length': 0,
                'path_cost': np.inf,
                'path': None
            })
    # Save results
    pd.DataFrame([{
        'start_cell_x': r['start_cell_x'],
        'start_cell_y': r['start_cell_y'],
        'goal_cell_x': r['goal_cell_x'],
        'goal_cell_y': r['goal_cell_y'],
        'goal_x': r['goal_x'],
        'goal_y': r['goal_y'],
        'path_length': r['path_length'],
        'path_cost': r['path_cost']
    } for r in results]).to_csv(os.path.join(output_dir, 'astar_paths.csv'), index=False)
    print(f"A* results saved to {output_dir}/astar_paths.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict goal and time using A* on risk/alert map with multiple candidate goals.')
    parser.add_argument('--traj_csv', type=str, required=True, help='Trajectory CSV')
    parser.add_argument('--pressure_alerts_csv', type=str, required=True, help='Pressure alerts CSV')
    parser.add_argument('--config_yaml', type=str, default='pipeline_config.yaml', help='YAML config file with candidate goals')
    parser.add_argument('--grid_size', type=int, default=30, help='Grid size')
    parser.add_argument('--output_dir', type=str, default='astar_results', help='Output directory')
    args = parser.parse_args()
    main(args.traj_csv, args.pressure_alerts_csv, args.config_yaml, args.grid_size, args.output_dir) 