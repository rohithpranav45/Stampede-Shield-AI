import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from utils import TIME_STEP  # Use standardized time step for consistency

# Social Force Model parameters
NORMAL_SPEED = 0.8
PANIC_SPEED =3
PANIC_RANDOMNESS = 1.0
PANIC_REPULSION = 3.0
PANIC_RADIUS = 30.0  # Distance for panic propagation
# GOAL will be set dynamically below
DT = 0.5

np.random.seed(42)

def get_goal_from_config(config_path='pipeline_config.yaml'):
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        goal_x = cfg.get('panic_simulation', {}).get('goal_x', 0)
        goal_y = cfg.get('panic_simulation', {}).get('goal_y', 0)
        return np.array([goal_x, goal_y])
    except Exception as e:
        print(f"Could not read goal from config: {e}. Using default [0, 0].")
        return np.array([0, 0])

GOAL = get_goal_from_config()

def social_force(agent, agents, panic=False):
    # Goal force: move toward the goal
    goal_vec = GOAL - np.array([agent['x'], agent['y']])
    if np.linalg.norm(goal_vec) > 0:
        goal_force = goal_vec / np.linalg.norm(goal_vec)
    else:
        goal_force = np.zeros(2)
    speed = PANIC_SPEED if panic else NORMAL_SPEED
    # Repulsion from other agents
    repulsion = np.zeros(2)
    for other in agents:
        if other['id'] == agent['id']:
            continue
        diff = np.array([agent['x'], agent['y']]) - np.array([other['x'], other['y']])
        dist = np.linalg.norm(diff)
        if dist < 1e-3:
            continue
        repulsion += diff / (dist**2)
    repulsion *= PANIC_REPULSION if panic else 1.0
    # Randomness in panic
    random_force = np.random.randn(2) * PANIC_RANDOMNESS if panic else np.zeros(2)
    # Total force
    total_force = goal_force * speed + repulsion + random_force
    return total_force

def simulate_panic_sfm(init_csv, steps=30, panic_trigger_step=10, output_csv='panic_sfm_trajectories.csv'):
    agents = pd.read_csv(init_csv).to_dict('records')
    # Store trajectories for visualization
    all_traj = {agent['id']: [(agent['x'], agent['y'], agent['panic'])] for agent in agents}
    for t in range(steps):
        # Trigger panic at the specified step for agents in a region (e.g., right half)
        if t == panic_trigger_step:
            for agent in agents:
                if agent['x'] > 0.3 * max(a['x'] for a in agents):  # e.g., right 70%
                    agent['panic'] = True
        # Propagate panic to neighbors
        for agent in agents:
            if agent['panic']:
                for other in agents:
                    if not other['panic']:
                        dist = np.linalg.norm([agent['x'] - other['x'], agent['y'] - other['y']])
                        if dist < PANIC_RADIUS:
                            other['panic'] = True
        # Update positions
        new_states = []
        for agent in agents:
            force = social_force(agent, agents, panic=agent['panic'])
            agent['x'] += force[0] * DT
            agent['y'] += force[1] * DT
            new_states.append(agent.copy())
            all_traj[agent['id']].append((agent['x'], agent['y'], agent['panic']))
        agents = new_states
    # Save trajectories
    rows = []
    for aid, traj in all_traj.items():
        for t, (x, y, panic) in enumerate(traj):
            rows.append({'id': aid, 'step': t, 'x': x, 'y': y, 'panic': panic})
    df_traj = pd.DataFrame(rows)
    df_traj.to_csv(output_csv, index=False)
    print(f"Panic SFM simulation saved to {output_csv}")
    return df_traj

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate panic situations using Social Force Model (SFM).')
    parser.add_argument('--init_csv', type=str, default='panic_agents_init.csv', help='CSV of initial agent states')
    parser.add_argument('--steps', type=int, default=30, help='Number of simulation steps')
    parser.add_argument('--panic_trigger_step', type=int, default=10, help='Step at which to trigger panic')
    parser.add_argument('--output_csv', type=str, default='panic_sfm_trajectories.csv', help='Output CSV for simulated trajectories')
    args = parser.parse_args()
    simulate_panic_sfm(args.init_csv, steps=args.steps, panic_trigger_step=args.panic_trigger_step, output_csv=args.output_csv)

# All positions, distances, and time steps are standardized across the pipeline.
# If grid-based logic is added, use get_grid_edges from utils.py for consistency. 