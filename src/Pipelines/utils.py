import cv2
import open3d as o3d
import numpy as np
import csv
import matplotlib.pyplot as plt

def load_image(path):
    return cv2.imread(path)

def save_point_cloud_ply(points, filename):
    pcd = o3d.geometry.PointCloud()
    arr = np.array([[p['x'], p['y'], p['z']] for p in points], dtype=np.float32)
    pcd.points = o3d.utility.Vector3dVector(arr)
    o3d.io.write_point_cloud(filename, pcd)

def save_point_cloud_csv(points, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['id', 'x', 'y', 'z'])
        writer.writeheader()
        for p in points:
            writer.writerow(p)

def show_point_cloud(points, ground_plane=True):
    arr = np.array([[p['x'], p['y'], p['z']] for p in points], dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    geometries = [pcd]
    if ground_plane and len(points) > 0:
        z0 = float(np.min(arr[:, 2])) - 0.05
        x0, x1 = np.min(arr[:, 0]), np.max(arr[:, 0])
        y0, y1 = np.min(arr[:, 1]), np.max(arr[:, 1])
        pad_x = (x1 - x0) * 0.1
        pad_y = (y1 - y0) * 0.1
        x0, x1 = x0 - pad_x, x1 + pad_x
        y0, y1 = y0 - pad_y, y1 + pad_y
        corners = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0]
        ])
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(corners)
        mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [2, 3, 0]])
        mesh.paint_uniform_color([0.8, 0.8, 0.8])
        mesh.compute_vertex_normals()
        geometries.append(mesh)
    o3d.visualization.draw_geometries(geometries, window_name="3D Crowd Point Cloud")

def save_detected_image(image, boxes, out_path):
    img = image.copy()
    for (x1, y1, x2, y2) in boxes.astype(int):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"Faces: {len(boxes)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imwrite(out_path, img)

# --- Standardized Utilities for Grid, Time, and Plotting ---

# Grid utilities
def get_grid_edges(x_min, x_max, y_min, y_max, grid_size):
    x_edges = np.linspace(x_min, x_max, grid_size + 1)
    y_edges = np.linspace(y_min, y_max, grid_size + 1)
    return x_edges, y_edges

def get_cell_size(x_min, x_max, y_min, y_max, grid_size):
    return (x_max - x_min) / grid_size, (y_max - y_min) / grid_size

# Time utilities
TIME_STEP = 0.1  # seconds per step (adjust as needed)
def step_to_seconds(step):
    return step * TIME_STEP

# Plotting utilities (full view, consistent style)
def plot_density_field(density, x_edges, y_edges, out_path, title='Density Field', cmap='Blues', alpha=0.5):
    plt.figure(figsize=(8,8))
    plt.imshow(density.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], cmap=cmap, alpha=alpha)
    plt.colorbar(label='Density')
    plt.xlabel('x (meters)')
    plt.ylabel('y (meters)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_pressure_field(pressure, x_edges, y_edges, out_path, title='Pressure Field', cmap='hot', alpha=0.7):
    plt.figure(figsize=(8,8))
    plt.imshow(pressure.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], cmap=cmap, alpha=alpha)
    plt.colorbar(label='Pressure')
    plt.xlabel('x (meters)')
    plt.ylabel('y (meters)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close() 