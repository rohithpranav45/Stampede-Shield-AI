import numpy as np
import open3d as o3d
import torch
import matplotlib.pyplot as plt
import os

def generate_3d_gaussian_heatmap(points, grid_size=(100, 100, 100), sigma=3, device='cuda'):
    """
    points: Nx3 numpy array of (x, y, z) positions
    grid_size: (gx, gy, gz) size of the voxel grid
    sigma: standard deviation of the Gaussian
    Returns: 3D numpy array (heatmap)
    """
    if len(points) == 0:
        return np.zeros(grid_size, dtype=np.float32)
    points = torch.tensor(points, dtype=torch.float32, device=device)
    gx, gy, gz = grid_size
    # Create meshgrid
    xs = torch.linspace(0, gx-1, gx, device=device)
    ys = torch.linspace(0, gy-1, gy, device=device)
    zs = torch.linspace(0, gz-1, gz, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing='ij')
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (gx, gy, gz, 3)
    heatmap = torch.zeros((gx, gy, gz), device=device)
    for p in points:
        d2 = ((grid - p)**2).sum(dim=-1)
        g = torch.exp(-d2 / (2 * sigma**2))
        heatmap += g
    return heatmap.cpu().numpy()

def create_colored_pointcloud_from_heatmap(heatmap, threshold=0.05):
    """
    Converts a 3D heatmap to an Open3D PointCloud for visualization.
    Only points above the threshold are shown.
    """
    gx, gy, gz = heatmap.shape
    indices = np.argwhere(heatmap > threshold)
    values = heatmap[heatmap > threshold]
    colors = plt.get_cmap('jet')(values / heatmap.max())
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(indices)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd

def main():
    import time

    # Simulate or load 3D points (replace with your real-time source)
    # Example: 100 random points in a 100x100x100 cube
    np.random.seed(42)
    num_points = 100
    grid_size = (100, 100, 100)
    sigma = 3

    output_dir = "heatmap_frames"
    os.makedirs(output_dir, exist_ok=True)

    # Try to use Open3D's OffscreenRenderer
    try:
        from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord
        renderer_available = True
    except ImportError:
        print("OffscreenRenderer not available in your Open3D version.")
        renderer_available = False

    for t in range(20):  # Save 20 frames as example
        points = np.random.rand(num_points, 3) * np.array(grid_size)
        heatmap = generate_3d_gaussian_heatmap(points, grid_size=grid_size, sigma=sigma, device='cuda')
        heatmap_pcd = create_colored_pointcloud_from_heatmap(heatmap, threshold=0.05)
        # Also add the original points as green
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (num_points, 1)))
        # Combine both
        all_pcd = pcd + heatmap_pcd

        if renderer_available:
            # Render with OffscreenRenderer
            renderer = OffscreenRenderer(640, 480)
            mat = MaterialRecord()
            mat.shader = "defaultUnlit"
            renderer.scene.add_geometry("all_pcd", all_pcd, mat)
            img = renderer.render_to_image()
            o3d.io.write_image(os.path.join(output_dir, f"frame_{t:03d}.png"), img)
            renderer.scene.clear_geometry()
            renderer.release()
        else:
            # Fallback: Save as .ply for later viewing
            o3d.io.write_point_cloud(os.path.join(output_dir, f"frame_{t:03d}.ply"), all_pcd)
        print(f"Saved frame {t:03d}")
        time.sleep(0.05)

if __name__ == '__main__':
    main()