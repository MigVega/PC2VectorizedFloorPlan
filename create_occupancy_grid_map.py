import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def create_occupancy_grid_map_and_save_image(pcd_path, height, grid_resolution, output_image_path):
    # Load the PCD file
    pcd = o3d.io.read_point_cloud(pcd_path)

    # Convert the PCD data to a NumPy array
    point_cloud_data = np.asarray(pcd.points)

    # Convert the PCD color to a NumPy array
    point_cloud_colors = np.asarray(pcd.colors) * 255

    # Define parameters for the occupancy grid map
    grid_size_x = int(np.ceil((np.max(point_cloud_data[:, 0]) - np.min(point_cloud_data[:, 0])) / grid_resolution))
    grid_size_y = int(np.ceil((np.max(point_cloud_data[:, 1]) - np.min(point_cloud_data[:, 1])) / grid_resolution))

    # Create an empty occupancy grid map filled with white
    occupancy_grid = np.full((grid_size_x, grid_size_y, 3), 255, dtype=np.uint8)

    # Convert point cloud data to grid coordinates
    grid_x = np.floor((point_cloud_data[:, 0] - np.min(point_cloud_data[:, 0])) / grid_resolution).astype(int)
    grid_y = np.floor((point_cloud_data[:, 1] - np.min(point_cloud_data[:, 1])) / grid_resolution).astype(int)

    # Get unique rows (unique colors)
    # unique_colors = np.unique(point_cloud_colors, axis=0)
    # print(unique_colors)
    # [[0. 101. 189.] ## blue --> floor
    # [134.  94.  60.] ## brown
    # [153. 153. 153.] ## gray
    # [153. 193. 241.] ## light blue --> windows
    # [162. 173. 0.] ## yellow green --> walls
    # [227. 114.  34.]] ## orange --> columns

    floor = [0., 101., 189.]
    windows = [153., 193., 241.]
    walls = [162., 173., 0.]
    columns = [227., 114.,  34.]

    # Filter data based on its height and color
    mask = (point_cloud_data[:, 2] <= height) & (np.all(point_cloud_colors == walls, axis=1))

    # Use the valid grid coordinates to set the color in the occupancy grid map
    occupancy_grid[grid_x[mask], grid_y[mask]] = walls

    # Save the occupancy grid map as a PNG image
    plt.imsave(output_image_path, occupancy_grid)
    return occupancy_grid

# Example usage for create occupancy grid map for each element:
pcd_path = "../ConSLAM_BIM_semantic_pcd.ply"  # Replace with the path to your PCD file
filtered_height = 13  # Adjust as needed according to each element
grid_resolution = 0.05  # Specify the grid resolution in meters
output_image_path = "../alignment_result/occupancy_grid_walls_1118_13_0.05_aligned.png"  # Specify the output image path
create_occupancy_grid_map_and_save_image(pcd_path, filtered_height, grid_resolution, output_image_path)

# Morphing
img = cv.imread(output_image_path)
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
plt.imsave("../alignment_result/morphing_wall_1118_13_0.05_aligned.png", opening)
#plt.imshow(opening)
#plt.show()
