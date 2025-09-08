# class PointGenerator:
#     def __init__(self):
#         pass
#     def generate_circle(radius=1, cente
'''
to generate 
'''

import numpy as np 
import matplotlib.pyplot as plt

def generate_circle(cx = 0, cy = 0, z=0, radius = 0.5, num_points=50):
    points = []
    for theta in np.linspace(0, 2*np.pi, num_points):
        x = cx + radius*np.cos(theta)
        y = cy + radius*np.sin(theta)
        points.append((x,y,z))
    # fig = plt.figure(figsize=(10,8))
    # ax = fig.add_subplot(111, projection='3d')
    points_array = np.array(points)
    # ax.plot3D(points_array[:,0], points_array[:,1], points_array[:,2], 'r', linewidth = 2, alpha = 0.8)
    # plt.draw()
    # plt.show()
    return points_array

def generate_square(cx=0, cy = 0, z = 0, width = 2.0, height = 1.0, num_points_per_edge = 25, roll =0, pitch = 0, yaw = 0):
    points = []
    x1, y1 = cx - width/2, cy - height/2
    x2, y2 = cx + width/2, cy - height/2
    x3, y3 = cx + width/2, cy + height/2
    x4, y4 = cx - width/2, cy + height/2
    for (xa, ya), (xb,yb) in [((x1,y1), (x2, y2)),
                              ((x2,y2), (x3,y3)),
                              ((x3,y3), (x4,y4)),
                              ((x4,y4), (x1,y1))]:
        for _ in np.linspace(0, 1, num_points_per_edge):
            x = xa + _*(xb-xa)
            y = ya + _*(yb-ya)
            rotated = rot_matrix(x,y,z,roll,pitch,yaw)
            x = rotated[0]
            y = rotated[1]
            z = rotated[2]
            points.append((x,y,z))
    return np.array(points)
def rot_matrix(x,y,z, roll, pitch, yaw):
    rx = np.array([[1,0,0],
                  [0, np.cos(roll), -np.sin(roll)],
                  [0, np.sin(roll), np.cos(roll)]])
    
    ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0,0,1]])
    R = rz @ ry @ rx
    return R @ np.array([x,y,z])
    

# Test function to visualize both shapes
def test_shapes():
    # Generate a circle
    circle_points = generate_circle(cx=0, cy=0, z=0, radius=1, num_points=100)
    
    # Generate a square with some rotation
    square_points = generate_square(cx=2, cy=0, z=0, width=1.5, height=1.5, 
                                  num_points_per_edge=25, roll=0, pitch=np.pi/2, yaw=0)
    
    # Plot both shapes
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot circle
    ax.plot3D(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], 
              'b-', linewidth=2, alpha=0.8, label='Circle')
    
    # Plot square
    ax.plot3D(square_points[:, 0], square_points[:, 1], square_points[:, 2], 
              'r-', linewidth=2, alpha=0.8, label='Rotated Square')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Circle and Rotated Square')
    
    plt.show()

if __name__ == '__main__':
    test_shapes()