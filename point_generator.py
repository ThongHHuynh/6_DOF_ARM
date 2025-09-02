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

def generate_square(cx=0, cy = 0, z = 0, width = 2.0, height = 1.0, num_points_per_edge = 25):
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
            points.append((x,y,z))
    return np.array(points)

# if __name__ =='__main__':
#     generate_circle()