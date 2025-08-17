import numpy as np 
import math
import matplotlib.pyplot as plt

class FK():
    def __init__(self):
        d1 = 37.5
        d2 = 140
        d3 = 53
        d4 = 128
        d5 = 57.5
        d6 = 13.5
        a1 = 54

        self.dh_table = [[0, 90, d1, a1],
                         [90, 0, 0, d2],
                         [0, 90, 0, d3],
                         [0, -90, d4, 0],
                         [0, 90, 0, 0],
                         [0, 0, d6, 0]]
        
    def dh_transfrom(self,theta, d, a, alpha): 
        return np.array([[math.cos(theta), -math.sin(theta)*math.cos(alpha), math.sin(theta)*math.sin(alpha), a*math.cos(theta)],
                           [math.sin(theta), math.cos(theta)*math.cos(alpha), -math.cos(theta)*math.sin(alpha), a*math.sin(theta)],
                           [0, math.sin(alpha), math.cos(alpha), d], 
                           [0, 0, 0, 1]])
    
    def trans_matrix(self, desired_angle):
        T = np.eye(4)
        positions = [T[:3, 3].copy()]  # start at base
        transformations = []

        for i, (theta_offset, alpha, d, a) in enumerate(self.dh_table):
            theta = math.radians(desired_angle[i] + theta_offset)
            alpha = math.radians(alpha)
            T_i = self.dh_transfrom(theta, d, a, alpha)
            T = T@T_i
            positions.append(T[:3, 3].copy())
            transformations.append(T_i.copy())
            print(f'Transformation matrix A{i+1}:\n{T_i}\n')
        print(f'Final transformation matrix T:\n{T}')
        return positions, T
    
def plot_robot(positions):
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, '-o', linewidth=2, markersize=6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('6 DOF Robot Arm - FK Visualization')
    ax.set_box_aspect([1, 1, 1])  # equal aspect ratio

    plt.show()
    
    
if __name__ == "__main__":
    fk = FK()
    joint_angles = [0,0,0,0,0,0]
    positions = fk.trans_matrix(joint_angles)
    plot_robot(positions)
