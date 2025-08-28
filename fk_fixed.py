import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Robot parameters
d1 = 0.2
a1 = 0.2
a2 = 0.7
a3 = 0.2
d4 = 0.5
d6 = 0.3

def dh_transform(a, alpha, d, theta):
    """Compute DH transformation matrix"""
    T = np.array([
        [np.cos(theta), -np.sin(theta), 0, a],
        [np.cos(alpha)*np.sin(theta), np.cos(alpha)*np.cos(theta), -np.sin(alpha), -np.sin(alpha)*d],
        [np.sin(alpha)*np.sin(theta), np.sin(alpha)*np.cos(theta), np.cos(alpha), np.cos(alpha)*d],
        [0, 0, 0, 1]
    ])
    return T

def forward_kinematic_6dof(q, r):
    """Forward kinematics computation"""
    T_01 = dh_transform(0, 0, d1, q[0, 0])
    T_12 = dh_transform(a1, np.pi/2, 0, q[1, 0])
    T_23 = dh_transform(a2, 0, 0, q[2, 0])
    T_34 = dh_transform(a3, np.pi/2, d4, q[3, 0])
    T_45 = dh_transform(0, np.pi/2, 0, q[4, 0])
    T_56 = dh_transform(0, np.pi/2, d6, q[5, 0])
    
    T_06 = T_01 @ T_12 @ T_23 @ T_34 @ T_45 @ T_56
    r0 = T_06 @ r
    return r0

def get_joint_positions(q):
    """Get positions of all joints for visualization"""
    # Initialize with base frame
    T_base = np.eye(4)
    
    # Compute cumulative transformations
    T_01 = dh_transform(0, 0, d1, q[0])
    T_02 = T_01 @ dh_transform(a1, np.pi/2, 0, q[1])
    T_03 = T_02 @ dh_transform(a2, 0, 0, q[2])
    T_04 = T_03 @ dh_transform(a3, np.pi/2, d4, q[3])
    T_05 = T_04 @ dh_transform(0, np.pi/2, 0, q[4])
    T_06 = T_05 @ dh_transform(0, np.pi/2, d6, q[5])
    
    # Extract positions (last column, first 3 rows ie XYZ)
    positions = np.array([
        [0, 0, 0],           # Base
        T_01[:3, 3],         # Joint 1
        T_02[:3, 3],         # Joint 2
        T_03[:3, 3],         # Joint 3
        T_04[:3, 3],         # Joint 4
        T_05[:3, 3],         # Joint 5
        T_06[:3, 3]          # End effector
    ])
    
    return positions, [T_base, T_01, T_02, T_03, T_04, T_05, T_06]

def plot_frame(ax, T, scale=0.1, alpha=0.8):
    """Plot coordinate frame axes"""
    #extract x,y,z position
    origin = T[:3, 3]
    #extract x,y,z vector
    x_axis = T[:3, 0] * scale
    y_axis = T[:3, 1] * scale
    z_axis = T[:3, 2] * scale
    
    # Plot axes
    ax.quiver(origin[0], origin[1], origin[2], 
              x_axis[0], x_axis[1], x_axis[2], 
              color='red', alpha=alpha, arrow_length_ratio=0.1)
    ax.quiver(origin[0], origin[1], origin[2], 
              y_axis[0], y_axis[1], y_axis[2], 
              color='green', alpha=alpha, arrow_length_ratio=0.1)
    ax.quiver(origin[0], origin[1], origin[2], 
              z_axis[0], z_axis[1], z_axis[2], 
              color='blue', alpha=alpha, arrow_length_ratio=0.1)

def visualize_robot(q, show_frames=True, show_end_effector_point=True):
    """Visualize the 6-DOF robot arm"""
    fig = plt.figure(figsize=(12, 8))
    #create canvas ax to plot on
    ax = fig.add_subplot(111, projection='3d')
    
    # Get joint positions and transformations (pass in initial pose)
    positions, transforms = get_joint_positions(q)
    
    # Plot robot links
    for i in range(len(positions)-1):
        #x,y,z from current joint to next joint
        ax.plot3D([positions[i, 0], positions[i+1, 0]], 
                  [positions[i, 1], positions[i+1, 1]], 
                  [positions[i, 2], positions[i+1, 2]], 
                  'b', linewidth=2, alpha=0.8)
    
    # Plot joints
    #pos[:,0]: all row of column 0 ie all x
    #pos[:,1]: all col of column 0 ie all y,....

    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],     # # Show coordinate frames
    # if show_frames:
    #     for i, T in enumerate(transforms):
    #         plot_frame(ax, T, scale=0.08, alpha=0.6)
               c='red', s=100, alpha=0.8)
    
    # Label joints
    labels = ['Base', 'J1', 'J2', 'J3', 'J4', 'J5', 'EE']
    #zip and enumerate position turn it to 1 [position] Base, 2 [position2 J1]
    for i, (pos, label) in enumerate(zip(positions, labels)):
        #at text at the position of the joint
        ax.text(pos[0], pos[1], pos[2], f'  {label}', fontsize=10)
    
    # # Show coordinate frames
    # if show_frames:
    #     for i, T in enumerate(transforms):
    #         plot_frame(ax, T, scale=0.08, alpha=0.6)
    
    # Show end effector point calculation
    if show_end_effector_point:
        r = np.array([[0.0], [0.0], [0.0], [1.0]])
        r0 = forward_kinematic_6dof(q.reshape(-1, 1), r)
        #r0 is a (4,1) matrix - > take xyz is the col 0, 3 rows
        ax.scatter([r0[0, 0]], [r0[1, 0]], [r0[2, 0]], 
                  c='yellow', s=150, marker='x', 
                  label=f'FK Result: ({r0[0,0]:.3f}, {r0[1,0]:.3f}, {r0[2,0]:.3f})')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('6-DOF Robot Arm Visualization')
    
    # Set equal aspect ratio
    max_range = 1.5
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, max_range*2])
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    
    # Your original configuration
    q = np.array([0.0, np.pi/2, np.pi/2, 0.0, np.pi, 0.0])
    n = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    k = q+n
    
    # Single robot visualization
    print("\nCreating single robot visualization...")
    fig1, ax1 = visualize_robot(k, show_frames=True, show_end_effector_point=True)
    
    # Verify FK calculation
    r = np.array([[0.0], [0.0], [0.0], [1.0]])
    r0 = forward_kinematic_6dof(k.reshape(-1, 1), r)
    print(f"\nForward Kinematics Result:")
    print(f"End effector position: [{r0[0,0]:.3f}, {r0[1,0]:.3f}, {r0[2,0]:.3f}]")
    
    plt.show()