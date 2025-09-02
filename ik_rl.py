import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
import point_generator as points
warnings.filterwarnings('ignore')

# Robot parameters (from your FK code)
d1 = 0.2
a1 = 0.2
a2 = 0.7
a3 = 0.2
d4 = 0.5
d6 = 0.3

def dh_transform(a, alpha, d, theta):
    """Compute DH transformation matrix (same as your implementation)"""
    T = np.array([
        [np.cos(theta), -np.sin(theta), 0, a],
        [np.cos(alpha)*np.sin(theta), np.cos(alpha)*np.cos(theta), -np.sin(alpha), -np.sin(alpha)*d],
        [np.sin(alpha)*np.sin(theta), np.sin(alpha)*np.cos(theta), np.cos(alpha), np.cos(alpha)*d],
        [0, 0, 0, 1]
    ])
    return T

def forward_kinematic_6dof(q):
    """Forward kinematics computation (adapted from your code)"""
    if q.ndim == 1:
        q = q.reshape(-1, 1)
    
    T_01 = dh_transform(0, 0, d1, q[0, 0])
    T_12 = dh_transform(a1, np.pi/2, 0, q[1, 0])
    T_23 = dh_transform(a2, 0, 0, q[2, 0])
    T_34 = dh_transform(a3, np.pi/2, d4, q[3, 0])
    T_45 = dh_transform(0, np.pi/2, 0, q[4, 0])
    T_56 = dh_transform(0, np.pi/2, d6, q[5, 0])
    
    T_06 = T_01 @ T_12 @ T_23 @ T_34 @ T_45 @ T_56
    return T_06

def get_joint_positions(q):
    """Get positions of all joints for visualization (from your code)"""
    T_base = np.eye(4)
    
    T_01 = dh_transform(0, 0, d1, q[0])
    T_02 = T_01 @ dh_transform(a1, np.pi/2, 0, q[1])
    T_03 = T_02 @ dh_transform(a2, 0, 0, q[2])
    T_04 = T_03 @ dh_transform(a3, np.pi/2, d4, q[3])
    T_05 = T_04 @ dh_transform(0, np.pi/2, 0, q[4])
    T_06 = T_05 @ dh_transform(0, np.pi/2, d6, q[5])
    
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

def visualize_robot(ax,q, show_frames=True, show_end_effector_point=True, title="Robot Arm"):
    """Visualize the 6-DOF robot arm (from your code with minor adaptations)"""
    
    positions, transforms = get_joint_positions(q)
    
    # Plot robot links
    for i in range(len(positions)-1):
        ax.plot3D([positions[i, 0], positions[i+1, 0]], 
                  [positions[i, 1], positions[i+1, 1]], 
                  [positions[i, 2], positions[i+1, 2]], 
                  'b', linewidth=3, alpha=0.8)
    
    # Plot joints
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
               c='red', s=100, alpha=0.8)
    
    # Label joints
    labels = ['Base', 'J1', 'J2', 'J3', 'J4', 'J5', 'EE']
    for i, (pos, label) in enumerate(zip(positions, labels)):
        ax.text(pos[0], pos[1], pos[2], f'  {label}', fontsize=10)
    
    # Show end effector point calculation
    if show_end_effector_point:
        T_06 = forward_kinematic_6dof(q)
        ee_pos = T_06[:3, 3]
        ax.scatter([ee_pos[0]], [ee_pos[1]], [ee_pos[2]], 
                  c='yellow', s=150, marker='x', 
                  label=f'EE: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = 1.5
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, max_range*2])
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax

class InverseKinematics6DOF:
    def __init__(self, joint_limits=None):
        """Initialize IK solver"""
        # Default joint limits 
        if joint_limits is None:
            self.joint_limits = [
                [-1*np.pi, 1*np.pi],        # Joint 1 (base rotation)
                [0*np.pi, 1*np.pi],    # Joint 2 
                [-1*np.pi, 1*np.pi],        # Joint 3
                [-1*np.pi, 1*np.pi],        # Joint 4
                [-1*np.pi, 1*np.pi],        # Joint 5
                [-1*np.pi, 1*np.pi]         # Joint 6
            ]
        else:
            self.joint_limits = joint_limits
    
    def pose_error(self, current_T, target_T):
        """Calculate pose error between current and target"""
        # Position error (3 row of 3rd column -> xyz) -> array
        pos_error = target_T[:3, 3] - current_T[:3, 3]
        
        # Orientation error using rotation matrices
        #extract rotation matrix, 3 row 3 col from top left
        R_current = current_T[:3, :3]
        R_target = target_T[:3, :3]
        #Transpose to rotate current position to world frame, then multiply to result frame for the rotation matrix needed
        R_error = R_target @ R_current.T
        
        # Convert to axis-angle representation for orientation error
        trace = np.trace(R_error)
        trace = np.clip(trace, -1.0, 3.0)  # Numerical stability
        
        #rotation angle in 3d space, not x y z
        angle = np.arccos((trace - 1) / 2)
        
        #if orientation error ~0 -> 0
        if abs(angle) < 1e-6:
            ori_error = np.array([0, 0, 0])
        else:
            axis = np.array([
                R_error[2, 1] - R_error[1, 2],
                R_error[0, 2] - R_error[2, 0],
                R_error[1, 0] - R_error[0, 1]
            ]) / (2 * np.sin(angle))
            ori_error = angle * axis
        
        #merge 2 array to a 1 row 
        return np.concatenate([pos_error, ori_error])
    
    def objective_function(self, joint_angles, target_T, weights=None):
        """Objective function for optimization"""
        if weights is None:
            weights = [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]  # Position vs orientation weights
        
        current_T = forward_kinematic_6dof(joint_angles)
        error = self.pose_error(current_T, target_T)
        weighted_error = error * weights
        
        #square to make sure positive and more penalty for bad result
        return np.sum(weighted_error**2)
    
    def solve_ik(self, target_position, target_rotation=None, 
                 initial_guess=None, method='SLSQP', weights=None, 
                 position_only=False):
        """
        Solve inverse kinematics
        
        Args:
            target_position: [x, y, z] target position
            target_rotation: 3x3 rotation matrix (if None, only position is considered)
            initial_guess: Initial joint angles
            method: Optimization method
            weights: Error weights [x, y, z, rx, ry, rz]
            position_only: If True, only solve for position (ignore orientation)
        
        Returns:
            dict with solution results
        """
        # Create target transformation matrix
        target_T = np.eye(4)
        target_T[:3, 3] = target_position #FILL IN LAST COL FOR ROTATION
        
        if target_rotation is not None:
            target_T[:3, :3] = target_rotation #FILL IN TOP LEFT ROTATION MATRIX
        
        # Initial guess
        if initial_guess is None:
            #default a 6 element 0 array
            initial_guess = np.zeros(6)
        
        # Adjust weights for position-only solving
        if position_only and weights is None:
            weights = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]  # Zero weight on orientation
        
        # Set up constraints (joint limits)
        bounds = [(self.joint_limits[i][0], self.joint_limits[i][1]) #CREATE 6 tuples with 2 element each
                 for i in range(6)]
        
        # Solve optimization
        result = minimize(
            self.objective_function, #function to optimize, pass in initial joint, calculate ee position, far from desired pose -> tweak each joint till reach
            initial_guess, #starting point to tweak
            args=(target_T, weights), #argument used to pass in the objective function
            method=method, #optimization algorithm
            bounds=bounds, #bounds
            options={'maxiter': 1000, 'ftol': 1e-9} #solver setting, 1000 epochs, stop early if tolerance is low
        )
        
        # Calculate final error
        final_T = forward_kinematic_6dof(result.x) #pass in the result for each joint, calculate fk
        final_error = self.pose_error(final_T, target_T)#compare wanted pose to result
        
        # Extract final position and rotation from result rotation matrix
        final_position = final_T[:3, 3] #3 row last col
        final_rotation = final_T[:3, :3] #rotation matrix
        
        return {
            'success': result.success,
            'joint_angles': result.x,
            'joint_angles_deg': np.degrees(result.x),
            'final_position': final_position,
            'final_rotation': final_rotation,
            'position_error': np.linalg.norm(final_error[:3]), #slice the first 3 elements, convert to scalar
            'orientation_error': np.linalg.norm(final_error[3:]), #slice last 3 elements
            'total_error': np.linalg.norm(final_error),
            'iterations': result.nit if hasattr(result, 'nit') else None,
            'message': result.message if hasattr(result, 'message') else None
        }
    
    def solve_multiple_solutions(self, target_position, target_rotation=None, 
                                num_attempts=5, position_only=False):
        """
        Try to find multiple IK solutions by using different initial guesses
        """
        solutions = []
        
        # Generate different initial guesses
        initial_guesses = []
        
        # Add zero configuration
        initial_guesses.append(np.zeros(6))
        
        # Add random configurations
        for _ in range(num_attempts - 1):
            guess = np.random.uniform(-np.pi/2, np.pi/2, 6)
            initial_guesses.append(guess)
        
        # Try each initial guess
        for i, guess in enumerate(initial_guesses):
            result = self.solve_ik(target_position, target_rotation, 
                                 initial_guess=guess, position_only=position_only)
            
            if result['success'] and result['total_error'] < 0.01:  # Good solution
                # Check if this solution is unique (not too close to existing ones)
                is_unique = True
                for existing in solutions:
                    angle_diff = np.linalg.norm(result['joint_angles'] - existing['joint_angles'])
                    if angle_diff < 0.1:  # Solutions are too similar
                        is_unique = False
                        break
                
                if is_unique:
                    result['solution_id'] = len(solutions) + 1
                    solutions.append(result)
        
        return solutions

def create_rotation_matrix_xyz(roll, pitch, yaw):
    """Create rotation matrix from Euler angles (XYZ convention)"""
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    
    R = np.array([
        [cy*cp, -sy*cr + cy*sp*sr, sy*sr + cy*sp*cr],
        [sy*cp, cy*cr + sy*sp*sr, -cy*sr + sy*sp*cr],
        [-sp, cp*sr, cp*cr]
    ])
    
    return R

def test_ik_solver():
    """Test the IK solver with a simple trajectory in real-time"""
    print("=== Real-time Trajectory Test ===")
    
    ik_solver = InverseKinematics6DOF()
    
    # Define trajectory
    start_pos = np.array([0, 0.5, 1.5])
    end_pos = np.array([0, -0.5, 1])
    target_rot = create_rotation_matrix_xyz(0, 3.14, 0)
    num_steps = 100
    
    trajectory = np.linspace(start_pos, end_pos, num_steps)
    
    print(f"Moving from {start_pos} to {end_pos} in {num_steps} steps...")
    # Create a single figure for animation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    #plt.ion()  # Turn on interactive mode

    prev_joint_angle = None
    ee_pos =[]
    circle_traj = points.generate_circle(0.5, 0.5, 0.5, 0.8, 500)
    square_traj = points.generate_square(0.1, -0.5, 1,1, 1, 50)

    for i, pos in enumerate(square_traj):
        if prev_joint_angle is not None: 
            result = ik_solver.solve_ik(pos, target_rot, initial_guess=prev_joint_angle)
        else: 
            result = ik_solver.solve_ik(pos, target_rot)
        
        #debug flag
        if not result['success']:
            print(f'Step {i}: Failed to reach pose {pos}')
            continue

        #take previous joint angle 
        prev_joint_angle = result['joint_angles']

        # Clear previous plot
        ax.clear()
        # Show robot position in real-time
        visualize_robot(ax,result['joint_angles'], title=f"Step {i}: Position {pos}")

        #append ee position
        ee_pos.append(result['final_position'])

        traj = np.array(ee_pos)

        ax.plot3D(traj[:,0], traj[:,1], traj[:,2], 'r', linewidth = 2, alpha= 0.8)
        #ax.plot3D(circle_traj[:,0], circle_traj[:,1], circle_traj[:,2], 'g', linewidth = 2, alpha= 0.8)
        #ax.plot3D(square_traj[:,0], square_traj[:,1], square_traj[:,2], 'g', linewidth = 2, alpha= 0.8)

        plt.pause(0.1)  # Small delay for animation effect

    plt.show()  # Keep final plot open
    print("Trajectory complete!")

if __name__ == "__main__":
    test_ik_solver()