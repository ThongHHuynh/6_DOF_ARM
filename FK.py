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
                         [0, 0, 0, d2],                          
                         [0, 90, 0, d3],                          
                         [0, -90, d4, 0],                          
                         [0, 90, 0, 0],                          
                         [0, 0, d6, 0]]              
    
    def dh_transfrom(self, theta, d, a, alpha):          
        return np.array([[math.cos(theta), -math.sin(theta)*math.cos(alpha), math.sin(theta)*math.sin(alpha), a*math.cos(theta)],                            
                         [math.sin(theta), math.cos(theta)*math.cos(alpha), -math.cos(theta)*math.sin(alpha), a*math.sin(theta)],                            
                         [0, math.sin(alpha), math.cos(alpha), d],                             
                         [0, 0, 0, 1]])          
    
    def trans_matrix(self, desired_angle, verbose=False):         
        T = np.eye(4)         
        positions = [T[:3, 3].copy()]  # start at base         
        transformations = []
        
        for i, (theta_offset, alpha, d, a) in enumerate(self.dh_table):             
            theta = math.radians(desired_angle[i] + theta_offset)             
            alpha = math.radians(alpha)             
            T_i = self.dh_transfrom(theta, d, a, alpha)             
            T = T @ T_i             
            positions.append(T[:3, 3].copy())
            transformations.append(T_i.copy())
            if verbose:
                print(f'Transformation matrix A{i+1}:\n{T_i}\n')         
        
        if verbose:
            print(f'Final transformation matrix T:\n{T}')         
        return positions, T
    
    def get_end_effector_pose(self, joint_angles):
        """Get end effector position and orientation"""
        _, T = self.trans_matrix(joint_angles)
        position = T[:3, 3]
        rotation = T[:3, :3]
        return position, rotation
    
    def compute_jacobian(self, joint_angles, delta=1e-6):
        """Compute numerical Jacobian matrix"""
        # Get current end effector position
        current_pos, _ = self.get_end_effector_pose(joint_angles)
        
        jacobian = np.zeros((3, 6))  # 3D position, 6 joints
        
        for i in range(6):
            # Perturb joint i
            perturbed_angles = joint_angles.copy()
            perturbed_angles[i] += delta
            
            # Get new position
            new_pos, _ = self.get_end_effector_pose(perturbed_angles)
            
            # Compute partial derivative
            jacobian[:, i] = (new_pos - current_pos) / delta
            
        return jacobian
    
    def inverse_kinematics(self, target_position, initial_guess=None, max_iterations=1000, tolerance=1e-3):
        """
        Solve inverse kinematics using Jacobian-based method
        
        Args:
            target_position: [x, y, z] target position
            initial_guess: initial joint angles (if None, uses zeros)
            max_iterations: maximum number of iterations
            tolerance: position error tolerance
            
        Returns:
            joint_angles: solution joint angles
            success: whether solution was found
        """
        if initial_guess is None:
            joint_angles = np.zeros(6)
        else:
            joint_angles = np.array(initial_guess)
            
        target_pos = np.array(target_position)
        
        for iteration in range(max_iterations):
            # Get current end effector position
            current_pos, _ = self.get_end_effector_pose(joint_angles)
            
            # Compute position error
            error = target_pos - current_pos
            error_magnitude = np.linalg.norm(error)
            
            # Check convergence
            if error_magnitude < tolerance:
                print(f"IK converged in {iteration} iterations")
                print(f"Final error: {error_magnitude:.6f}")
                return joint_angles, True
            
            # Compute Jacobian
            J = self.compute_jacobian(joint_angles)
            
            # Compute pseudo-inverse of Jacobian
            try:
                J_pinv = np.linalg.pinv(J)
            except np.linalg.LinAlgError:
                print("Jacobian is singular")
                return joint_angles, False
            
            # Update joint angles
            delta_angles = J_pinv @ error
            
            # Apply step size damping
            step_size = 0.1
            joint_angles += step_size * delta_angles
            
            # Optional: Add joint limits here
            # joint_angles = np.clip(joint_angles, -np.pi, np.pi)
            
        print(f"IK did not converge after {max_iterations} iterations")
        print(f"Final error: {np.linalg.norm(target_pos - self.get_end_effector_pose(joint_angles)[0]):.6f}")
        return joint_angles, False

def plot_robot(positions, target_pos=None):     
    xs = [p[0] for p in positions]     
    ys = [p[1] for p in positions]     
    zs = [p[2] for p in positions]      
    
    fig = plt.figure(figsize=(10, 8))     
    ax = fig.add_subplot(111, projection='3d')     
    ax.plot(xs, ys, zs, '-o', linewidth=2, markersize=6, label='Robot Arm')
    
    # Plot target position if provided
    if target_pos is not None:
        ax.scatter(*target_pos, color='red', s=100, label='Target Position')
        
    # Plot end effector
    ax.scatter(xs[-1], ys[-1], zs[-1], color='green', s=100, label='End Effector')
    
    ax.set_xlabel('X')     
    ax.set_ylabel('Y')     
    ax.set_zlabel('Z')     
    ax.set_title('6 DOF Robot Arm - FK/IK Visualization')     
    ax.set_box_aspect([1, 1, 1])  # equal aspect ratio
    ax.legend()
    
    plt.show()           

if __name__ == "__main__":     
    fk = FK()
    
    # Test Forward Kinematics
    print("=== Forward Kinematics Test ===")
    joint_angles_fk = [0, 45, 0, 0, 0, 0]     
    positions_fk, _ = fk.trans_matrix(joint_angles_fk, verbose=True)
    end_pos_fk, _ = fk.get_end_effector_pose(joint_angles_fk)
    print(f"End effector position: {end_pos_fk}")
    
    # Test Inverse Kinematics
    print("\n=== Inverse Kinematics Test ===")
    target_position = [0, 0, 0]  # Target position
    print(f"Target position: {target_position}")
    
    # Solve IK
    ik_solution, success = fk.inverse_kinematics(target_position)
    
    if success:
        print(f"IK Solution (radians): {ik_solution}")
        print(f"IK Solution (degrees): {np.degrees(ik_solution)}")
        
        # Verify the solution
        positions_ik, _ = fk.trans_matrix(ik_solution)
        achieved_pos, _ = fk.get_end_effector_pose(ik_solution)
        print(f"Achieved position: {achieved_pos}")
        print(f"Position error: {np.linalg.norm(achieved_pos - target_position):.6f}")
        plot_robot(positions_fk)
        # Plot both FK and IK results
        # plot_robot(positions_ik, target_position)
    else:
        print("IK solution not found")
        # Still plot the FK result
        # plot_robot(positions_fk)