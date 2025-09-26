import serial
import time
import ik_rl
import numpy as np
import matplotlib.pyplot as plt
import struct

PORT = "COM5"   # change to your ESP32 port
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

def angle_publisher():
    IK = ik_rl.InverseKinematics6DOF()

    start_pos = np.array([0.2, -0.4, 0.3])  # Scale for visibility
    end_pos = np.array([0.2, 0.4, 0.3])  # Scale for visibility
    target_rot = ik_rl.create_rotation_matrix_xyz(0, 1.57, 0)
    num_steps = 50
    all_angles = []

    trajectory = np.linspace(start_pos, end_pos, num_steps)
    print(f"Moving from {start_pos} to {end_pos} in {num_steps} steps...")
    # Create a single figure for animation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    prev_angles = None
    ee_pos = []
    angle1, angle2, angle3, angle4, angle5, angle6 = 0, 0, 0, 0, 0, 0


    for i, pos in enumerate(trajectory):
        if prev_angles is not None:
            result = IK.solve_ik(pos, target_rotation=target_rot, initial_guess=prev_angles)
        else:
            result = IK.solve_ik(pos, target_rotation=target_rot, initial_guess=None)
        if result['success']:
            prev_angles = result['joint_angles']
            ee_pos.append(result['final_position'])
            
            ee_traj = np.array(ee_pos)

            #Special offset for joint angles
            np.set_printoptions(suppress=True, precision=3)
            result['joint_angles_deg'] = np.round(result['joint_angles_deg'], 3)
            print(f"Joint angles (deg): {result['joint_angles_deg'][0]}, {result['joint_angles_deg'][1]}, {result['joint_angles_deg'][2]-(63)}, {result['joint_angles_deg'][3]}, {result['joint_angles_deg'][4]}, {result['joint_angles_deg'][5]}")
            angle1 = result['joint_angles_deg'][0] +135
            angle2 = result['joint_angles_deg'][1] - 90 + 135
            angle3 = result['joint_angles_deg'][2] - 70 +90 + 135 # Offset for joint 3
            angle4 = result['joint_angles_deg'][3] + 90 + 135
            angle5 = result['joint_angles_deg'][4] 
            angle6 = result['joint_angles_deg'][5]
            all_angles.append([angle1, angle2, angle3, angle4, angle5, angle6])
            # cmd = f"{angle1},{angle2},{angle3},{angle4},{angle5},{angle6}\n"
            # ser.write(cmd.encode())
            # print("Sent:", cmd.strip())
            # time.sleep(0.1)

            # Build one big packet: header, number of steps, all angles, footer
            num_steps = len(all_angles)
            packet = struct.pack('<BH', 0xAA, num_steps)  # header + num_steps (unsigned short)
            for step in all_angles:
                packet += struct.pack('<6f', *step)
            packet += struct.pack('<B', 0x55)

            ser.write(packet)
            print(f"Sent {num_steps} steps in one packet")

            #pack the joint angles as binary data
            # packet = struct.pack('<B6fB', 0xAA, float(angle1), float(angle2),
            #                     float(angle3), float(angle4), float(angle5), float(angle6), 0x55)
            # ser.write(packet)
            # print("Packet sent")



            #Draw the robot
            ax.clear()
            ik_rl.visualize_robot(ax,result['joint_angles'], title=f"Step {i}: Position {pos}")
            #Draw the pre-defined path
            ax.plot3D(trajectory[:,0], trajectory[:,1], trajectory[:,2], 'g', linewidth = 2, alpha= 0.8)
            #Draw the end-effector trajectory
            ax.plot3D(ee_traj[:,0], ee_traj[:,1], ee_traj[:,2], 'r', linewidth = 2, alpha= 0.8)

            print(f"Current position: [{result['final_position'][0]:.3f}, {result['final_position'][1]:.3f}, {result['final_position'][2]:.3f}]")

            #special setting for joing angles
            np.set_printoptions(suppress=True, precision=3)
            result['joint_angles_deg'] = np.round(result['joint_angles_deg'], 3)
            print(f"Joint angles (deg): {result['joint_angles_deg'][0]}, {result['joint_angles_deg'][1]}, {result['joint_angles_deg'][2]-(63)}, {result['joint_angles_deg'][3]}, {result['joint_angles_deg'][4]}, {result['joint_angles_deg'][5]}")
            plt.pause(0.1)  # Small delay for animation effect
        
        else: 
            print("IK solution not found.")
            break
    
    print("Goal reached.")
    print(f"Position error: {result['position_error']:.3f}")
    print(f'Final position: [{result["final_position"][0]:.3f}, {result["final_position"][1]:.3f}, {result["final_position"][2]:.3f}]')



# Example movement sequence
#angles = [135,45,135,225,135]
# angles = [90,0,90,180,90]

# for angle in angles:
#     cmd = f"{angle}\n"
#     ser.write(cmd.encode())
#     print("Sent:", angle)
#     time.sleep(1)  # 1 second between moves
if __name__ == "__main__":
    angle_publisher()
    ser.close()

