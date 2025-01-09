import pandas as pd
import numpy as np
from scipy.optimize import linprog
from funct_fin import steer_angle, rolling_friction, motor_force, F_friction_due_to_steering, slip_angles, lateral_tire_forces, friction
from discretization_function import compute_discrete_function_terms_single_step_euler
from continuous_matrix_function import continuous_matrices
# import discretization_function
### --- IMPORT AND LOADING DATA IN 'CONTINUOUS TIME' --- ###

file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_18_2024_14_43_23.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_18_2024_14_52_47.csv'

df = pd.read_csv(file_path)

# Load data
vy = df['velocity y'].values
x = df['vicon x'].values
y = df['vicon y'].values
vx = df['vel encoder'].values
t = df['elapsed time sensors'].values
yaw = df['vicon yaw'].values
w = df['W (IMU)'].values
tau = df['throttle'].values
steering_input = df['steering'].values

# Define dynamic model parameter
l = 0.175
lr = 0.54 * l
lf = l - lr
m = 1.67
m_front_wheel = 0.847
m_rear_wheel = 0.733
Cf = m_front_wheel / m
Cr = m_rear_wheel / m
Jz = 0.006513

## --- INITIAL ITERATION INSTANT --- ###

error_list = []
error_vx = []
error_vy = []
error_w = []
for index in range(3, len(df)):

    F_0_minus2, G_0_minus2 = continuous_matrices(index - 2, steering_input, vx, vy, w, tau)
    F_0_minus1, G_0_minus1 = continuous_matrices(index - 1, steering_input, vx, vy, w, tau)
    
    delta_minus_2 = steer_angle(steering_input[index - 2])
    x_cont_minus_2 = np.array([[vx[index -2]], [vy[index -2]], [w[index -2]]])  # state vector in continuous time
    u_cont_minus_2 = np.array([[tau[index -2]], [delta_minus_2]])

    delta_minus_1 = steer_angle(steering_input[index - 1])
    x_cont_minus_1 = np.array([[vx[index -1]], [vy[index -1]], [w[index -1]]]) 
    u_cont_minus_1 = np.array([[tau[index -1]], [delta_minus_2]])

    autonomous_func_minus_2 = F_0_minus2
    input_func_minus_2 = G_0_minus2
    autonomous_func_minus_1 = F_0_minus1
    input_func_minus_1 = G_0_minus1

    f_dicr_minus_2, g_discr_minus_2, state_discr_minus_2 = compute_discrete_function_terms_single_step_euler(x_cont_minus_2, u_cont_minus_2, autonomous_func_minus_2, input_func_minus_2)
    f_dicr_minus_1, g_discr_minus_1, state_discr_minus_1 = compute_discrete_function_terms_single_step_euler(x_cont_minus_1, u_cont_minus_1, autonomous_func_minus_1, input_func_minus_1)
    
    error = state_discr_minus_1 - x_cont_minus_1
    error_list.append(np.linalg.norm(error))
    error_vx.append(error[0, 0])  # Error fro longitudinal velocity
    error_vy.append(error[1, 0])  # Error for lateral velocity
    error_w.append(error[2, 0])   # Error for angolar velocity

average_error = np.mean(error_list)
print(f"Average error: {average_error}")
average_error_vx = np.mean(error_vx)
average_error_vy = np.mean(error_vy)
average_error_w = np.mean(error_w)

print(f"Average error for vx: {average_error_vx}")
print(f"Average error for vy: {average_error_vy}")
print(f"Average error for w: {average_error_w}")

