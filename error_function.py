import pandas as pd
import numpy as np
from scipy.optimize import linprog
from funct_fin import steer_angle, rolling_friction, motor_force, F_friction_due_to_steering, slip_angles, lateral_tire_forces, friction
from discretization_function import compute_discrete_function_terms_single_step_euler
from continuous_matrix_function import continuous_matrices
from sympy import symbols, Matrix,And, solve, reduce_inequalities, simplify


### --- IMPORT AND LOADING DATA IN 'CONTINUOUS TIME' --- ###

#

# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_18_2024_14_43_23.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_18_2024_14_52_47.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_10_2025_14_35_50.csv'  # noise_up = 0.05
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_10_2025_14_39_57.csv'  # noise_up = 0.5
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_10_2025_14_50_47.csv'  # noise_up = 1
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_10_2025_17_07_52.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_10_2025_17_12_50.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_16_41_15.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_16_58_58.csv' # unifrom noise
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_17_04_25.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_17_10_42.csv' # uniform nosi not centered
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_17_13_40.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_16_2024_16_59_04.csv'
file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_14_2025_18_20_50.csv'  # NO NOISE 


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

# Parameters needed for SME evaluation
n_state = 3                       # number of states
I_n = np.eye(n_state, dtype=int)  # dim (3,3)
H = np.vstack([I_n, -I_n])        # dim (6,3)

d_up = 10                 # noise upper bound
d_low = - d_up                    # noise lower bound
h_d = np.concatenate([
    np.full((n_state, 1), d_up),
    np.full((n_state, 1), -d_low)
])




error_list = []
error_vx = []
error_vy = []
error_w = []


    

starting_instant = 3

for index in range(starting_instant, len(df)):
    
# for index in range(starting_instant, starting_instant + 10):

    ### ========================================== ###
    ###      UNFALSIFIED PARAMETER SET   	Δ_k    ###
    ### ========================================== ###
    
    # Istant i minus 2
    F_0_minus2, G_0_minus2 = continuous_matrices(index - 2, steering_input, vx, vy, w, tau) # Maps F and G in continuos time
    delta_minus_2 = steer_angle(steering_input[index - 2])                      # Steering angle
    x_cont_minus_2 = np.array([[vx[index -2]], [vy[index -2]], [w[index -2]]])  # state vector in continuous time
    u_cont_minus_2 = np.array([[tau[index -2]], [delta_minus_2]])               # imput vector in continuous time
    
    # Istant i minus 1
    F_0_minus1, G_0_minus1 = continuous_matrices(index - 1, steering_input, vx, vy, w, tau)
    delta_minus_1 = steer_angle(steering_input[index - 1])
    x_cont_minus_1 = np.array([[vx[index -1]], [vy[index -1]], [w[index -1]]]) 
    u_cont_minus_1 = np.array([[tau[index -1]], [delta_minus_2]])

    # Actual instant
    x_act = np.array([[vx[index]], [vy[index]], [w[index]]])

    autonomous_func_minus_2 = F_0_minus2
    input_func_minus_2 = G_0_minus2
    autonomous_func_minus_1 = F_0_minus1
    input_func_minus_1 = G_0_minus1

    # Maps in discrete time
    f_dicr_minus_2, g_discr_minus_2, state_discr_minus_2 = compute_discrete_function_terms_single_step_euler(x_cont_minus_2, u_cont_minus_2, autonomous_func_minus_2, input_func_minus_2)
    f_dicr_minus_1, g_discr_minus_1, state_discr_minus_1 = compute_discrete_function_terms_single_step_euler(x_cont_minus_1, u_cont_minus_1, autonomous_func_minus_1, input_func_minus_1)
    
    error = (state_discr_minus_1) - (x_cont_minus_2)
    # error_vx.append(error[0, 0])  # Error fro longitudinal velocity
    # error_vy.append(error[1, 0])  # Error for lateral velocity
    # error_w.append(error[2, 0])   # Error for angolar velocity
    print(f"error 1 = {error[0, 0]}, error 2 = {error[1, 0]}, error 3 = {error[2, 0]}")


    
### --- ERROR EVALUATION --- ###

# average_error_vx = np.mean(error_vx)
# max_error_vx = np.max(error_vx)
# average_error_vy = np.mean(error_vy)
# max_error_vy = np.max(error_vy)
# average_error_w = np.mean(error_w)
# max_error_w = np.max(error_w)

# print(f"Average error for vx: {average_error_vx}, max error for vx: {max_error_vx}")
# print(f"Average error for vy: {average_error_vy}, max error for vy: {max_error_vy}")
# print(f"Average error for w: {average_error_w}, max error for w: {max_error_w}")

