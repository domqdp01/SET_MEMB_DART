import pandas as pd
import numpy as np
from scipy.optimize import linprog
from funct_fin import steer_angle, rolling_friction, motor_force, F_friction_due_to_steering, slip_angles, lateral_tire_forces, friction
from discretization_function import compute_discrete_function_terms_single_step_euler
from continuous_matrix_function import continuous_matrices
from sympy import symbols, Matrix,And, solve, reduce_inequalities, simplify
from function_for_parameter_estimation import solve_constrained_QP, compute_vertex_centroid


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
file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_17_10_42.csv' # uniform nosi not centered
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

d_up = 0.001                  # noise upper bound
d_low = - d_up                    # noise lower bound
h_d = np.concatenate([
    np.full((n_state, 1), d_up),
    np.full((n_state, 1), -d_low)
])

def remove_duplicates(values, epsilon):
    filtered = []
    for value in values:
        if all(abs(value - f) > epsilon for f in filtered):
            filtered.append(value)
    return np.array(sorted(filtered))

def solve_sys(A, b):
    mu = symbols('mu')
    A = Matrix(A).tolist()
    b = Matrix(b).tolist()
    inequalities = [(simplify(A[i][0] * mu <= b[i][0])) for i in range(len(A))]
    # system = And(*inequalities)
    solution = solve(inequalities, mu)
    # ineq = [(simplify(A[i][0] * mu <= b[i][0])) for i in range(len(A))]
    # solution = reduce_inequalities(ineq, mu)
    
    return solution

# error_list = []
# error_vx = []
# error_vy = []
# error_w = []

regressor = []
observation = []
Hp = []
hp = []
    

starting_instant = 3

for index in range(starting_instant, len(df)):
# for index in range(starting_instant, starting_instant + 30):

    ### ========================================== ###
    ###      UNFALSIFIED PARAMETER SET   	Δ_k    ###
    ### ========================================== ###
    
    # Istant i minus 2
    F_0_minus2, G_0_minus2 = continuous_matrices(index - 2, steering_input, vx, vy, w, tau) # Maps F and G in continuos time
    delta_minus_2 = steer_angle(steering_input[index - 2])
    x_cont_minus_2 = np.array([[vx[index -2]], [vy[index -2]], [w[index -2]]])  # state vector in continuous time
    u_cont_minus_2 = np.array([[tau[index -2]], [delta_minus_2]])               # imput vector in continuous time
    
    # Istant i minus 1
    F_0_minus1, G_0_minus1 = continuous_matrices(index - 1, steering_input, vx, vy, w, tau)
    delta_minus_1 = steer_angle(steering_input[index - 1])
    x_cont_minus_1 = np.array([[vx[index -1]], [vy[index -1]], [w[index -1]]]) 
    u_cont_minus_1 = np.array([[tau[index -1]], [delta_minus_2]])

    autonomous_func_minus_2 = F_0_minus2
    input_func_minus_2 = G_0_minus2
    autonomous_func_minus_1 = F_0_minus1
    input_func_minus_1 = G_0_minus1

    # Maps in discrete time
    f_dicr_minus_2, g_discr_minus_2, state_discr_minus_2 = compute_discrete_function_terms_single_step_euler(x_cont_minus_2, u_cont_minus_2, autonomous_func_minus_2, input_func_minus_2)
    f_dicr_minus_1, g_discr_minus_1, state_discr_minus_1 = compute_discrete_function_terms_single_step_euler(x_cont_minus_1, u_cont_minus_1, autonomous_func_minus_1, input_func_minus_1)
    
    # error = np.abs(state_discr_minus_1) - np.abs(x_cont_minus_1)
    # error_vx.append(error[0, 0])  # Error fro longitudinal velocity
    # error_vy.append(error[1, 0])  # Error for lateral velocity
    # error_w.append(error[2, 0])   # Error for angolar velocity

    ## The inequality to solve is: - H * G * mu <= h_d - H * x_discr + H * F
    ## Grouping the terms: A = - H * G and b = h_d - H * x_discr + H * F
    ## Finally:  A * mu <= b

    if index == starting_instant:
        A_i_minus2 = - H @ g_discr_minus_2
        b_i_minus2 = h_d - H @ state_discr_minus_2 + H @ f_dicr_minus_2
    else:
        pass
    # A_i_minus2 = - H @ g_discr_minus_2
    # b_i_minus2 = h_d - H @ state_discr_minus_2 + H @ f_dicr_minus_2
    
    A_i_minus1 = - H @ g_discr_minus_1
    b_i_minus1 = h_d - H @ state_discr_minus_1 + H @ f_dicr_minus_1

    A = np.concatenate([A_i_minus1, A_i_minus2])
    b = np.concatenate([b_i_minus1, b_i_minus2])



    ### ========================================== ###
    ###        FEASIBLE PARAMETER SET      Θ_k     ###
    ### ========================================== ###

    mu_values = np.where(A != 0, b / A, np.inf)  # Skip division where A = 0
    act_mu = b_i_minus1/A_i_minus1

    # print(f"Act_mu = {act_mu}")
    valid_mu = []
    valid_A = []
    valid_b = [] 
    
    for idx, mu in enumerate(mu_values.flatten()):
        satisfy_all = True
        for j in range(len(A)):
            if not (A[j] * mu <= b[j] + 1e-6):
                satisfy_all = False
                break
        
        if satisfy_all:
            valid_A.append(A[idx])
            valid_b.append(b[idx])
            valid_mu.append(max(mu, 0))  # Ensure non-negative mu
    
    valid_mu = remove_duplicates(valid_mu, epsilon=1e-4)   # This removes values similar to each other
    valid_mu = np.sort(valid_mu)
  
    print(f"Iteration: {index}: mu ∈ [{valid_mu[0]:.4f}, {valid_mu[1]:.4f}] ")
    A_i_minus2 = []
    b_i_minus2 = []
    # Update Ai_minus1 and bi_minus1 with the valid values for the next iteration
    if valid_A and valid_b:  # Only update if valid values exist
        
        A_i_minus2 = np.vstack(valid_A)  # Stack all valid A
        b_i_minus2 = np.vstack(valid_b)  # Stack all valid b
    
    valid_A = remove_duplicates(valid_A, epsilon=1e-1)
    valid_b = remove_duplicates(valid_b, epsilon=1e-1)
    A_i_minus2 = remove_duplicates(valid_A, epsilon=1e-1)
    b_i_minus2 = remove_duplicates(valid_b, epsilon=1e-1)
    
    # ### --- SOLUTION TYPE 2 --- ###
    # # Another type of solution (MUCH SLOWER)    
    # solution = solve_sys(A, b)
    # print(f"iteration {index}: mu = {solution}")
    # A_i_minus2 = A_i_minus1
    # b_i_minus2 = b_i_minus1

    # print("### Debugging Iteration ###")
    # print(f"Iteration: {index}")
    # print("A_i_minus2:")
    # print(A_i_minus2)
    # print("b_i_minus2:")
    # print(b_i_minus2)
    # # print("valid_A:")
    # # print(valid_A)
    # # print("valid_b:")
    # # print(valid_b)
    # print("###########################\n")


    ### ========================================== ###
    ###         PARAMETER ESIMATION      Θ^        ###
    ### ========================================== ###

    vertices = [valid_mu[0], valid_mu[1]]
    # print(vertices)

    centroid = compute_vertex_centroid(vertices)
    # print(centroid)

    N = 24 # window of data


    if len(regressor) < N:
        for i in range(len(A_i_minus1)):
            regressor.append(A_i_minus1[i])
            observation.append(b_i_minus1[i])

    else:
        for i in range(len(A_i_minus1)):
            regressor.append(A_i_minus1[i])
            observation.append(b_i_minus1[i])
        regressor = regressor[6:]
        observation = observation[6:]


    if len(Hp) < N:
        for i in range(len(A_i_minus2)):
            Hp.append(A_i_minus2[i])
            hp.append(b_i_minus2[i])
    else:
        for i in range(len(A_i_minus2)):
            Hp.append(A_i_minus2[i])
            hp.append(b_i_minus2[i])
        Hp = Hp[6:]
        hp = hp[6:]   
    # print("Regressor:", regressor)
    # print("Observation:", observation)
    # print("Hp:", Hp)
    # print("hp:", hp)
    mu_hat = solve_constrained_QP(np.array(regressor), np.array(observation), np.array(Hp), np.array(hp))
    print(f"mu estimated = {mu_hat}")


# for i, elem in enumerate(regressor):
#     print(f"Dimensione di regressor[{i}]:", np.shape(elem))

# for i, elem in enumerate(observation):
#     print(f"Dimensione di observation[{i}]:", np.shape(elem))

# for i, elem in enumerate(Hp):
#     print(f"Dimensione di Hp[{i}]:", np.shape(elem))

# for i, elem in enumerate(hp):
#     print(f"Dimensione di hp[{i}]:", np.shape(elem))



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

