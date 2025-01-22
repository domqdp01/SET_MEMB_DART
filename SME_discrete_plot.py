import pandas as pd
import numpy as np
from scipy.optimize import linprog
from funct_fin import steer_angle, rolling_friction, motor_force, F_friction_due_to_steering, slip_angles, lateral_tire_forces, friction
from discretization_function import compute_discrete_function_terms_single_step_euler
from continuous_matrix_function import continuous_matrices
from sympy import symbols, Matrix,And, solve, reduce_inequalities, simplify
import matplotlib.pyplot as plt
from function_for_parameter_estimation import solve_constrained_QP, compute_vertex_centroid


### --- IMPORT AND LOADING DATA IN 'CONTINUOUS TIME' --- ###


# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_18_2024_14_43_23.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_18_2024_14_52_47.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_10_2025_14_35_50.csv'  # noise_up = 0.05
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_10_2025_14_39_57.csv'  # noise_up = 0.5 ### TOP
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_10_2025_14_50_47.csv'  # noise_up = 1
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_10_2025_17_07_52.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_10_2025_17_12_50.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_10_47_21.csv'  ### TOP
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_16_41_15.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_16_58_58.csv' # unifrom noise
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_17_04_25.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_17_10_42.csv' # uniform nosi not centered
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_17_13_40.csv'
file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_20_2025_10_46_12.csv'
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

d_up = 0.01                # noise upper bound
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


# Initialize the plot
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots(figsize=(12, 8))  # Adjust figure size
ax.set_xlim()
ax.set_ylim(0.6, 1.4)
ax.set_title("")
ax.set_xlabel("")
ax.set_xticks([])
ax.set_ylabel("μ")

lines_ai = [ax.plot([], [], label=f"ai[{j}]", color="blue")[0] for j in range(2)]
lines_valid_a = [ax.plot([], [], label=f"valid_a[{j}]", color="green", linestyle=":")[0] for j in range(2)]

fill_mu_i = None
fill_valid_mu = None
line_mu_hat = None

# Add legend entries for the areas
area_handles = [
    plt.Line2D([0], [0], color="blue", alpha=0.35, lw=10, label=r"$\Delta_k$"),
    plt.Line2D([0], [0], color="green", alpha=0.7, lw=10, label=r"$\Theta_k$")
]

ax.legend(handles= area_handles, loc='upper left')


regressor = []
observation = []
Hp = []
hp = []

starting_instant = 35
ending_instant = len(df)
for index in range(starting_instant, ending_instant):
# for index in range(starting_instant, starting_instant + 10):

    # FILLED PART IN THE PLOT
    if fill_mu_i:
        fill_mu_i.remove()
    
    if fill_valid_mu:
        fill_valid_mu.remove()

    fill_mu_i = None
    fill_valid_mu = None


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

    mu_i = np.unique(b_i_minus1 / A_i_minus1)  # These are the values of mu obtained from the actual iteration.
    mu_i = np.sort(mu_i)

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
    if valid_mu.shape == 2:
        print(f"Iteration: {index}: mu ∈ [{valid_mu[0]:.4f}, {valid_mu[1]:.4f}] ")
    else:
        print(f"Iteration: {index}: mu = {valid_mu}")
    
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
    # print(f"mu estimated = {mu_hat}")
    
    x_vals = np.linspace(0, 1, 10)

    mu_up = valid_mu[1]
    mu_low = valid_mu[0]
    line_up = mu_up * np.ones(10)
    line_low = mu_low * np.ones(10)
    fill_valid_mu = ax.fill_between(x_vals, line_low, line_up, alpha= 0.9, color='green')

    mu_i_up = mu_i[3]
    mu_i_low = mu_i[2]
    line_up_i = mu_i_up * np.ones(10)
    line_low_i = mu_i_low * np.ones(10)
    fill_mu_i = ax.fill_between(x_vals, line_low_i, line_up_i, alpha=0.2, color='blue')
    
    if line_mu_hat:
        line_mu_hat.remove()

    line_mu_hat = ax.axhline(mu_hat, color='red', linestyle='--', linewidth=1.5, label=r"$\mu_{\mathrm{estimated}}$")

    ax.set_title(f"$μ \\; \\in \\; [{mu_low:.3f}, {mu_up:.3f}]$\n"
                 f"$\\hat{{\\mu}} = {mu_hat.item():.3f}$\n"
                f"Iteration number: {index}")

    plt.pause(0.001)

    # print(mu_i)

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

plt.ioff()
plt.show()

