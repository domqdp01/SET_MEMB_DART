import pandas as pd
import numpy as np
from scipy.optimize import linprog
from funct_fin import  steer_angle, rolling_friction, motor_force, F_friction_due_to_steering, slip_angles, lateral_tire_forces
import cvxpy as cp
import pdb

### --- Import data --- ###
# file_path = 'car_1_Datarecording_12_04_2024_13_55_13.csv'
# file_path = '100_lines.csv'
# file_path = 'circle_fin.csv'
# file_path = 'car_1_Datarecording_12_11_2024_11_43_35.csv'
# file_path = 'car_1_Datarecording_12_11_2024_12_23_35.csv'
# file_path = 'car_1_Datarecording_12_11_2024_12_23_08.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_11_2024_14_22_11.csv'  # No noise
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_11_2024_14_59_16.csv'   # Noise
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_16_2024_14_41_11.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_16_2024_15_08_13.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_16_2024_15_17_32.csv'
file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_16_2024_15_42_03.csv'

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

# define dynamic models
# hard coded vehicle parameters
l = 0.175
lr = 0.54*l # the reference point taken by the data is not exaclty in the center of the vehicle
#lr = 0.06 # reference position from rear axel
lf = l-lr

m = 1.67
m_front_wheel = 0.847
m_rear_wheel = 0.733
Cf = m_front_wheel / m
Cr = m_rear_wheel / m
Jz = 0.006513 # uniform rectangle of shape 0.18 x 0.12

### --- POLYTOPIC PARAMETERS --- ###

n = 3
I_n = np.eye(n, dtype=int)  # dim (3,3)
H = np.vstack([I_n, - I_n]) # dim (12,6)

d_up = 5
d_low = -d_up
h_d = np.concatenate([      # dim (12,1)
    np.full((n, 1), d_up),
    np.full((n, 1), -d_low)
])

### --- INITIAL INSTANT --- ###

# Here the initial instant start from interation number 1, not from 0. Remeber F() and G() are defined at instant i-1

# STATE VECTOR

z_1 = np.array([[vx[1]], [vy[1]], [w[1]]])  # dim: (3,1)

# LATERAL DYNAMICS

delta0 = steer_angle(steering_input[0])
alpha_f0, alpha_r0 = slip_angles(vx[0], vy[0], w[0], delta0)
F_y_f_0, F_y_r_0 = lateral_tire_forces(alpha_f0, alpha_r0)

# LONGITUDINAL DYNAMICS

F_rolling_0 = rolling_friction(vx[0])
F_m_0 = motor_force(tau[0], vx[0])
F_fric_0 = F_friction_due_to_steering(delta0, vx[0])
F_x_0 = np.sum(F_rolling_0) + np.sum(F_m_0) + np.sum(F_fric_0)
F_x_0_f = Cf * F_x_0
F_x_0_r = Cr * F_x_0

# SET UP MATRICES

F_0 = np.array([                 # dim: (3,1)
    (1/m * (F_x_0_r + F_x_0_f * np.cos(delta0))) + w[0] * vy[0],
    (1/m * (F_x_0_f * np.sin(delta0))) - w[0] * vx[0],
    lf/Jz * (F_x_0_f * np.sin(delta0))
]).reshape(-1, 1)

G_0 = np.array([                # dim: (3,1)
    -1/m * F_y_f_0 * np.sin(delta0),
    1/m * (F_y_r_0 + F_y_f_0 * np.cos(delta0)),
    1/Jz * (lf * F_y_f_0 * np.cos(delta0) - lr * F_y_r_0)
]).reshape(-1, 1)



Ai_minus1 = -H @ G_0  # dim (6,1)
bi_minus1 = h_d - H @ z_1 + H @ F_0 # dim (6,1)


### --- SME ALGORITHM --- ###

# for i in range (700, 1600):
# for i in range(1175, len(df)):
# for i in range(401, 1201):
for i in range(2, len(df)):

    # STATE VECTOR

    z_i = np.array([[vx[i]], [vy[i]], [w[i]]])

    # LATERAL DINAMICS

    deltai_minus1 = steer_angle(steering_input[i - 1])
    alpha_fi_minus1, alpha_ri_minus1 = slip_angles(vx[i - 1], vy[i - 1], w[i - 1], deltai_minus1)
    Fy_f_i_minus1, Fy_r_i_minus1 = lateral_tire_forces(alpha_fi_minus1, alpha_ri_minus1)

    # LONGITUDINAL DYNAMICS

    F_rolling_i_minus1 = rolling_friction(vx[i - 1])
    F_m_i_minus1 = motor_force(tau[i - 1], vx[i - 1])
    F_fric_i_minus1 = F_friction_due_to_steering(deltai_minus1, vx[i - 1])
    F_x_i_minus1 = F_rolling_i_minus1 + F_m_i_minus1 + F_fric_i_minus1
    F_x_i_minus1_f = Cf * F_x_i_minus1
    F_x_i_minus1_r = Cr * F_x_i_minus1

    # SET UP MATRICES

    F_i_minus1 = np.array([
        1/m * (F_x_i_minus1_r + F_x_i_minus1_f * np.cos(deltai_minus1)) + (w[i - 1] * vy[i - 1]),
        1/m * (F_x_i_minus1_f * np.sin(deltai_minus1)) - (w[i - 1] * vx[i - 1]),
        lf/Jz * (F_x_i_minus1_f * np.sin(deltai_minus1))
    ]).reshape(-1, 1)

    G_i_minus1 = np.array([
        -1/m * Fy_f_i_minus1 * np.sin(deltai_minus1),
        1/m * (Fy_r_i_minus1 + Fy_f_i_minus1 * np.cos(deltai_minus1)),
        1/Jz * (lf * Fy_f_i_minus1 * np.cos(deltai_minus1) - lr * Fy_r_i_minus1)
    ])

    G_i_minus1 = (G_i_minus1).reshape(-1, 1)

    Ai = - H @ G_i_minus1
    bi = h_d - H @ z_i + H @ F_i_minus1
    
    A = np.concatenate([Ai, Ai_minus1]) + 1e-6   ## Add tollerance to avoid dividing to zero
    b = np.concatenate([bi, bi_minus1]) 


    ### METHOD 1 ###

    # mu_values = b/A
    mu_values = np.where(A != 0, b / A, np.inf)


    valid_mu = []
    valid_A = []
    valid_b = []

    for idx, mu in enumerate(mu_values.flatten()):
        satisfy_all = True
        for j in range(len(A)):
            if not (A[j] * mu <= b[j] +1e-6):
                satisfy_all = False
                break

        if satisfy_all:
            valid_A.append(A[idx])
            valid_b.append(b[idx])
            # valid_mu.append(mu)
            if mu < 0:
                valid_mu.append(0)
            else:
                valid_mu.append(mu)

    valid_mu = np.sort(valid_mu)
    print(f"Iteration: {i}, valid_mu: {valid_mu}")

    # Update Ai_minus1 and bi_minus1 with the valid values for the next iteration
    if valid_A and valid_b:  # Only update if valid values exist
        Ai_minus1 = np.vstack(valid_A)  # Stack all valid A
        bi_minus1 = np.vstack(valid_b)  # Stack all valid 
        


    # ### METHOD 2: GLPK library ###

    # c = np.zeros((A.shape[1], 1))  # Objective function coefficients
    # mu_optimal = cp.Variable((A.shape[1], 1) ) # Decision variable
   
    # # Constraints
    # constraints = [A @ mu_optimal <= b]

    # # Objective: Minimize 0 (since you want feasibility)
    # objective = cp.Minimize(c @ mu_optimal)

    # # Problem definition
    # problem = cp.Problem(objective, constraints)

    # # Solve using GLPK
    # try:
    #     problem.solve(solver=cp.GLPK)
    #     if problem.status == cp.OPTIMAL:
    #         print(f"Iteration {i}, mu_optimal: {mu_optimal.value}")
    #     else:
    #         print(f"Iteration {i}: No valid solution found (Status: {problem.status})")
    # except Exception as e:
    #     print(f"Iteration {i}: Solver error - {str(e)}")

    # Ai_minus1 = Ai
    # bi_minus1 = bi

    # ### METHOD 3: linear programming library ###
    # c = np.eye(A.shape[1])
    # result = linprog(c, A_ub=A, b_ub=b, method='highs')

    # if result.success:
    #     # print(result)
    #     mu_optimal = result.x
    #     print(f"Iteration {i}, mu_optimal: {mu_optimal}")
    # else:
    #     print(f"Iteration {i}: No valid solution found")

    # Ai_minus1 = Ai
    # bi_minus1 = bi