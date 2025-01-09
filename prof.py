import pandas as pd
import numpy as np
from scipy.optimize import linprog
from funct_fin import steer_angle, rolling_friction, motor_force, F_friction_due_to_steering, slip_angles, lateral_tire_forces, friction
import cvxpy as cp
from sympy import symbols, Matrix,And, solve, reduce_inequalities, simplify
from z3 import Solver, Real, And


### --- Import data --- ###

# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_16_2024_16_28_38.csv'  # straight line
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_16_2024_16_45_57.csv' # for d = 5: mu = 0.1352, 0.2007;; for d = 4.5: mu = 0.147, 0.179
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_16_2024_16_53_06.csv' # for d = 5: mu = 0.0372, 0.1767
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_16_2024_16_55_42.csv' # for d = 5: mu = 0.0239, 0.1957
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_16_2024_16_59_04.csv' # for d = 5, mu = 0.0289, 0.1753
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_16_2024_17_41_33.csv' # for d = 5, mu = 0.1509, 0.1526
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_16_2024_17_46_34.csv' # for d = 5, mu = 0.0370; 0.1946
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_16_2024_17_55_47.csv' # for d = 5: mu = 0.0295, 0.1995
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_16_2024_17_57_52.csv' # for d = 5: mu = 0.0417, 0.2007
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_16_2024_18_05_10.csv' # for d = 5: mu = 0.0229, 0.1929
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_16_2024_18_07_06.csv' # for d = 5: mu = 0.0310, 0.1954
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_16_2024_18_12_36.csv' # for d = 5: mu = 0.0983, 0.1900
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_17_2024_10_42_40.csv' # for d = 5: mu = 0.0338, 0.1748  different noise
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_17_2024_10_50_47.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_17_2024_11_23_14.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_17_2024_11_15_57.csv' # for d = 5: mu = 0.1406, 0.1858
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_17_2024_11_27_16.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_17_2024_11_31_19.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_17_2024_11_38_17.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_17_2024_11_41_08.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_17_2024_11_45_13.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_17_2024_15_21_35.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_18_2024_14_26_01.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_18_2024_14_40_00.csv'
file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_18_2024_14_43_23.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_18_2024_14_52_47.csv'

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


# def solve_sys(A, b):

#     mu = Real('mu')
#     solver = Solver()

#     # Costruzione delle disuguaglianze
#     for i in range(len(A)):
#         solver.add(A[i][0] * mu <= b[i][0])

#     # Risoluzione del sistema
#     if solver.check() == 'sat':
#         model = solver.model()
#         return model
    

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

# Define dynamic models
l = 0.175
lr = 0.54 * l
lf = l - lr
m = 1.67
m_front_wheel = 0.847
m_rear_wheel = 0.733
Cf = m_front_wheel / m
Cr = m_rear_wheel / m
Jz = 0.006513

### --- POLYTOPIC PARAMETERS --- ###

n = 3
I_n = np.eye(n, dtype=int)  # dim (3,3)
H = np.vstack([I_n, -I_n])  # dim (12,6)

d_up = 3.5
d_low = - d_up
h_d = np.concatenate([
    np.full((n, 1), d_up),
    np.full((n, 1), -d_low)
])

### --- INITIAL ITERATION INSTANT --- ###

z_1 = np.array([[vx[1]], [vy[1]], [w[1]]])  # dim: (6,1)

# LATERAL DYNAMICS
delta0 = steer_angle(steering_input[0])
alpha_f0, alpha_r0 = slip_angles(vx[0], vy[0], w[0], delta0)
F_y_f_0, F_y_r_0 = lateral_tire_forces(alpha_f0, alpha_r0)

# LONGITUDINAL DYNAMICS
# F_rolling_0 = rolling_friction(vx[0])
F_rolling_0 = rolling_friction(vx[0])
F_m_0 = motor_force(tau[0], vx[0])
F_fric_0 = F_friction_due_to_steering(delta0, vx[0])
F_x_0 = np.sum(F_rolling_0) + np.sum(F_m_0) + np.sum(F_fric_0)
F_x_0_f = Cf * F_x_0
F_x_0_r = Cr * F_x_0
# F_m_0 = motor_force(tau[0], vx[0])
# # F_fric_0 = F_friction_due_to_steering(delta0, vx[0])
# F_fric_0 = friction(vx[0])
# F_x_0 = np.sum(F_m_0) + np.sum(F_fric_0)
# # F_x_0_f = Cf * F_x_0
# # F_x_0_r = Cr * F_x_0
# F_x_0_f =  F_x_0/2
# F_x_0_r = F_x_0/2

# SET UP MATRICES
F_0 = np.array([
    1/m * (F_x_0_r + F_x_0_f * np.cos(delta0)) + w[0] * vy[0],
    1/m * (F_x_0_f * np.sin(delta0)) - w[0] * vx[0],
    lf/Jz * (F_x_0_f * np.sin(delta0))
]).reshape(-1, 1)

G_0 = np.array([
    -1/m * F_y_f_0 * np.sin(delta0),
    1/m * (F_y_r_0 + F_y_f_0 * np.cos(delta0)),
    1/Jz * (lf * F_y_f_0 * np.cos(delta0) - lr * F_y_r_0)
]).reshape(-1, 1)

Ai_minus1 = -H @ G_0  # dim (6,1)
bi_minus1 = h_d - H @ z_1 + H @ F_0  # dim (6,1)

### --- FUNCTION TO REMOVE DUPLICATES --- ###
def remove_duplicates(values, epsilon=1e-6):
    filtered = []
    for value in values:
        if all(abs(value - f) > epsilon for f in filtered):
            filtered.append(value)
    return np.array(sorted(filtered))

### --- SME ALGORITHM --- ###

for i in range(900, len(df)):

    # STATE VECTOR
    z_i = np.array([[vx[i]], [vy[i]], [w[i]]])

    # LATERAL DYNAMICS
    deltai_minus1 = steer_angle(steering_input[i - 1])
    alpha_fi_minus1, alpha_ri_minus1 = slip_angles(vx[i - 1], vy[i - 1], w[i - 1], deltai_minus1)
    Fy_f_i_minus1, Fy_r_i_minus1 = lateral_tire_forces(alpha_fi_minus1, alpha_ri_minus1)

    # LONGITUDINAL DYNAMICS
    # # F_rolling_i_minus1 = rolling_friction(vx[i - 1])
    # F_m_i_minus1 = motor_force(tau[i - 1], vx[i - 1])
    # F_fric_i_minus1 = friction(vx[i - 1])
    # # F_fric_i_minus1 = F_friction_due_to_steering(deltai_minus1, vx[i - 1])
    # # F_x_i_minus1 = F_rolling_i_minus1 + F_m_i_minus1 + F_fric_i_minus1
    # F_x_i_minus1 = F_m_i_minus1 + F_fric_i_minus1
    # F_x_i_minus1_f = Cf * F_x_i_minus1
    # F_x_i_minus1_r = Cr * F_x_i_minus1
    # F_x_i_minus1_f = F_x_i_minus1/2
    # F_x_i_minus1_r = F_x_i_minus1/2

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


    Ai = -H @ G_i_minus1
    bi = h_d - H @ z_i + H @ F_i_minus1
    # mu_i = bi/Ai
    # print(f"mu_i: {mu_i}")
    

    A = np.concatenate([Ai, Ai_minus1]) + 1e-6  # Add tolerance to avoid dividing by zero
    b = np.concatenate([bi, bi_minus1])

    # ### METHOD 1 ###
    # mu_values = np.where(A != 0, b / A, np.inf)
    # # print(f"mu_values: {mu_values}")
    # # print(mu_values)
    # valid_mu = []
    # valid_A = []
    # valid_b = []

    # for idx, mu in enumerate(mu_values.flatten()):
    #     satisfy_all = True
    #     for j in range(len(A)):
    #         if not (A[j] * mu <= b[j] + 1e-6):
    #             satisfy_all = False
    #             break
    #     # print(mu)
    #     if satisfy_all:
    #         valid_A.append(A[idx])
    #         valid_b.append(b[idx])
    #         # valid_mu.append(mu)
    #         valid_mu.append(max(mu, 0))  # Ensure non-negative mu

    # # Remove duplicates in valid_mu
    # valid_mu = remove_duplicates(valid_mu, epsilon=1e-6)
    # valid_mu = np.sort(valid_mu)

    # print(f"Iteration: {i}, valid_mu: {valid_mu}")

    # # Update Ai_minus1 and bi_minus1 with the valid values for the next iteration
    # if valid_A and valid_b:  # Only update if valid values exist
    #     Ai_minus1 = np.vstack(valid_A)  # Stack all valid A
    #     bi_minus1 = np.vstack(valid_b)  # Stack all valid b


### METHOD 2: SYMPY library ###

    solution = solve_sys(A, b)
    print(f"iteration {i}: mu = {solution}")

    Ai_minus1 = Ai
    bi_minus1 = bi



