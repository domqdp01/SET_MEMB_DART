import pandas as pd
import numpy as np
from scipy.optimize import linprog
from function_needed import  steer_angle, evaluate_slip_angles, lateral_tire_force, rolling_friction, motor_force, F_friction_due_to_steering
import cvxpy as cp


### --- Import data --- ###
# file_path = 'car_1_Datarecording_12_04_2024_13_55_13.csv'
# file_path = '100_lines.csv'
# file_path = 'circle_fin.csv'
# file_path = 'car_1_Datarecording_12_10_2024_16_33_20.csv'
file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_17_2024_11_45_13.csv'

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

### --- MODEL FIXED PARAMETERS --- ###
theta_correction = 0.00768628716468811
lr_reference = 0.115
l_lateral_shift_reference = -0.01

l = 0.1735
m = 1.580
m_front_wheel = 0.847
m_rear_wheel = 0.733
Cf = m_front_wheel / m
Cr = m_rear_wheel / m

COM_positon = l / (1 + m_rear_wheel / m_front_wheel)
lr = COM_positon
lf = l - lr

l_width = 0.08
Jz = 1 / 12 * m * (l**2 + l_width**2)

a_m, b_m, c_m = 25.3585, 4.8153, -0.1638
a_f, b_f, c_f, d_f = 1.266, 7.666, 0.739, -0.1123

a_s, b_s, c_s, d_s, e_s = 1.393, 0.3658, -0.027, 0.5148, 1.0230
d_t_f, c_t_f, b_t_f = -0.8407, 0.8407, 8.5980
d_t_r, c_t_r, b_t_r = -0.8547, 0.9591, 11.5493

a_stfr, b_stfr, d_stfr, e_stfr = -0.1183, 5.9159, 0.2262, 0.7793

### --- POLYTOPIC PARAMETERS --- ###

n = 6
# I_n = np.ones((n,n))
I_n = np.eye(6, dtype=int)
H = np.vstack([I_n, -I_n])

d_up = 5
d_low = - d_up
h_d = np.concatenate([
    np.full((n, 1), d_up),
    np.full((n, 1), -d_low)
])

### --- INITIAL INSTANT --- ###

z_0 = np.array([x[0], y[0], yaw[0], vx[0], vy[0], w[0]])
z_1 = np.array([x[1], y[1], yaw[1], vx[1], vy[1], w[1]])
delta0 = steer_angle(steering_input[0])
alpha_f0, alpha_r0 = evaluate_slip_angles(vx[0], vy[0], w[0], lf, lr, delta0)
Fy_f_0 = lateral_tire_force(alpha_f0, d_t_f, c_t_f, b_t_f, m_front_wheel)
Fy_r_0 = lateral_tire_force(alpha_r0, d_t_r, c_t_r, b_t_r, m_rear_wheel)
F_rolling_0 = rolling_friction(vx[0], a_f, b_f, c_f, d_f)
F_m_0 = motor_force(tau[0], vx[0], a_m, b_m, c_m)
F_fric_0 = F_friction_due_to_steering(delta0, vx[0], a_stfr, b_stfr, d_stfr, e_stfr)
F_x_0 = np.sum(F_rolling_0) + np.sum(F_m_0) + np.sum(F_fric_0)
F_x_0_f = Cf * F_x_0
F_x_0_r = Cr * F_x_0

F_0 = np.array([
    vx[0] * np.cos(yaw[0]) - vy[0] * np.sin(yaw[0]),
    vx[0] * np.sin(yaw[0]) + vy[0] * np.cos(yaw[0]),
    w[0],
    1/m * (F_x_0_r + F_x_0_f * np.cos(delta0)) + w[0] * vy[0],
    1/m * (F_x_0_f * np.sin(delta0)) - w[0] * vx[0],
    lf/Jz * (F_x_0_f * np.sin(delta0))
])

G_0 = np.array([
    0,
    0,
    0,
    -1/m * Fy_f_0 * np.sin(delta0),
    1/m * (Fy_r_0 + Fy_f_0 * np.cos(delta0)),
    1/Jz * (lf * Fy_f_0 * np.cos(delta0) - lr * Fy_r_0)
])

Ai_minus1 = - H @ G_0.reshape(-1, 1)
bi_minus1 = h_d - H @ z_1.reshape(-1, 1) + H @ F_0.reshape(-1, 1)



### --- SME ALGORITHM --- ###

for i in range (2, len(df)):
# for i in range(1, len(df)):
    z_i_minus1 = np.array([x[i - 1], y[i - 1], yaw[i - 1], vx[i - 1], vy[i - 1], w[i - 1]])
    z_i = np.array([x[i], y[i], yaw[i], vx[i], vy[i], w[i]])
    deltai_minus1 = steer_angle(steering_input[i - 1])
    alpha_fi_minus1, alpha_ri_minus1 = evaluate_slip_angles(vx[i - 1], vy[i - 1], w[i - 1], lf, lr, deltai_minus1)
    Fy_f_i_minus1 = lateral_tire_force(alpha_fi_minus1, d_t_f, c_t_f, b_t_f, m_front_wheel)
    Fy_r_i_minus1 = lateral_tire_force(alpha_ri_minus1, d_t_r, c_t_r, b_t_r, m_rear_wheel)
    F_rolling_i_minus1 = rolling_friction(vx[i - 1], a_f, b_f, c_f, d_f)
    F_m_i_misus1 = motor_force(tau[i - 1], vx[i - 1], a_m, b_m, c_m)
    F_fric_i_minus1 = F_friction_due_to_steering(deltai_minus1, vx[i - 1], a_stfr, b_stfr, d_stfr, e_stfr)
    F_x_i_minus1 = np.sum(F_rolling_i_minus1) + np.sum(F_m_i_misus1) + np.sum(F_fric_i_minus1)
    F_x_i_minus1_f = Cf * F_x_i_minus1
    F_x_i_minus1_r = Cr * F_x_i_minus1

    F_i = np.array([
        vx[i - 1] * np.cos(yaw[i - 1]) - vy[i - 1] * np.sin(yaw[i - 1]),
        vx[i - 1] * np.sin(yaw[i - 1]) + vy[i - 1] * np.cos(yaw[i - 1]),
        w[i - 1],
        1/m * (F_x_i_minus1_r + F_x_i_minus1_f * np.cos(deltai_minus1)) + w[i - 1] * vy[i - 1],
        1/m * (F_x_i_minus1_f * np.sin(deltai_minus1)) - w[i - 1] * vx[i - 1],
        lf/Jz * (F_x_i_minus1_f * np.sin(deltai_minus1))
    ])

    G_i = np.array([
        0,
        0,
        0,
        -1/m * Fy_f_i_minus1 * np.sin(deltai_minus1),
        1/m * (Fy_r_i_minus1 + Fy_f_i_minus1 * np.cos(deltai_minus1)),
        1/Jz * (lf * Fy_f_i_minus1 * np.cos(deltai_minus1) - lr * Fy_r_i_minus1)
    ])

    Ai = -H @ G_i.reshape(-1, 1)
    bi = h_d - H @ z_i.reshape(-1, 1) + H @ F_i.reshape(-1, 1)

    A = np.concatenate([Ai, Ai_minus1])
    b = np.concatenate([bi, bi_minus1])

    # mu_values = b/A

    # valid_mu = []
    # valid_A = []
    # valid_b = []

    # for idx, mu in enumerate(mu_values.flatten()):
    #     satisfy_all = True
    #     for j in range(len(A)):
    #         if not (A[j] * mu <= b[j]):
    #             satisfy_all = False
    #             break

    #     if satisfy_all:
    #         valid_A.append(A[idx])
    #         valid_b.append(b[idx])
    #         if mu < 0:
    #             valid_mu.append(0)
    #         else:
    #             valid_mu.append(mu)

    # valid_mu = np.sort(valid_mu)
    # print(valid_mu)

    # # Update Ai_minus1 and bi_minus1 with the valid values for the next iteration
    # if valid_A and valid_b:  # Only update if valid values exist
    #     Ai_minus1 = np.vstack(valid_A)  # Stack all valid A
    #     bi_minus1 = np.vstack(valid_b)  # Stack all valid b
    

    # print(b.shape)

    # Linear programming solution


    c = np.zeros(A.shape[1])
    result = linprog(c, A_ub=A, b_ub=b, method='highs')

    if result.success:
        
        mu_optimal = result.x
        print(f"Iteration {i}, mu_optimal: {mu_optimal}")
    else:
        print(f"Iteration {i}: No valid solution found")

    Ai_minus1 = Ai
    bi_minus1 = bi
    
    # c = np.zeros(A.shape[1])  # Objective function coefficients
    # mu_optimal = cp.Variable(A.shape[1])  # Decision variable

    # # Constraints
    # constraints = [A @ mu_optimal <= b.flatten()+ 10e-3]

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

    Ai_minus1 = Ai
    bi_minus1 = bi
    


