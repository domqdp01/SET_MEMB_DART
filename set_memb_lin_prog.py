import pandas as pd
import numpy as np
from scipy.optimize import linprog
from function_needed import steering_2_steering_angle, evaluate_slip_angles, lateral_tire_force, rolling_friction, motor_force, F_friction_due_to_steering

### --- Import data --- ###
# file_path = 'car_1_Datarecording_12_04_2024_13_55_13.csv'
file_path = '100_lines.csv'

df = pd.read_csv(file_path)

# Load data
vx = df['velocity x'].values
vy = df['velocity y'].values
x = df['vicon x'].values
y = df['vicon y'].values
v = df['vel encoder'].values
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
I_n = np.ones((6, 6), dtype=int)
H = np.vstack([I_n, -I_n])

d_up = 10
h_d = np.concatenate([
    np.full((n, 1), d_up),
    np.full((n, 1), -d_up)
])

### --- INITIAL INSTANT --- ###
z_0 = np.array([x[0], y[0], yaw[0], vx[0], vy[0], w[0]])

delta0 = steering_2_steering_angle(steering_input[0], a_s, b_s, c_s, d_s, e_s)
alpha_f0, alpha_r0 = evaluate_slip_angles(vx[0], vy[0], w[0], lf, lr, delta0)
Fy_f_0 = lateral_tire_force(alpha_f0, d_t_f, c_t_f, b_t_f, m_front_wheel)
Fy_r_0 = lateral_tire_force(alpha_r0, d_t_r, c_t_r, b_t_r, m_rear_wheel)
F_rolling_0 = rolling_friction(vx[0], a_f, b_f, c_f, d_f)
F_m_0 = motor_force(tau[0], v, a_m, b_m, c_m)
F_fric_0 = F_friction_due_to_steering(delta0, vx[0], a_stfr, b_stfr, d_stfr, e_stfr)
F_x_0 = np.sum(F_rolling_0) + np.sum(F_m_0) + np.sum(F_fric_0)
F_x_0_f = Cf * F_x_0
F_x_0_r = Cr * F_x_0

f_0 = np.array([
    vx[0] * np.cos(yaw[0]) + vy[0] * np.sin(yaw[0]),
    vx[0] * np.sin(yaw[0]) + vy[0] * np.cos(yaw[0]),
    w[0],
    1/m * (F_x_0_r + F_x_0_f * np.cos(delta0)) + w[0] * vy[0],
    1/m * (F_x_0_f * np.sin(delta0)) - w[0] * vx[0],
    lf/Jz * (F_x_0_f * np.sin(delta0))
])

g_0 = np.array([
    0,
    0,
    0,
    1/m * Fy_f_0 * np.sin(delta0),
    1/m * (Fy_r_0 + Fy_f_0 * np.cos(delta0)),
    1/Jz * (lf * Fy_f_0 * np.cos(delta0) - lr * Fy_r_0)
])

Ai_minus1 = -H @ g_0.reshape(-1, 1)
bi_minus1 = h_d - H @ z_0.reshape(-1, 1) + H @ f_0.reshape(-1, 1)

### --- SME ALGORITHM --- ###
for i in range(1, len(df)):
    z_i = np.array([x[i], y[i], yaw[i], vx[i], vy[i], w[i]])
    deltai = steering_2_steering_angle(steering_input[i], a_s, b_s, c_s, d_s, e_s)
    alpha_fi, alpha_ri = evaluate_slip_angles(vx[i], vy[i], w[i], lf, lr, deltai)
    Fy_f_i = lateral_tire_force(alpha_fi, d_t_f, c_t_f, b_t_f, m_front_wheel)
    Fy_r_i = lateral_tire_force(alpha_ri, d_t_r, c_t_r, b_t_r, m_rear_wheel)
    F_rolling_i = rolling_friction(vx[i], a_f, b_f, c_f, d_f)
    F_m_i = motor_force(tau[i], v, a_m, b_m, c_m)
    F_fric_i = F_friction_due_to_steering(deltai, vx[i], a_stfr, b_stfr, d_stfr, e_stfr)
    F_x_i = np.sum(F_rolling_i) + np.sum(F_m_i) + np.sum(F_fric_i)
    F_x_i_f = Cf * F_x_i
    F_x_i_r = Cr * F_x_i

    f_i = np.array([
        vx[i] * np.cos(yaw[i]) + vy[i] * np.sin(yaw[i]),
        vx[i] * np.sin(yaw[i]) + vy[i] * np.cos(yaw[i]),
        w[i],
        1/m * (F_x_i_r + F_x_i_f * np.cos(deltai)) + w[i] * vy[i],
        1/m * (F_x_i_f * np.sin(deltai)) - w[i] * vx[i],
        lf/Jz * (F_x_i_f * np.sin(deltai))
    ])

    g_i = np.array([
        0,
        0,
        0,
        1/m * Fy_f_i * np.sin(deltai),
        1/m * (Fy_r_i + Fy_f_i * np.cos(deltai)),
        1/Jz * (lf * Fy_f_i * np.cos(deltai) - lr * Fy_r_i)
    ])

    Ai = -H @ g_i.reshape(-1, 1)
    bi = h_d - H @ z_i.reshape(-1, 1) + H @ f_i.reshape(-1, 1)

    # Linear programming solution
    c = np.zeros(Ai.shape[1])
    result = linprog(c, A_ub=Ai, b_ub=bi, method='highs')

    if result.success:
        a_optimal = result.x
        print(f"Iteration {i}, a_optimal: {a_optimal}")
    else:
        print(f"Iteration {i}: No valid solution found")
