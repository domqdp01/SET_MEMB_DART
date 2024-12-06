import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from function_needed import steering_2_steering_angle, evaluate_slip_angles, lateral_tire_force, rolling_friction, motor_force, F_friction_due_to_steering

### --- Import data --- ###

# file_path = 'car_1_Datarecording_12_04_2024_13_55_13.csv'
file_path = '100_lines.csv'

df = pd.read_csv(file_path)

# print(df.columns)
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

### --- ###

### --- MODEL FIXED PARAMETERS --- ###


theta_correction = 0.00768628716468811 # error between vehicle axis and vicon system reference axis
lr_reference = 0.115  #0.11650    # (measureing it wit a tape measure it's 0.1150) reference point location taken by the vicon system measured from the rear wheel
l_lateral_shift_reference = -0.01 # the reference point is shifted laterally by this amount 
#COM_positon = 0.084 #0.09375 #centre of mass position measured from the rear wheel

# car parameters
l = 0.1735 # [m]length of the car (from wheel to wheel)
m = 1.580 # mass [kg]
m_front_wheel = 0.847 #[kg] mass pushing down on the front wheel
m_rear_wheel = 0.733 #[kg] mass pushing down on the rear wheel
Cf = m_front_wheel/m
Cr = m_rear_wheel/m

COM_positon = l / (1+m_rear_wheel/m_front_wheel)
lr = COM_positon
lf = l-lr
# Automatically adjust following parameters according to tweaked values
l_COM = lr_reference - COM_positon

#lateral measurements
l_width = 0.08 # width of the car is 8 cm
m_left_wheels = 0.794 # mass pushing down on the left wheels
m_right_wheels = 0.805 # mass pushing down on the right wheels
# so ok the centre of mass is pretty much in the middle of the car so won't add this to the derivations


Jz = 1/12 * m *(l**2+l_width**2)

# full velocity range
# motor parameters
a_m =  25.35849952697754
b_m =  4.815326690673828
c_m =  -0.16377617418766022
time_C_m =  0.0843319296836853
# friction parameters
a_f =  1.2659882307052612
b_f =  7.666370391845703
c_f =  0.7393041849136353
d_f =  -0.11231517791748047

# steering angle curve --from fitting on vicon data
a_s =  1.392930030822754
b_s =  0.36576229333877563
c_s =  0.0029959678649902344 - 0.03 # littel adjustment to allign the tire curves
d_s =  0.5147881507873535
e_s =  1.0230425596237183


# Front wheel parameters:
d_t_f =  -0.8406859636306763
c_t_f =  0.8407371044158936
b_t_f =  8.598039627075195
# Rear wheel parameters:
d_t_r =  -0.8546739816665649
c_t_r =  0.959108829498291
b_t_r =  11.54928207397461


#additional friction due to steering angle
# Friction due to steering parameters:
a_stfr =  -0.11826395988464355
b_stfr =  5.915864944458008
d_stfr =  0.22619032859802246
e_stfr =  0.7793111801147461

### --- ###

### --- POLYTOPIC PARAMETERS --- ###

n = 6   # number of states of the system
I_n = np.ones((6, 6), dtype=int)
# I_n = np.eye(n)
H = np.vstack([I_n, -I_n])

d_up = 10

h_d = np.concatenate([
    np.full((n, 1), d_up),  
    np.full((n, 1), -d_up)  
])

# print(h_d)
### --- ### 

### --- INITIAL INSTANT --- ###

z_0 = np.array([
                x[0],
                y[0],
                yaw[0],
                vx[0],
                vy[0],
                w[0]
])

delta0 = steering_2_steering_angle(steering_input[0],a_s,b_s,c_s,d_s,e_s)
alpha_f0, alpha_r0 = evaluate_slip_angles(vx[0], vy[0], w[0], lf, lr, delta0)
Fy_f_0 = lateral_tire_force(alpha_f0, d_t_f, c_t_f, b_t_f, m_front_wheel)
Fy_r_0 = lateral_tire_force(alpha_r0, d_t_r, c_t_r, b_t_r, m_rear_wheel)
F_rolling_0 = rolling_friction(vx[0], a_f, b_f, c_f, d_f)
F_m_0 = motor_force(tau[0], v, a_m,b_m,c_m)
F_fric_0 = F_friction_due_to_steering(delta0, vx[0], a_stfr, b_stfr, d_stfr, e_stfr)
F_x_0 = np.sum(F_rolling_0) + np.sum(F_m_0) + np.sum(F_fric_0)  # Riduci a scalare se necessario
F_x_0_f = Cf * F_x_0
F_x_0_r = Cr * F_x_0

f_0 = np.array([
                vx[0] * np.cos(yaw[0]) - vy[0] * np.sin(yaw[0]),
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
                - 1/m * Fy_f_0 * np.sin(delta0),
                1/m * (Fy_r_0 + Fy_f_0 * np.cos(delta0)),
                1/Jz * (lf * Fy_f_0 * np.cos(delta0) - lr * Fy_r_0)
])

tolerance = np.full((2 * n, 1), 1e-5)
Ai_minus1 = -H @ g_0.reshape(-1, 1) + tolerance
bi_minus1 = h_d - H @ z_0.reshape(-1, 1) + H @ f_0.reshape(-1, 1)


### --- ###

### --- SME ALGORITHM --- ###

for i in range(1, len(df)):

    z_i = [x[i], y[i], yaw[i], vx[i], vy[i], w[i]]
    delta_i = steering_2_steering_angle(steering_input[i],a_s,b_s,c_s,d_s,e_s)
    alpha_fi, alpha_ri = evaluate_slip_angles(vx[i], vy[i], w[i], lf, lr, delta_i)
    Fy_f_i = lateral_tire_force(alpha_fi, d_t_f, c_t_f, b_t_f, m_front_wheel)
    Fy_r_i = lateral_tire_force(alpha_ri, d_t_r, c_t_r, b_t_r, m_rear_wheel)
    F_rolling_i = rolling_friction(vx[i], a_f, b_f, c_f, d_f)
    F_m_i = motor_force(tau[i], v, a_m,b_m,c_m)
    F_fric_i = F_friction_due_to_steering(delta_i, vx[i], a_stfr, b_stfr, d_stfr, e_stfr)
    F_x_i = np.sum(F_rolling_i) + np.sum(F_m_i) + np.sum(F_fric_i)  
    F_x_i_f = Cf * F_x_i
    F_x_i_r = Cr * F_x_i

    f_i = np.array([
                    vx[i] * np.cos(yaw[i]) - vy[i] * np.sin(yaw[i]),
                    vx[i] * np.sin(yaw[i]) + vy[i] * np.cos(yaw[i]),
                    w[i],
                    1/m * (F_x_i_r + F_x_i_f * np.cos(delta_i)) + w[i] * vy[i],
                    1/m * (F_x_i_f * np.sin(delta_i)) - w[i] * vx[i],
                    lf/Jz * (F_x_i_f * np.sin(delta_i))
    ])

    g_i = np.array([
                    0,
                    0,
                    0,
                    - 1/m * Fy_f_i * np.sin(delta_i),
                    1/m * (Fy_r_i + Fy_f_i * np.cos(delta_i)),
                    1/Jz * (lf * Fy_f_i * np.cos(delta_i) - lr * Fy_r_i)
    ])

    tolerance = np.full((2 * n, 1), 1e-5)
    Ai = -H @ g_i.reshape(-1, 1) + tolerance
    bi = h_d - H @ np.array(z_i).reshape(-1, 1) + H @ f_i.reshape(-1, 1)

    if np.any(Ai == 0):
        print(f"Iteration {i}: Skipping due to zero in A")
        continue

    # Concatenate to form A and b
    A = np.vstack((Ai, Ai_minus1))  # Concatenate vertically
    b = np.vstack((bi, bi_minus1))  # Concatenate vertically

    # Solve for a: element-wise ratio b / A
    a_values = b / A
    valid_a = []
    valid_A = []
    valid_b = []

    for idx, a in enumerate(a_values.flatten()):  # Iterate over all potential 'a' values
        satisfy_all = True

        for j in range(len(A)):  # Check if all inequalities are satisfied
            if not (A[j] * a <= b[j]):
                satisfy_all = False
                break

        if satisfy_all:  # If valid, save the values
            valid_a.append(a)
            valid_A.append(A[idx])  # Ensure 2D shape
            valid_b.append(b[idx])  # Ensure 2D shape
    valid_a = np.sort(valid_a)
    # Update Ai_minus1 and bi_minus1 with the valid values for the next iteration
    if valid_A and valid_b:  # Only update if valid values exist
        Ai_minus1 = np.vstack(valid_A)  # Stack all valid A
        bi_minus1 = np.vstack(valid_b)  # Stack all valid b
    print(f" Iteration {i}")
    print(f"a : {valid_a}")

    
    
