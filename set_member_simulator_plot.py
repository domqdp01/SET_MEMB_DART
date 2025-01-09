import pandas as pd
import numpy as np
from scipy.optimize import linprog
from funct_fin import steer_angle, rolling_friction, motor_force, F_friction_due_to_steering, slip_angles, lateral_tire_forces
import cvxpy as cp
import matplotlib.pyplot as plt

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
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_17_2024_16_05_56.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_17_2024_16_16_00.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_17_2024_16_19_22.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_17_2024_16_46_23.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_17_2024_16_51_36.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_18_2024_14_22_50.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_18_2024_14_26_01.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_18_2024_14_40_00.csv'
file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_18_2024_14_43_23.csv'

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
H = np.vstack([I_n, -I_n])  # dim (6,3)

d_up = 4
d_low = - d_up
h_d = np.concatenate([
    np.full((n, 1), d_up),
    np.full((n, 1), -d_low)
])

### --- INITIAL INSTANT --- ###

z_1 = np.array([[vx[1]], [vy[1]], [w[1]]])  # dim: (6,1)

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

prev_mu_i = np.array([])  # To store ai from the previous iteration
prev_valid_mu = []  # To store valid a from the previous iteration

# Initialize the plot
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots(figsize=(12, 8))  # Adjust figure size
ax.set_xlim()
ax.set_ylim(-0.1,0.5)
ax.set_title("")
ax.set_xlabel("")
ax.set_xticks([])
ax.set_ylabel("μ")

lines_ai = [ax.plot([], [], label=f"ai[{j}]", color="blue")[0] for j in range(2)]
lines_valid_a = [ax.plot([], [], label=f"valid_a[{j}]", color="green", linestyle=":")[0] for j in range(2)]

fill_mu_i = None
fill_valid_mu = None

# Add legend entries for the areas
area_handles = [
    plt.Line2D([0], [0], color="blue", alpha=0.35, lw=10, label=r"$\Delta_k$"),
    plt.Line2D([0], [0], color="green", alpha=0.7, lw=10, label=r"$\Theta_k$")
]

ax.legend(handles= area_handles, loc='upper left')


### --- FUNCTION TO REMOVE DUPLICATES --- ###
def remove_duplicates(values, epsilon=1e-6):
    filtered = []
    for value in values:
        if all(abs(value - f) > epsilon for f in filtered):
            filtered.append(value)
    return np.array(sorted(filtered))

### --- SME ALGORITHM --- ###


for i in range(700, len(df)):

    if fill_mu_i:
        fill_mu_i.remove()
    
    if fill_valid_mu:
        fill_valid_mu.remove()

    fill_mu_i = None
    fill_valid_mu = None

    # STATE VECTOR
    z_i = np.array([[vx[i]], [vy[i]], [w[i]]])

    # LATERAL DYNAMICS
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


    Ai = -H @ G_i_minus1
    bi = h_d - H @ z_i + H @ F_i_minus1

    mu_i = np.unique(bi / Ai)
    mu_i = np.sort(mu_i)
    # print(f"mu_i = {mu_i}")

    A = np.concatenate([Ai, Ai_minus1]) + 1e-6  # Add tolerance to avoid dividing by zero
    b = np.concatenate([bi, bi_minus1])

    ### METHOD 1 ###
    mu_values = np.where(A != 0, b / A, np.inf)

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
            # valid_mu.append(mu)
            valid_mu.append(max(mu, 0))  # Ensure non-negative mu

    # Remove duplicates in valid_mu
    valid_mu = remove_duplicates(valid_mu, epsilon=1e-6)
    valid_mu = np.sort(valid_mu)

    print(f"Iteration: {i}, valid_mu: [{valid_mu[0]:.4f}, {valid_mu[1]:.4f}]")

    # Update Ai_minus1 and bi_minus1 with the valid values for the next iteration
    if valid_A and valid_b:  # Only update if valid values exist
        Ai_minus1 = np.vstack(valid_A)  # Stack all valid A
        bi_minus1 = np.vstack(valid_b)  # Stack all valid b

    x_vals = np.linspace(0,1, 10)

    if len(mu_i) >= 2:
        mu_i_up = mu_i[3]
        mu_i_low = mu_i[2]
        line_up_i = mu_i_up * np.ones(10)
        line_low_i = mu_i_low * np.ones(10)
        fill_mu_i = ax.fill_between(x_vals, line_low_i, line_up_i, alpha=0.35, color='blue')



    if len(valid_mu) == 2:
        mu_up = valid_mu[1]
        mu_low = valid_mu[0]
        line_up = mu_up * np.ones(10)
        line_low = mu_low * np.ones(10)
        fill_valid_mu = ax.fill_between(x_vals, line_low, line_up, alpha= 0.7, color='green')
    
    
    ax.set_title(f"$μ(i) \\; \\in \\; [{mu_low:.3f}, {mu_up:.3f}]$\n"
                f"Iteration number: {i}")

    plt.pause(0.00001)

plt.ioff()
plt.show()