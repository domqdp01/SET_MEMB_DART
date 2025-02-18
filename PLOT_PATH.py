import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

# File path for the dataset
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_18_2024_14_43_23.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_18_2024_14_52_47.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_10_2025_14_35_50.csv'  # noise_up = 0.05
file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_10_2025_14_39_57.csv'  # noise_up = 0.5
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_10_2025_14_50_47.csv'  # noise_up = 1
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_10_2025_17_07_52.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_10_2025_17_12_50.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_16_41_15.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_16_58_58.csv' # unifrom noise
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_17_04_25.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_17_10_42.csv' # uniform nosi not centered
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_18_2024_14_43_23.csv'

# Load the dataset
df = pd.read_csv(file_path)

# Extract relevant data columns

vx = df['vel encoder'].values           # Velocity from encoder in the x direction
vy = df['velocity y'].values           # Velocity in the y direction
x = df['vicon x'].values               # Position in x
y = df['vicon y'].values               # Position in y
t = df['elapsed time sensors'].values  # Elapsed time from sensors
yaw = df['vicon yaw'].values           # Yaw angle from vicon
w = df['W (IMU)'].values               # Angular velocity (W) from IMU

# Create the figure and define a custom layout
gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
fig = plt.figure(figsize=(10, 8))

# First subplot: Trajectory plot (occupies the entire first row)
ax1 = fig.add_subplot(gs[0, :])  # Spans all columns in the first row
ax1.plot(x, y, label='Trajectory', zorder=1)
ax1.scatter(x[0], y[0], color='red', s=30, label='Starting point', zorder=2)
ax1.scatter(x[-1], y[-1], color='black', s=30, label='Ending point', zorder=2)
ax1.set_title("CAR SIMULATOR TRAJECTORY", fontsize=14)
ax1.set_xlabel("X [m]", fontsize=12)
ax1.set_ylabel("Y [m]", fontsize=12)
ax1.legend(loc='best')

# Second subplot: Yaw angle vs. time
top_ax2 = fig.add_subplot(gs[1, 0])
top_ax2.plot(t, yaw, label='Yaw')
top_ax2.set_xlabel("t [s]", fontsize=12)
top_ax2.set_ylabel("YAW [rad]", fontsize=12)
top_ax2.legend(loc='best')

# Third subplot: Angular velocity (W) vs. time
top_ax3 = fig.add_subplot(gs[1, 1])
top_ax3.plot(t, w, label='W')
top_ax3.set_xlabel("t [s]", fontsize=12)
top_ax3.set_ylabel("W [rad/s]", fontsize=12)
top_ax3.legend(loc='best')

# Fourth subplot: Velocity in the x direction vs. time
bottom_ax4 = fig.add_subplot(gs[2, 0])
bottom_ax4.plot(t, vx, label='Vx')
bottom_ax4.set_xlabel("t [s]", fontsize=12)
bottom_ax4.set_ylabel("vx [m/s]", fontsize=12)
bottom_ax4.legend(loc='best')

# Fifth subplot: Velocity in the y direction vs. time
bottom_ax5 = fig.add_subplot(gs[2, 1])
bottom_ax5.plot(t, vy, label='Vy')
bottom_ax5.set_xlabel("t [s]", fontsize=12)
bottom_ax5.set_ylabel("vy [m/s]", fontsize=12)
bottom_ax5.legend(loc='best')

# Optimize layout to prevent overlap
plt.tight_layout()
plt.show()

# plt.subplot(2,1,1)
# plt.plot(t, yaw)
# plt.subplot(2,1,2)
# plt.plot(t,w)
# plt.show()