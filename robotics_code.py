import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, solve
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Ορισμός μηκών συνδέσμων
link_base = 81.0
link1 = 20.0
link2 = 60.0
link4 = 14.0
link5 = 55.0
link7 = 10.0

# Αρχικές θέσεις στόχων A και B
pos_A = np.array([0, 20, 120])
pos_B = np.array([50, 40, 120])

# Χρονικά διαστήματα
total_time = 400.0
half_time = total_time / 2.0
delay = half_time * 0.25
active_time = half_time - 2 * delay
max_velocity_x = (pos_B[0] - pos_A[0]) / (half_time - delay)
max_velocity_y = (pos_B[1] - pos_A[1]) / (half_time - delay)

# Διακριτικοί χρόνοι
num_samples = 7000
phase1_time = np.linspace(0, delay, num_samples, endpoint=True)
phase2_time = np.linspace(delay, delay + active_time, num_samples, endpoint=True)
phase3_time = np.linspace(delay + active_time, half_time, num_samples, endpoint=True)

def solve_polynomial(start_pos, end_pos, start_vel, end_vel, t_start, t_end):
    a4, a3, a2, a1, a0 = symbols('a4 a3 a2 a1 a0', real=True)

    # Συνθήκες στην αρχή και στο τέλος
    pos_start = a4*t_start**4 + a3*t_start**3 + a2*t_start**2 + a1*t_start + a0
    pos_end = a4*t_end**4 + a3*t_end**3 + a2*t_end**2 + a1*t_end + a0

    vel_start = 4*a4*t_start**3 + 3*a3*t_start**2 + 2*a2*t_start + a1
    vel_end = 4*a4*t_end**3 + 3*a3*t_end**2 + 2*a2*t_end + a1

    accel_start = 12*a4*t_start**2 + 6*a3*t_start + 2*a2

    # Σύστημα εξισώσεων
    equations = [
        pos_start - start_pos,
        pos_end - end_pos,
        vel_start - start_vel,
        vel_end - end_vel,
        accel_start  # Επιπλέον συνθήκη για επιτάχυνση μηδέν στην αρχή
    ]

    λύση = solve(equations, (a4, a3, a2, a1, a0), dict=True)
    if not λύση:
        raise ValueError("Δεν βρέθηκε λύση για τα πολυωνυμικά συστατικά.")
    
    συντελεστές = [float(λύση[0][var]) for var in (a4, a3, a2, a1, a0)]
    return συντελεστές

# Υπολογισμός συντελεστών για τη φάση 1 (αύξηση ταχύτητας)
coeffs_x_phase1 = solve_polynomial(
    start_pos=pos_A[0],
    end_pos=pos_A[0] + (pos_B[0] - pos_A[0]) * (delay / half_time) * 0.6,
    start_vel=0,
    end_vel=max_velocity_x,
    t_start=0,
    t_end=delay
)
x_phase1 = np.polyval(coeffs_x_phase1, phase1_time)
vel_x_phase1 = np.polyval(np.polyder(coeffs_x_phase1), phase1_time)

coeffs_y_phase1 = solve_polynomial(
    start_pos=pos_A[1],
    end_pos=pos_A[1] + (pos_B[1] - pos_A[1]) * (delay / half_time) * 0.6,
    start_vel=0,
    end_vel=max_velocity_y,
    t_start=0,
    t_end=delay
)
y_phase1 = np.polyval(coeffs_y_phase1, phase1_time)
vel_y_phase1 = np.polyval(np.polyder(coeffs_y_phase1), phase1_time)

# Φάση 2: Κινήση με σταθερή ταχύτητα
x_phase2 = x_phase1[-1] + vel_x_phase1[-1] * (phase2_time - phase2_time[0])
vel_x_phase2 = np.full_like(phase2_time, vel_x_phase1[-1])

y_phase2 = y_phase1[-1] + vel_y_phase1[-1] * (phase2_time - phase2_time[0])
vel_y_phase2 = np.full_like(phase2_time, vel_y_phase1[-1])

# Φάση 3: Μείωση ταχύτητας
coeffs_x_phase3 = solve_polynomial(
    start_pos=x_phase2[-1],
    end_pos=pos_B[0],
    start_vel=vel_x_phase2[-1],
    end_vel=0,
    t_start=delay + active_time,
    t_end=half_time
)
x_phase3 = np.polyval(coeffs_x_phase3, phase3_time)
vel_x_phase3 = np.polyval(np.polyder(coeffs_x_phase3), phase3_time)

coeffs_y_phase3 = solve_polynomial(
    start_pos=y_phase2[-1],
    end_pos=pos_B[1],
    start_vel=vel_y_phase2[-1],
    end_vel=0,
    t_start=delay + active_time,
    t_end=half_time
)
y_phase3 = np.polyval(coeffs_y_phase3, phase3_time)
vel_y_phase3 = np.polyval(np.polyder(coeffs_y_phase3), phase3_time)

# Συνένωση φάσεων για κίνηση από A προς B
x_AB = np.concatenate((x_phase1, x_phase2, x_phase3))
vel_x_AB = np.concatenate((vel_x_phase1, vel_x_phase2, vel_x_phase3))
time_AB = np.concatenate((phase1_time, phase2_time, phase3_time))

y_AB = np.concatenate((y_phase1, y_phase2, y_phase3))
vel_y_AB = np.concatenate((vel_y_phase1, vel_y_phase2, vel_y_phase3))

# Κίνηση από B προς A (αντίστροφη)
x_BA = x_AB[::-1]
vel_x_BA = -vel_x_AB[::-1]
time_BA = time_AB + time_AB[-1]

y_BA = y_AB[::-1]
vel_y_BA = -vel_y_AB[::-1]

# Συνολική κίνηση και χρόνοι
x_total = np.concatenate((x_AB, x_BA))
vel_x_total = np.concatenate((vel_x_AB, vel_x_BA))
time_total_arr = np.concatenate((time_AB, time_BA))

y_total = np.concatenate((y_AB, y_BA))
vel_y_total = np.concatenate((vel_y_AB, vel_y_BA))

# Σταθερό Z
z_total = np.full_like(x_total, pos_A[2])

# Υπολογισμός ταχυτήτων με βάση τις θέσεις
velocity_x = np.gradient(x_total, time_total_arr)
velocity_y = np.gradient(y_total, time_total_arr)
velocity_z = np.gradient(z_total, time_total_arr)

def inverse_kinematics(x, y, z, base, link1, link2, link4, link5, link7):
    theta1 = np.arctan2(y, x)
    
    d = np.sqrt((link5 + link7)**2 + link4**2)
    alpha = np.arctan2(link5 + link7, link4)
    
    numerator = (z - base)**2 + (y / np.sin(theta1) - link1)**2 - link2**2 - d**2
    denominator = 2 * link2 * d
    argument = numerator / denominator
    argument = np.clip(argument, -1.0, 1.0)
    
    theta3 = np.arccos(argument) - alpha
    beta = np.arctan2(d * np.sin(theta3 + alpha), (link2 + d * np.cos(theta3 + alpha)))
    
    theta2 = np.arctan2((y / np.sin(theta1) - link1), (z - base)) - beta
    
    return theta1, theta2, theta3

def forward_kinematics(theta1, theta2, theta3, base, link1, link2, link4, link5, link7):
    joints = []
    
    # Βάση
    joints.append((0, 0, 0))
    
    # Υψόμετρο βάσης
    joints.append((0, 0, base))
    
    # Άρση πρώτου άρθρωσης
    x1 = link1 * np.cos(theta1)
    y1 = link1 * np.sin(theta1)
    z1 = base
    joints.append((x1, y1, z1))
    
    # Άρση δεύτερου άρθρωσης
    x2 = x1 + link2 * np.cos(theta1) * np.sin(theta2)
    y2 = y1 + link2 * np.sin(theta1) * np.sin(theta2)
    z2 = z1 + link2 * np.cos(theta2)
    joints.append((x2, y2, z2))
    
    # Άρση τρίτου άρθρωσης
    x3 = x2 + link4 * np.cos(theta1) * np.sin(theta2 + theta3)
    y3 = y2 + link4 * np.sin(theta1) * np.sin(theta2 + theta3)
    z3 = z2 + link4 * np.cos(theta2 + theta3)
    joints.append((x3, y3, z3))
    
    # Τέλος χεριού
    xe = x3 + (link5 + link7) * np.cos(theta1) * np.cos(theta2 + theta3)
    ye = y3 + (link5 + link7) * np.sin(theta1) * np.cos(theta2 + theta3)
    ze = z3 - (link5 + link7) * np.sin(theta2 + theta3)
    joints.append((xe, ye, ze))
    
    return joints

# Υπολογισμός γωνιών άρθρωσης για κάθε χρονική στιγμή
angles_q1 = []
angles_q2 = []
angles_q3 = []

for xi, yi, zi in zip(x_total, y_total, z_total):
    q1, q2, q3 = inverse_kinematics(xi, yi, zi, link_base, link1, link2, link4, link5, link7)
    angles_q1.append(q1)
    angles_q2.append(q2)
    angles_q3.append(q3)

angles_q1 = np.array(angles_q1)
angles_q2 = np.array(angles_q2)
angles_q3 = np.array(angles_q3)

# Υπολογισμός θέσεων αρθρώσεων
all_joints = []
for q1, q2, q3 in zip(angles_q1, angles_q2, angles_q3):
    joints = forward_kinematics(q1, q2, q3, link_base, link1, link2, link4, link5, link7)
    all_joints.append(joints)

all_joints = np.array(all_joints, dtype=object)

def compute_joint_velocities(q1, q2, q3, vx, vy, vz):
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    c23, s23 = np.cos(q2 + q3), np.sin(q2 + q3)
    
    # Ορισμός μητρώου Jacobian
    J = np.zeros((3, 3))
    J[0, 0] = -link1 * s1 - s1 * c23 * (link5 + link7) - link2 * s1 * s2 - link4 * s1 * s23
    J[0, 1] = c1 * link2 * c2 - c1 * s23 * (link5 + link7) + c1 * link4 * c23
    J[0, 2] = c1 * link4 * c23 - c1 * s23 * (link5 + link7)
    
    J[1, 0] = link1 * c1 + c1 * c23 * (link5 + link7) + link2 * c1 * s2 + link4 * c1 * s23
    J[1, 1] = s1 * link2 * c2 - s1 * s23 * (link5 + link7) + s1 * link4 * c23
    J[1, 2] = s1 * link4 * c23 - s1 * s23 * (link5 + link7)
    
    J[2, 1] = -link2 * s2 - c23 * (link5 + link7) - link4 * s23
    J[2, 2] = -c23 * (link5 + link7) - link4 * s23
    
    # Δημιουργία διανύσματος ταχυτήτων end-effector
    end_eff_vel = np.array([vx, vy, vz]).reshape(-1, 1)
    
    # Υπολογισμός ταχυτήτων αρθρώσεων χρησιμοποιώντας Moore-Penrose pseudoinverse
    joint_vel = np.linalg.pinv(J) @ end_eff_vel
    return joint_vel.flatten()

# Υπολογισμός ταχυτήτων αρθρώσεων για κάθε χρονική στιγμή
velocities_joints = np.zeros((3, len(time_total_arr)))

for i in range(len(time_total_arr)):
    velocities_joints[:, i] = compute_joint_velocities(
        angles_q1[i], angles_q2[i], angles_q3[i],
        velocity_x[i], velocity_y[i], velocity_z[i]
    )

q1_dot = velocities_joints[0, :]
q2_dot = velocities_joints[1, :]
q3_dot = velocities_joints[2, :]

# Σχεδίαση θέσεων end-effector στον χρόνο
plt.figure(figsize=(12, 5))
plt.plot(time_total_arr, x_total, label="x(t)", color='deeppink')
plt.plot(time_total_arr, y_total, label="y(t)", color='blue')
plt.plot(time_total_arr, z_total, label="z(t)", color='skyblue')
plt.xlabel("Χρόνος (s)")
plt.ylabel("Θέση (cm)")
plt.title("Θέση End-Effector συναρτήσει του χρόνου")
plt.grid(True)
plt.legend()
plt.show()

# Σχεδίαση ταχυτήτων end-effector στον χρόνο
plt.figure(figsize=(12, 5))
plt.plot(time_total_arr, velocity_x, label="vx(t)", color='limegreen')
plt.plot(time_total_arr, velocity_y, label="vy(t)", color='blue')
plt.plot(time_total_arr, velocity_z, label="vz(t)", color='deeppink')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel("Χρόνος (s)")
plt.ylabel("Ταχύτητα (cm/s)")
plt.title("Γραμμική ταχύτητα End-Effector συναρτήσει του χρόνου")
plt.grid(True)
plt.legend()
plt.show()

# Σχεδίαση γωνιών αρθρώσεων στον χρόνο
plt.figure(figsize=(12, 6))
plt.plot(time_total_arr, np.degrees(angles_q1), label='$q_1$ (deg)', color='deeppink')
plt.plot(time_total_arr, np.degrees(angles_q2), label='$q_2$ (deg)', color='blue')
plt.plot(time_total_arr, np.degrees(angles_q3), label='$q_3$ (deg)', color='skyblue')
plt.xlabel('Χρόνος (s)')
plt.ylabel('Γωνίες Αρθρώσεων (degrees)')
plt.title('Γωνίες Αρθρώσεων συνατήσει του χρόνου')
plt.legend()
plt.grid(True)
plt.show()

# Σχεδίαση γωνιακών ταχυτήτων αρθρώσεων στον χρόνο
plt.figure(figsize=(12, 6))
plt.plot(time_total_arr, np.degrees(q1_dot), label='$\dot{q}_1$ (deg/s)', color='limegreen')
plt.plot(time_total_arr, np.degrees(q2_dot), label='$\dot{q}_2$ (deg/s)', color='blue')
plt.plot(time_total_arr, np.degrees(q3_dot), label='$\dot{q}_3$ (deg/s)', color='deeppink')
plt.xlabel('Χρόνος (s)')
plt.ylabel('Γωνιακές Ταχύτητες Αρθρώσεων (degrees/s)')
plt.title('Γωνιακές Ταχύτητες Αρθρώσεων συνατήσει του χρόνου')
plt.legend()
plt.grid(True)
plt.show()

# Δημιουργία 3D γραφικής παράστασης της κίνησης του βραχίονα
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Ρυθμίσεις αξόνων
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 150)
ax.set_xlabel('X (cm)')
ax.set_ylabel('Y (cm)')
ax.set_zlabel('Z (cm)')
ax.set_title('Κίνηση Ρομποτικού Βραχίονα')

# Αντικείμενα για την κινούμενη γραμμή και το scatter
trajectory_segments = []
end_effector = ax.scatter([], [], [], c='deeppink', label='End-Effector')
plt.legend()

# Παράγοντας αραιότητας για την animation
sparsity = 800

def animate(frame):
    global trajectory_segments

    if frame == 0:
        # Αφαίρεση προηγούμενων γραμμών
        for segment in trajectory_segments:
            segment.remove()
        trajectory_segments.clear()

    # Υπολογισμός τρέχοντος frame με βάση τον παράγοντα αραιότητας
    current_idx = frame * sparsity
    if current_idx >= len(all_joints):
        return []

    # Λήψη θέσεων αρθρώσεων
    joints = all_joints[current_idx]

    # Καθορισμός χρώματος γραμμής ανά φάση κίνησης
    midpoint = len(all_joints) // 2
    color = 'limegreen' if current_idx >= midpoint else 'g'

    # Σχεδίαση γραμμών μεταξύ αρθρώσεων
    for i in range(len(joints) - 1):
        xs = [joints[i][0], joints[i + 1][0]]
        ys = [joints[i][1], joints[i + 1][1]]
        zs = [joints[i][2], joints[i + 1][2]]
        seg, = ax.plot(xs, ys, zs, color=color, linewidth=2)
        trajectory_segments.append(seg)

    # Ενημέρωση θέσης end-effector
    end_effector._offsets3d = (
        x_total[:current_idx:sparsity],
        y_total[:current_idx:sparsity],
        z_total[:current_idx:sparsity]
    )
    return trajectory_segments + [end_effector]

# Δημιουργία animation
total_frames = len(all_joints) // sparsity
animation = FuncAnimation(fig, animate, frames=total_frames, interval=100, blit=False)

plt.show()