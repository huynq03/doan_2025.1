import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import os

# ==== Parameters ====
dt = 0.02
m_q, m_g, g = 0.5, 0.158, 9.81
l_p, l_q = 0.35, 0.2
J_q, J_g, L_g = 0.15, 0.0, 0.35

# ==== Dynamics ====
def jax_dynamics_matrix(state, control, dt=dt):
    y, y_dot, z, z_dot, phi, phi_dot, beta, beta_dot = state
    u1, u2, tau = control
    M = m_q + m_g
    s, c = jnp.sin(beta), jnp.cos(beta)
    D = jnp.array([
        [M, 0, 0, -L_g*m_g*s],
        [0, M, 0, -L_g*m_g*c],
        [0, 0, J_q, 0],
        [-L_g*m_g*s, -L_g*m_g*c, 0, J_g + L_g**2 * m_g]
    ], dtype=state.dtype)
    C = jnp.array([
        [0, 0, 0, -L_g*m_g*c * beta_dot],
        [0, 0, 0,  L_g*m_g*s * beta_dot],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=state.dtype)
    G = jnp.array([0, g*M, 0, -g*L_g*m_g*c], dtype=state.dtype)
    F = jnp.array([u1*jnp.sin(phi), u1*jnp.cos(phi), u2-tau, tau], dtype=state.dtype)
    qdot = jnp.array([y_dot, z_dot, phi_dot, beta_dot], dtype=state.dtype)
    rhs = F - C @ qdot - G
    qddot = jnp.array(np.linalg.solve(np.array(D), np.array(rhs)), dtype=state.dtype)
    y_ddot, z_ddot, phi_ddot, beta_ddot = qddot
    state_dot = jnp.array([y_dot, y_ddot, z_dot, z_ddot, phi_dot, phi_ddot, beta_dot, beta_ddot], dtype=state.dtype)
    return state + state_dot * dt

# ==== Load Controls ====
def load_controls_from_csv(filename):
    """Đọc u1, u2, u3 từ CSV file"""
    u1s, u2s, u3s = [], [], []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            u1s.append(float(row['u1']))
            u2s.append(float(row['u2']))
            u3s.append(float(row['u3']))
    controls = np.array(list(zip(u1s, u2s, u3s)))
    print(f"✓ Đã load {len(controls)} bước điều khiển")
    return controls

# ==== Simulation ====
def simulate_from_controls(controls_array, dt=dt, initial_state=None):
    """Chạy simulation với mảng controls có sẵn"""
    if initial_state is None:
        state = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        state = initial_state.copy()
    
    states = [state]
    for i in range(len(controls_array)):
        u1, u2, u3 = controls_array[i]
        u = jnp.array([u1, u2, u3], dtype=jnp.float32)
        state = np.array(jax_dynamics_matrix(state, u))
        states.append(state)
    
    print(f"✓ Đã simulate {len(controls_array)} bước")
    return np.array(states)

# ==== Visualization ====
def animate(
    states,
    controls,
    dt=dt,
    save_image=True,
    save_frame_time=None,
    save_all_frames=False,
    frames_dir="frames",
):
    scale_draw = 2
    l_q_vis = l_q * scale_draw
    l_p_vis = l_p * scale_draw
    L_finger = 0.10 * scale_draw
    offset = 0.05 * scale_draw
    lw_body = 5 * scale_draw
    lw_pend = 2 * scale_draw
    lw_finger = 2 * scale_draw
    lw_trail = 1 * scale_draw
    lw_thrust = 2 * scale_draw
    thrust_scale = 0.04 * scale_draw
    thrust_base = 0.08 * scale_draw

    y, z, phi, theta = states[:,0], states[:,2], states[:,4], states[:,6]

    fig, ax = plt.subplots(figsize=(10*scale_draw/2, 10*scale_draw/2), dpi=120)
    margin = 1.0
    ax.set_xlim(y.min()-margin, y.max()+margin)
    ax.set_ylim(z.min()-margin, z.max()+margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    frame_line, = ax.plot([], [], "k", lw=lw_body)
    tether_line, = ax.plot([], [], "gray", lw=lw_pend)
    trail, = ax.plot([], [], "b-", lw=lw_trail, alpha=0.6)
    left_line, = ax.plot([], [], "r", lw=lw_finger)
    right_line, = ax.plot([], [], "r", lw=lw_finger)
    left_thrust_line, = ax.plot([], [], color="orange", lw=lw_thrust)
    right_thrust_line, = ax.plot([], [], color="orange", lw=lw_thrust)

    def rotor_forces(u1, u2, arm):
        fL = 0.5 * (u1 + u2 / max(1e-9, arm))
        fR = 0.5 * (u1 - u2 / max(1e-9, arm))
        return max(0.0, fL), max(0.0, fR)
    
    # Giới hạn góc quay gripper
    GRIPPER_ANGLE_MIN = -np.pi/4  # -45°
    GRIPPER_ANGLE_MAX = np.pi/4   # +45°

    def update(i):
        j = i if i < len(controls) else len(controls) - 1
        u1c, u2c, _ = controls[j]
        fL, fR = rotor_forces(u1c, u2c, l_q)
        yc, zc, phic, thetac = y[i], z[i], phi[i], theta[i]

        c, s = np.cos(phic), np.sin(phic)
        R_body = np.array([[c, s], [-s, c]])
        T = np.array([[yc, yc], [zc, zc]])

        main = np.array([[-l_q_vis, l_q_vis], [0.0, 0.0]])
        body = R_body @ main + T
        frame_line.set_data(body[0], body[1])

        # Giới hạn góc gripper
        ang = phic + thetac
        ang = np.clip(ang, GRIPPER_ANGLE_MIN, GRIPPER_ANGLE_MAX)
        pend_w = np.array([[0.0, l_p_vis*np.sin(ang)], [0.0, -l_p_vis*np.cos(ang)]])
        pend_w = pend_w + np.array([[yc, yc], [zc, zc]])
        tether_line.set_data(pend_w[0], pend_w[1])

        end_x, end_y = pend_w[0,1], pend_w[1,1]
        vx, vy = np.sin(ang), -np.cos(ang)
        nx, ny = -vy, vx
        dx, dy = vx * L_finger, vy * L_finger
        left_line.set_data([end_x + nx*offset, end_x + nx*offset + dx],
                          [end_y + ny*offset, end_y + ny*offset + dy])
        right_line.set_data([end_x - nx*offset, end_x - nx*offset + dx],
                           [end_y - ny*offset, end_y - ny*offset + dy])

        left_bar_local = np.array([[-l_q_vis, -l_q_vis], [thrust_base, thrust_base + thrust_scale * fL]])
        right_bar_local = np.array([[l_q_vis, l_q_vis], [thrust_base, thrust_base + thrust_scale * fR]])
        left_bar = R_body @ left_bar_local + T
        right_bar = R_body @ right_bar_local + T
        left_thrust_line.set_data(left_bar[0], left_bar[1])
        right_thrust_line.set_data(right_bar[0], right_bar[1])

        trail.set_data(y[:i+1], z[:i+1])

        return (frame_line, tether_line, trail, left_line, right_line, left_thrust_line, right_thrust_line)

    ani = FuncAnimation(fig, update, frames=len(states), interval=dt*1000, blit=True)
    
    # Lưu frame tại thời điểm chỉ định
    if save_frame_time is not None:
        idx = min(int(round(save_frame_time / dt)), len(states) - 1)
        update(idx)
        fig.canvas.draw()
        fname = f"simulation_t{save_frame_time:.2f}s.png"
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"✓ Đã lưu hình tại t={save_frame_time:.2f}s: {fname}")

    # Lưu tất cả các frame nếu cần
    if save_all_frames and save_image:
        os.makedirs(frames_dir, exist_ok=True)
        total_frames = len(states)
        for i in range(total_frames):
            update(i)
            fig.canvas.draw()
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
            fig.savefig(frame_path, dpi=150, bbox_inches='tight')
        print(f"✓ Đã lưu {total_frames} frame vào thư mục: {frames_dir}")

    # Lưu frame cuối cùng thành ảnh
    if save_image:
        fig.canvas.draw()
        fig.savefig('simulation_final_frame.png', dpi=150, bbox_inches='tight')
        print(f"✓ Đã lưu hình ảnh frame cuối: simulation_final_frame.png")
    
    try:
        plt.show()
    except:
        pass
    finally:
        plt.close(fig)

# ==== Save Trajectory ====
def save_trajectory(states, controls, filename="trajectory_output.csv"):
    """Lưu quỹ đạo vào file CSV"""
    import csv
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['time', 'y', 'y_dot', 'z', 'z_dot', 'phi', 'phi_dot', 'beta', 'beta_dot', 'u1', 'u2', 'u3'])
        # Data
        for i, state in enumerate(states):
            t = i * dt
            u1, u2, u3 = controls[i] if i < len(controls) else controls[-1]
            writer.writerow([f'{t:.4f}'] + [f'{val:.6f}' for val in state] + [f'{u1:.6f}', f'{u2:.6f}', f'{u3:.6f}'])
    print(f"✓ Đã lưu quỹ đạo vào: {filename}")

# ==== Main ====
def main():
    # Load controls từ CSV
    controls = load_controls_from_csv("thuan.csv")
    
    # Mô phỏng
    states = simulate_from_controls(controls)
    
    # Lưu quỹ đạo
    # save_trajectory(states, controls, "trajectory_sim.csv")
    
    # Hiển thị animation
    animate(states, controls, save_image=True, save_frame_time=2.50, save_all_frames=True, frames_dir="frames_all")

if __name__ == "__main__":
    main()