import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# ==== Parameters ====
dt = 0.02
m_q, m_g, g = 0.5, 0.158, 9.81
l_p, l_q = 0.35, 0.2
J_q, J_g, L_g = 0.15, 0.001, 0.35

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
    qddot = jnp.array(np.linalg.solve(np.array(D), np.array(rhs)), dtype=state.dtype) # D @ qddot = rhs => qddot = D^-1 @ rhs
    y_ddot, z_ddot, phi_ddot, beta_ddot = qddot
    state_dot = jnp.array([y_dot, y_ddot, z_dot, z_ddot, phi_dot, phi_ddot, beta_dot, beta_ddot], dtype=state.dtype)
    return state + state_dot * dt

# ==== Visualization ====
def animate(states, controls, target=(5.0, 5.0), dt=dt, save_frames=False, output_dir="../media/animation_frames", playback_speed=2.0):
    # ======= Tham số hiển thị (không ảnh hưởng dynamics) =======
    scale_draw   = 2
    l_q_vis      = l_q * scale_draw
    l_p_vis      = l_p * scale_draw
    L_finger     = 0.10 * scale_draw
    offset       = 0.05 * scale_draw
    lw_body      = 5 * scale_draw
    lw_pend      = 2 * scale_draw
    lw_finger    = 2 * scale_draw
    lw_trail     = 1 * scale_draw
    lw_thrust    = 2 * scale_draw
    thrust_scale = 0.04 * scale_draw
    thrust_base  = 0.08 * scale_draw

    # ======= Chuẩn bị lưu frames =======
    if save_frames:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        num_frames_to_save = 40
        total_frames = len(states)
        frame_indices = np.linspace(0, total_frames - 1, num_frames_to_save, dtype=int)
        print(f"[INFO] Sẽ lưu {num_frames_to_save} frames vào {output_dir}")
        print(f"[INFO] Frame indices: {frame_indices.tolist()}")
    
    # ======= Dữ liệu trạng thái =======
    y, z, phi, beta = states[:,0], states[:,2], states[:,4], states[:,6]
    
    # ======= Tính chỉ số tại t=2s =======
    time_array = np.arange(len(states)) * dt
    target_time = 2.0
    idx_target = np.argmin(np.abs(time_array - target_time))
    print(f"[INFO] Chỉ số tại t={target_time}s: {idx_target}, thời gian thực tế: {time_array[idx_target]:.3f}s")
    print(f"[INFO] Vị trí tại t={target_time}s: y={y[idx_target]:.3f}, z={z[idx_target]:.3f}")

    # --- TẠO 2 KHUNG HÌNH RỘC BIỆT ---
    # fig1, ax1 = plt.subplots(figsize=(10, 10), dpi=120)
    # fig1.suptitle('Quadrotor Simulation: Camera Tracking')
    
    fig2, ax2 = plt.subplots(figsize=(10, 10), dpi=120)
    fig2.suptitle("Mô phỏng quỹ đạo thực tế")

    # Cấu hình ax1 (Camera Tracking)
    # ax1.set_aspect("equal"); ax1.grid(True, alpha=0.3)
    # ax1.set_title("Camera Tracking")
    # ax1.add_patch(plt.Circle(target, 0.1*scale_draw, color="g", fill=False))

    # Cấu hình ax2 (Full View)
    ax2.set_xlim(min(y.min(), target[0])-1, max(y.max(), target[0])+1)
    ax2.set_ylim(min(z.min(), target[1])-1, max(z.max(), target[1])+1)
    ax2.set_aspect("equal"); ax2.grid(True, alpha=0.3)
    ax2.add_patch(plt.Circle(target, 0.1 , color="g", fill=False))

    # Marker sẽ bám vào gripper kể từ t >= 2s
    grip_marker, = ax2.plot([], [], marker='o', markersize=12, color='blue')
    ax2.legend(fontsize=10, loc='best')

    # --- Đối tượng vẽ ---
    # frame_line1,  = ax1.plot([], [], "k", lw=lw_body)
    # tether_line1, = ax1.plot([], [], "gray", lw=lw_pend)
    # trail1,       = ax1.plot([], [], "b-", lw=lw_trail, alpha=0.6)
    # left_line1,   = ax1.plot([], [], "r", lw=lw_finger)
    # right_line1,  = ax1.plot([], [], "r", lw=lw_finger)
    # left_thrust_line1,  = ax1.plot([], [], color="orange", lw=lw_thrust)
    # right_thrust_line1, = ax1.plot([], [], color="orange", lw=lw_thrust)

    frame_line2,  = ax2.plot([], [], "k", lw=lw_body)
    tether_line2, = ax2.plot([], [], "gray", lw=lw_pend)
    trail2,       = ax2.plot([], [], "b-", lw=lw_trail, alpha=0.6)
    left_line2,   = ax2.plot([], [], "r", lw=lw_finger)
    right_line2,  = ax2.plot([], [], "r", lw=lw_finger)
    left_thrust_line2,  = ax2.plot([], [], color="orange", lw=lw_thrust)
    right_thrust_line2, = ax2.plot([], [], color="orange", lw=lw_thrust)

    def rotor_forces(u1, u2, arm):
        fR = 0.5 * (u1 + u2 / max(1e-9, arm))
        fL = 0.5 * (u1 - u2 / max(1e-9, arm))
        return max(0.0, fL), max(0.0, fR)

    def update(i):
        j = i if i < len(controls) else len(controls) - 1
        u1c, u2c, _ = controls[j]
        fL, fR = rotor_forces(u1c, u2c, l_q)

        yc, zc, phic, betac = y[i], z[i], phi[i], beta[i]

        # # Camera Tracking view
        # view_span = 6.0
        # ax1.set_xlim(yc - view_span / 2, yc + view_span / 2)
        # ax1.set_ylim(zc - view_span / 2, zc + view_span / 2)

        c, s = np.cos(phic), np.sin(phic)
        R_body = np.array([[ c,  s],
                           [-s,  c]])
        T = np.array([[yc, yc],
                      [zc, zc]])
        main = np.array([[-l_q_vis,  l_q_vis],
                         [   0.0,       0.0]])
        body = R_body @ main + T

        # Gripper line (chỉ dùng cho fig2)
        ang = betac
        pend_w = np.array([[0.0,                   l_p_vis * np.cos(ang)],
                   [0.0,                 - l_p_vis * np.sin(ang)]]) + np.array([[yc, yc],
                                                   [zc, zc]])

        end_x, end_y = pend_w[0, 1], pend_w[1, 1]
        vx, vy = np.cos(ang), -np.sin(ang)
        nx, ny = -vy, vx
        dx, dy = vx * L_finger, vy * L_finger

        # Thrust bars
        left_bar_local  = np.array([[-l_q_vis, -l_q_vis],
                                    [ thrust_base, thrust_base + thrust_scale * fL]])
        right_bar_local = np.array([[ l_q_vis,  l_q_vis],
                                    [ thrust_base, thrust_base + thrust_scale * fR]])
        left_bar  = R_body @ left_bar_local  + T
        right_bar = R_body @ right_bar_local + T
        # Full view updates
        frame_line2.set_data(body[0], body[1])
        tether_line2.set_data(pend_w[0], pend_w[1])
        left_line2.set_data([end_x + nx*offset, end_x + nx*offset + dx],
                            [end_y + ny*offset, end_y + ny*offset + dy])
        right_line2.set_data([end_x - nx*offset, end_x - nx*offset + dx],
                             [end_y - ny*offset, end_y - ny*offset + dy])
        left_thrust_line2.set_data(left_bar[0],  left_bar[1]) 
        right_thrust_line2.set_data(right_bar[0], right_bar[1])
        trail2.set_data(y[:i+1], z[:i+1])

        # Marker: trước t=2s hiển thị tại vị trí (y,z) ở t=2s; sau đó bám gripper
        if i >= idx_target:
            grip_marker.set_data([end_x], [end_y-0.1])
        else:
            grip_marker.set_data([y[idx_target]], [z[idx_target]-0.8])

        # Lưu frame chỉ từ Full View (fig2) nếu được yêu cầu
        if save_frames and i in frame_indices:
            frame_filename = os.path.join(output_dir, f"frame_{i:04d}.png")
            fig2.savefig(frame_filename, dpi=150, bbox_inches='tight')
            print(f"  Đã lưu: {frame_filename}")

        return (# frame_line1, tether_line1, trail1, left_line1, right_line1,
            # left_thrust_line1, right_thrust_line1,
            frame_line2, tether_line2, trail2, left_line2, right_line2,
            left_thrust_line2, right_thrust_line2, grip_marker)

    interval_ms = max(1.0, (dt * 1000.0) / max(1e-6, playback_speed))
    ani = FuncAnimation(fig2, update, frames=len(states), interval=interval_ms, blit=False)
    plt.show()

# Read minsnap results (chỉ chạy khi được gọi trực tiếp)
if __name__ == "__main__":
    data_path = os.path.join("minsnap_results", "flat_outputs.csv")
    data = np.loadtxt(data_path, delimiter=",", skiprows=1)  # Bỏ qua header
    states = data[:, :8]
    controls = data[:, 8:]