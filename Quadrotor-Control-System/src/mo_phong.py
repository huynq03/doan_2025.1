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
def animate(states, controls, target=(5.0, 5.0), dt=dt):
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

    # ======= Dữ liệu trạng thái =======
    y, z, phi, beta = states[:,0], states[:,2], states[:,4], states[:,6]

    # --- TẠO 2 KHUNG HÌNH CON SONG SONG ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20*scale_draw/2, 10*scale_draw/2), dpi=120)
    fig.suptitle('Quadrotor Simulation: Camera Tracking (Left) vs. Full View (Right)')

    # Cấu hình ax1 (Camera Tracking)
    ax1.set_aspect("equal"); ax1.grid(True, alpha=0.3)
    ax1.set_title("Camera Tracking")
    ax1.add_patch(plt.Circle(target, 0.1*scale_draw, color="g", fill=False))

    # Cấu hình ax2 (Full View)
    ax2.set_xlim(min(y.min(), target[0])-1, max(y.max(), target[0])+1)
    ax2.set_ylim(min(z.min(), target[1])-1, max(z.max(), target[1])+1)
    ax2.set_aspect("equal"); ax2.grid(True, alpha=0.3)
    ax2.set_title("Full View")
    ax2.add_patch(plt.Circle(target, 0.1*scale_draw, color="g", fill=False))

    # --- Đối tượng vẽ ---
    frame_line1,  = ax1.plot([], [], "k", lw=lw_body)
    tether_line1, = ax1.plot([], [], "gray", lw=lw_pend)
    trail1,       = ax1.plot([], [], "b-", lw=lw_trail, alpha=0.6)
    left_line1,   = ax1.plot([], [], "r", lw=lw_finger)
    right_line1,  = ax1.plot([], [], "r", lw=lw_finger)
    left_thrust_line1,  = ax1.plot([], [], color="orange", lw=lw_thrust)
    right_thrust_line1, = ax1.plot([], [], color="orange", lw=lw_thrust)

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

        # Camera Tracking view
        view_span = 6.0
        ax1.set_xlim(yc - view_span / 2, yc + view_span / 2)
        ax1.set_ylim(zc - view_span / 2, zc + view_span / 2)

        c, s = np.cos(phic), np.sin(phic)
        R_body = np.array([[ c,  s],
                           [-s,  c]])
        T = np.array([[yc, yc],
                      [zc, zc]])
        main = np.array([[-l_q_vis,  l_q_vis],
                         [   0.0,       0.0]])
        body = R_body @ main + T
        frame_line1.set_data(body[0], body[1])

        # Gripper line
        ang = betac
        pend_w = np.array([[0.0,                   l_p_vis * np.cos(ang)],
                           [0.0,                 - l_p_vis * np.sin(ang)]]) + np.array([[yc, yc],
                                                                                           [zc, zc]])
        tether_line1.set_data(pend_w[0], pend_w[1])

        end_x, end_y = pend_w[0, 1], pend_w[1, 1]
        vx, vy = np.cos(ang), -np.sin(ang)
        nx, ny = -vy, vx
        dx, dy = vx * L_finger, vy * L_finger
        left_line1.set_data([end_x + nx*offset, end_x + nx*offset + dx],
                            [end_y + ny*offset, end_y + ny*offset + dy])
        right_line1.set_data([end_x - nx*offset, end_x - nx*offset + dx],
                             [end_y - ny*offset, end_y - ny*offset + dy])

        # Thrust bars
        left_bar_local  = np.array([[-l_q_vis, -l_q_vis],
                                    [ thrust_base, thrust_base + thrust_scale * fL]])
        right_bar_local = np.array([[ l_q_vis,  l_q_vis],
                                    [ thrust_base, thrust_base + thrust_scale * fR]])
        left_bar  = R_body @ left_bar_local  + T
        right_bar = R_body @ right_bar_local + T
        left_thrust_line1.set_data(left_bar[0],  left_bar[1])
        right_thrust_line1.set_data(right_bar[0], right_bar[1])

        trail1.set_data(y[:i+1], z[:i+1])

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

        return (frame_line1, tether_line1, trail1, left_line1, right_line1,
                left_thrust_line1, right_thrust_line1,
                frame_line2, tether_line2, trail2, left_line2, right_line2,
                left_thrust_line2, right_thrust_line2)

    ani = FuncAnimation(fig, update, frames=len(states), interval=dt*1000, blit=False)
    plt.show()

# Read minsnap results (chỉ chạy khi được gọi trực tiếp)
if __name__ == "__main__":
    data_path = os.path.join("minsnap_results", "flat_outputs.csv")
    data = np.loadtxt(data_path, delimiter=",", skiprows=1)  # Bỏ qua header
    states = data[:, :8]
    controls = data[:, 8:]