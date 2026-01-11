# control1.py
# -*- coding: utf-8 -*-
"""
PD + Feed-forward controller cho quadrotor + gripper (mặt phẳng x–z),
tuân theo IDETC'13 (Thomas et al., 2013). Các feed-forward u1^d, u3^d, tau^d,
theta^d, thetadot^d lấy từ chuyen_doi.recover_inputs_from_flat (ánh xạ (16)–(31)).

Tham khảo: AVIAN-INSPIRED GRASPING FOR QUADROTOR MICRO UAVS, Eq. (1),(16)–(31).
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
import os

import chuyen_doi as tf 


# đạo hàm y theo t bằng sai phân hữu hạn
def finite_diff(t: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    """Central finite-difference of given order."""
    out = y.astype(float).copy()
    for _ in range(order):
        out = np.gradient(out, t, edge_order=2)
    return out


@dataclass
class Gains:
    # Vòng ngoài (x -> attitude)
    kpx: float = 1.2
    kdx: float = 0.6
    # Cao độ
    kpz: float = 10.0
    kdz: float = 5.5
    # Attitude (theta)
    kp_theta: float = 6.0
    kd_theta: float = 2.5
    # Cánh tay (beta)
    kp_beta: float = 2
    kd_beta: float = 0.6


class PDFFController:
    def __init__(
        self,
        flat_csv: str,
        params: Optional[Dict[str, float]] = None,
        gains: Optional[Gains] = None,
    ):
        # tham số dynamics từ tf module
        self.params: Dict[str, float] = dict(tf.PARAMS if params is None else params)
        self.gains = gains if gains is not None else Gains()

        # Đọc quỹ đạo phẳng từ QP (t, x_q, z_q, beta) từ CSV
        df = pd.read_csv(flat_csv)
        for col in ("t", "x_q", "z_q", "beta"):
            if col not in df.columns:
                raise ValueError(f"Thiếu cột '{col}' trong {flat_csv}")
        self.t       = df["t"].to_numpy(dtype=float) # thời gian lấy từ CSV
        self.x_qd    = df["x_q"].to_numpy(dtype=float) # vị trí x mong muốn từ CSV
        self.z_qd    = df["z_q"].to_numpy(dtype=float) # vị trí z mong muốn từ CSV
        self.beta_d  = df["beta"].to_numpy(dtype=float) # góc gripper mong muốn từ CSV 

        # Đạo hàm mong muốn cho PD
        self.xdot_qd    = finite_diff(self.t, self.x_qd, 1)
        self.zdot_qd    = finite_diff(self.t, self.z_qd, 1)
        self.betadot_d  = finite_diff(self.t, self.beta_d, 1)

        # Feed‑forward từ differential flatness --> control inputs
        ff = tf.recover_inputs_from_flat(self.t, self.x_qd, self.z_qd, self.beta_d, self.params)
        self.u1_d        = ff["u1"].astype(float)
        self.u3_d_paper  = ff["u3"].astype(float)      
        self.tau_d_paper = ff["tau"].astype(float)     
        self.theta_d     = ff["theta"].astype(float)
        self.theta_dot_d = ff["theta_dot"].astype(float)

        self.tau_d = self.tau_d_paper
        self.u3_d  = self.u3_d_paper

        # dt
        self.dt = float(np.mean(np.diff(self.t)))

    def step(self, i: int, meas: Dict[str, float]):
        """
        Một bước điều khiển tại chỉ số i.
        meas : chỉ số đo lường thực tế
        meas cần có:
            x_q, z_q, xdot_q, zdot_q, theta, theta_dot, beta, beta_dot
        Trả về: u1_cmd, u3_cmd, tau_cmd (theo quy ước 'plant')
        """
        g = self.gains # các thông số pd

        # các giá trị mong muốn tại i
        x_d, z_d   = self.x_qd[i], self.z_qd[i]
        xd_d, zd_d = self.xdot_qd[i], self.zdot_qd[i]
        th_d, thd_d = self.theta_d[i], self.theta_dot_d[i]
        beta_d, betad_d = self.beta_d[i], self.betadot_d[i]
        u1_ff, u3_ff, tau_ff = self.u1_d[i], self.u3_d[i], self.tau_d[i]

        # Sai lệch
        ex   = x_d - float(meas["x_q"])
        ez   = z_d - float(meas["z_q"])
        exd  = xd_d - float(meas["xdot_q"])
        ezd  = zd_d - float(meas["zdot_q"])
        eth  = th_d - float(meas["theta"])
        ethd = thd_d - float(meas["theta_dot"])
        eb   = beta_d - float(meas["beta"])
        ebd  = betad_d - float(meas["beta_dot"])

        # --- Eq. (11): thrust PD + FF ---
        u1_c = g.kpz * ez + g.kdz * ezd + u1_ff

        # --- Eq. (13): lateral -> attitude command ---
        lenh_ngang = g.kpx * ex + g.kdx * exd
        lenh_ngang = float(np.clip(lenh_ngang, -0.999, 0.999))  # tránh lỗi arcsin
        theta_c = np.arcsin(lenh_ngang) + th_d

        # --- Eq. (12): attitude moment PD + FF ---
        u3_pd = g.kp_theta * (theta_c - float(meas["theta"])) + g.kd_theta * ethd
        u3_c  = u3_pd + u3_ff

        # --- Cánh tay (β): PD + FF ---
        tau_pd = g.kp_beta * eb + g.kd_beta * ebd  # torque trực tiếp
        tau_c = tau_pd + tau_ff

        return float(u1_c), float(u3_c), float(tau_c)

    def mophong(self, save_csv: Optional[str] = None, animate: bool = False):
        """
        Harness tối giản chạy mô phỏng phẳng với mo_phong (nếu có).
        mo_phong dùng state = [y, y_dot, z, z_dot, phi, phi_dot, beta, beta_dot],
        và động học: J_q*phi_ddot = (u2 - tau). ĐỂ ÁNH XẠ ĐÚNG (Eq. 31), PHẢI DÙNG u2 = u3.
        """
        try:
            from mo_phong import jax_dynamics_matrix
        except Exception as e:
            # Fallback: dùng động lực học tối giản bằng NumPy nếu không import được mo_phong (ví dụ thiếu JAX)
            print(f"[WARN] Không import được mo_phong.py ({e}). Dùng mô phỏng fallback đơn giản.")

            def jax_dynamics_matrix(state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
                """
                Fallback dynamics (explicit Euler) cho hệ phẳng x–z:
                state = [y, y_dot, z, z_dot, phi, phi_dot, beta, beta_dot]
                control = [u1, u2, tau] với u2 = u3 (phần thân), tau: gripper.
                """
                m_q = float(self.params.get("m_q", 0.5))
                m_g = float(self.params.get("m_g", 0.158))
                J_q = float(self.params.get("J_q", 1.2e-2))
                J_g = float(self.params.get("J_g", 1.0e-3))
                g   = float(self.params.get("g", 9.81))
                m_s = m_q + m_g

                y, y_dot, z, z_dot, phi, phi_dot, beta, beta_dot = state.tolist()
                u1, u2, tau = control.tolist()

                # Động lực học tối giản (tương thích Eq. (31) và lực thrust)
                y_ddot   = (u1 / max(m_s, 1e-9)) * np.sin(phi)
                z_ddot   = (u1 / max(m_s, 1e-9)) * np.cos(phi) - g
                phi_ddot = (u2 - tau) / max(J_q, 1e-9)
                beta_ddot= tau / max(J_g, 1e-9)

                # Tích phân Euler
                y_dot   = y_dot   + y_ddot   * dt
                y       = y       + y_dot    * dt
                z_dot   = z_dot   + z_ddot   * dt
                z       = z       + z_dot    * dt
                phi_dot = phi_dot + phi_ddot * dt
                phi     = phi     + phi_dot  * dt
                beta_dot= beta_dot+ beta_ddot* dt
                beta    = beta    + beta_dot * dt

                return np.array([y, y_dot, z, z_dot, phi, phi_dot, beta, beta_dot], dtype=float)

        # Khởi tạo tại tư thế mong muốn ban đầu
        y0   = self.x_qd[0]
        z0   = self.z_qd[0]
        phi0 = self.theta_d[0]
        beta0 = self.beta_d[0]
        state = np.array([y0, 0.0, z0, 0.0, phi0, 0.0, beta0, 0.0], dtype=float)

        states = [state.copy()]
        cmds   = []  # (u1, u2, u3, tau)

        for i in range(len(self.t)-1):
            # vị trí (m)
            sigma_x = 0.15
            sigma_z = 0.15
            # vận tốc (m/s)
            sigma_xd = 0.15
            sigma_zd = 0.15
            # góc (rad) 
            sigma_theta = np.deg2rad(0.5)
            sigma_beta  = np.deg2rad(5.0)
            # tốc độ góc (rad/s)
            sigma_thetad = 0.03
            sigma_betad  = 1.0

            # nhiễu Gaussian 
            nx  = np.random.randn() * sigma_x
            nz  = np.random.randn() * sigma_z
            nxd = np.random.randn() * sigma_xd
            nzd = np.random.randn() * sigma_zd
            nth = np.random.randn() * sigma_theta
            nthd= np.random.randn() * sigma_thetad
            nb  = np.random.randn() * sigma_beta
            nbd = np.random.randn() * sigma_betad

            # cộng nhiễu vào đo lường
            meas = dict(
                x_q=float(state[0] + nx),     xdot_q=float(state[1] + nxd),
                z_q=float(state[2] + nz),     zdot_q=float(state[3] + nzd),
                theta=float(state[4] + nth), theta_dot=float(state[5] + nthd),
                beta=float(state[6] + nb),   beta_dot=float(state[7] + nbd),
            )

            u1, u3, tau = self.step(i, meas)

            # ÁNH XẠ ĐÚNG: mo_phong dùng J_q*phi_ddot = (u2 - tau) ⇒ Đặt u2 = u3
            u2 = u3

            # Tích phân một bước
            control = np.array([u1, u2, tau], dtype=float)
            state = np.array(jax_dynamics_matrix(state, control, dt=self.dt), dtype=float)

            states.append(state.copy())
            cmds.append([u1, u2, u3, tau])

        states = np.array(states)
        cmds   = np.array(cmds)

        if save_csv:
            if not save_csv.lower().endswith('.csv'):
                folder_path = save_csv
                
                # Tạo thư mục nếu chưa có
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                
                save_csv = os.path.join(folder_path, "C:\\Users\\2003h\\OneDrive\\Máy tính\\doan_2025.1\\Quadrotor-Control-System\\src\\minsnap_results\\ketqua.csv")
            else:
                folder_path = os.path.dirname(save_csv)
                if folder_path and not os.path.exists(folder_path):
                    os.makedirs(folder_path)
            log = pd.DataFrame({
                "t": self.t[:len(cmds)],
                "u1": cmds[:,0], "u2": cmds[:,1], "u3": cmds[:,2], "tau": cmds[:,3],
                "x_q": states[:-1,0], "xdot_q": states[:-1,1],
                "z_q": states[:-1,2], "zdot_q": states[:-1,3],
                "theta": states[:-1,4], "theta_dot": states[:-1,5],
                "beta": states[:-1,6],  "beta_dot": states[:-1,7],
            })
            log.to_csv(save_csv, index=False)
            print(f"[OK] Saved sim log -> {save_csv}")

        if animate:
            try:
                from mo_phong import animate as mo_phong_animate
                mo_phong_animate(states, cmds[:, :3], target=(self.x_qd[-1], self.z_qd[-1]), dt=self.dt, save_frames=True, output_dir="../media/animation_frames")
            except Exception as e:
                print(f"Animation failed: {e}")

        return states, cmds
# main 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PD + FF quad controller")
    parser.add_argument("--flat_csv", type=str, default="C:\\Users\\2003h\\OneDrive\\Máy tính\\doan_2025.1\\Quadrotor-Control-System\\src\\minsnap_results\\flat_outputs.csv",
                        help="CSV planner: cột t,x_q,z_q,beta (beta: rad).")
    parser.add_argument("--simulate", action="store_true", help="Chạy mô phỏng với mo_phong.py")
    parser.add_argument("--save_csv", type=str, default=None, help="luu file csv ket qua mo phong.")
    parser.add_argument("--animate", action="store_true", help="mo phong")
    args = parser.parse_args()

    ctrl = PDFFController(flat_csv=args.flat_csv)

    if args.simulate:
        ctrl.mophong(save_csv=args.save_csv, animate=args.animate)
    else:
        print("Controller ready. Gọi PDFFController.step(i, meas) mỗi chu kỳ điều khiển.")
