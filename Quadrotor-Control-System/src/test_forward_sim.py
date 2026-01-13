import argparse
import os
import numpy as np

# Reuse dynamics and optional animation from mo_phong
try:
    from mo_phong import jax_dynamics_matrix, animate, m_q, m_g, g, dt as DEFAULT_DT
except Exception as e:
    raise RuntimeError(f"Failed to import mo_phong dependencies: {e}")


def simulate_constant_controls(u1: float, u2: float, tau: float, steps: int, dt_sim: float, init_state=None):
    """Integrate forward dynamics with constant inputs.

    State: [y, y_dot, z, z_dot, phi, phi_dot, beta, beta_dot]
    Control: [u1, u2, tau]
    """
    if init_state is None:
        init_state = np.zeros(8, dtype=np.float64)
    states = np.zeros((steps, 8), dtype=np.float64)
    controls = np.tile(np.array([u1, u2, tau], dtype=np.float64), (steps, 1))

    states[0] = np.array(init_state, dtype=np.float64)
    for i in range(1, steps):
        states[i] = np.array(jax_dynamics_matrix(states[i-1], controls[i-1], dt=dt_sim), dtype=np.float64)
    return states, controls

def build_control_profile(profile: str,
                          steps: int,
                          dt_sim: float,
                          hover_u1: float,
                          u1_delta: float = 0.0,
                          torque: float = 0.0,
                          sine_amp: float = 0.0,
                          sine_freq: float = 0.5,
                          tilt_steps: int = 50):
    """Create a time-varying control array [u1, u2, tau] for testing.

    Profiles:
      - hover: u1 = hover, u2 = 0, tau = 0
      - ascend: u1 = hover + u1_delta
      - descend: u1 = hover - u1_delta
      - sine: u1 = hover + sine_amp * sin(2π f t)
      - forward: apply pitch moment for `tilt_steps`, then hold tilt with u2=0; u1≈hover
    """
    controls = np.zeros((steps, 3), dtype=np.float64)
    t = np.arange(steps) * dt_sim
    if profile == "hover":
        controls[:, 0] = hover_u1
        controls[:, 1] = 0.0
        controls[:, 2] = 0.0
    elif profile == "ascend":
        controls[:, 0] = hover_u1 + u1_delta
        controls[:, 1] = 0.0
        controls[:, 2] = 0.0
    elif profile == "descend":
        controls[:, 0] = hover_u1 - u1_delta
        controls[:, 1] = 0.0
        controls[:, 2] = 0.0
    elif profile == "sine":
        controls[:, 0] = hover_u1 + sine_amp * np.sin(2.0 * np.pi * sine_freq * t)
        controls[:, 1] = 0.0
        controls[:, 2] = 0.0
    elif profile == "forward":
        # Apply net pitch moment (u2 - tau) for tilt_steps to create forward acceleration (via phi)
        controls[:, 0] = hover_u1 + 0.0  # keep near hover
        controls[:tilt_steps, 1] = torque
        controls[:tilt_steps, 2] = 0.0
        controls[tilt_steps:, 1] = 0.0
        controls[tilt_steps:, 2] = 0.0
    else:
        raise ValueError(f"Unknown profile: {profile}")
    return controls


def save_states_controls_csv(path: str, states: np.ndarray, controls: np.ndarray):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = "y,y_dot,z,z_dot,phi,phi_dot,beta,beta_dot,u1,u2,tau"
    stacked = np.concatenate([states, controls], axis=1)
    np.savetxt(path, stacked, delimiter=",", header=header, comments="")


def load_controls_from_csv(path: str, steps: int):
    """Load control inputs from CSV file.
    
    Expected format:
    - Single row (u1, u2, tau): constant controls replicated for all steps
    - Multiple rows: one control per time step (must match steps or be truncated/padded)
    """
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.shape[0] == 1:
        # Single row - replicate for all steps
        controls = np.tile(data[0, :3], (steps, 1))
        print(f"[INFO] Loaded constant controls from CSV: u1={data[0,0]:.3f}, u2={data[0,1]:.3f}, tau={data[0,2]:.3f}")
    else:
        # Time-series controls
        if data.shape[0] < steps:
            # Pad with last value
            pad = np.tile(data[-1:, :3], (steps - data.shape[0], 1))
            controls = np.vstack([data[:, :3], pad])
            print(f"[INFO] Loaded {data.shape[0]} controls from CSV, padded to {steps} steps")
        else:
            controls = data[:steps, :3]
            print(f"[INFO] Loaded {steps} controls from CSV")
    return controls


def main():
    parser = argparse.ArgumentParser(description="Forward simulation test with constant inputs")
    parser.add_argument("--u1", type=float, default=(m_q + m_g) * g, help="Total thrust (hover ≈ (m_q+m_g)*g)")
    parser.add_argument("--u2", type=float, default=0.0, help="Moment term u2")
    parser.add_argument("--tau", type=float, default=0.0, help="Moment term tau")
    parser.add_argument("--steps", type=int, default=300, help="Number of simulation steps")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT, help="Simulation dt")
    parser.add_argument("--save-csv", default="", help="Path to save states+controls CSV")
    parser.add_argument("--animate", action="store_true", help="Show animation using mo_phong.animate")

    # Profiles
    parser.add_argument("--profile", choices=["constant", "hover", "ascend", "descend", "forward", "sine"], default="constant",
                        help="Choose a control profile")
    parser.add_argument("--u1-delta", type=float, default=1.0, help="Delta thrust for ascend/descend profiles")
    parser.add_argument("--torque", type=float, default=0.5, help="Pitch moment for forward profile (u2)")
    parser.add_argument("--tilt-steps", type=int, default=50, help="Steps to apply torque in forward profile")
    parser.add_argument("--sine-amp", type=float, default=0.5, help="Amplitude for sine thrust profile")
    parser.add_argument("--sine-freq", type=float, default=0.5, help="Frequency (Hz) for sine thrust profile")

    # Load controls from CSV
    parser.add_argument("--controls-csv", default="", help="Path to CSV with controls (u1,u2,tau). Single row = constant, multiple rows = time series")

    args = parser.parse_args()

    if args.controls_csv:
        # Load controls from CSV file
        controls = load_controls_from_csv(args.controls_csv, args.steps)
        # Integrate with loaded controls
        states = np.zeros((args.steps, 8), dtype=np.float64)
        states[0] = np.zeros(8, dtype=np.float64)
        for i in range(1, args.steps):
            states[i] = np.array(jax_dynamics_matrix(states[i-1], controls[i-1], dt=args.dt), dtype=np.float64)
    elif args.profile == "constant":
        states, controls = simulate_constant_controls(args.u1, args.u2, args.tau, steps=args.steps, dt_sim=args.dt)
    else:
        hover_u1 = (m_q + m_g) * g
        controls = build_control_profile(
            profile=args.profile,
            steps=args.steps,
            dt_sim=args.dt,
            hover_u1=hover_u1,
            u1_delta=args.u1_delta,
            torque=args.torque,
            sine_amp=args.sine_amp,
            sine_freq=args.sine_freq,
            tilt_steps=args.tilt_steps,
        )
        # Integrate with time-varying controls
        states = np.zeros((args.steps, 8), dtype=np.float64)
        states[0] = np.zeros(8, dtype=np.float64)
        for i in range(1, args.steps):
            states[i] = np.array(jax_dynamics_matrix(states[i-1], controls[i-1], dt=args.dt), dtype=np.float64)
    T_total = args.steps * args.dt
    print(f"[INFO] Simulated {args.steps} steps, dt={args.dt:.3f}s, T={T_total:.2f}s")
    if args.controls_csv:
        print(f"[INFO] Using controls from CSV: {args.controls_csv}")
    elif args.profile == "constant":
        print(f"[INFO] Inputs (constant): u1={args.u1:.3f}, u2={args.u2:.3f}, tau={args.tau:.3f}")
    else:
        print(f"[INFO] Profile: {args.profile}")
    print(
        f"[INFO] Final state: y={states[-1,0]:.3f}, z={states[-1,2]:.3f}, phi={states[-1,4]:.3f}, beta={states[-1,6]:.3f}"
    )

    if args.save_csv:
        save_states_controls_csv(args.save_csv, states, controls)
        print(f"[INFO] Saved CSV: {args.save_csv}")

    if args.animate:
        try:
            animate(states, controls, dt=args.dt, save_frames=False, playback_speed=2.0)
        except Exception as e:
            print(f"[WARN] animate failed: {e}")


if __name__ == "__main__":
    main()
