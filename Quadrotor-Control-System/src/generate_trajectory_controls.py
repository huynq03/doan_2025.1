"""
Generate control sequence to fly from (0,0) to target (y_target, z_target)
Simple open-loop strategy:
1. Tilt forward by applying pitch moment
2. Apply thrust to move diagonally
3. Reduce tilt and maintain altitude near target
"""
import numpy as np
import argparse

def generate_to_target(y_target, z_target, steps, dt, hover_thrust, output_csv):
    """Generate time-series controls to reach target."""
    controls = np.zeros((steps, 3))
    
    # Phase durations (in steps)
    phase1 = int(0.15 * steps)  # Tilt forward
    phase2 = int(0.60 * steps)  # Fly diagonal with tilt
    phase3 = steps - phase1 - phase2  # Level out and maintain
    
    # Phase 1: Tilt forward with moment, hover thrust
    controls[:phase1, 0] = hover_thrust
    controls[:phase1, 1] = 0.8  # Pitch moment u2
    controls[:phase1, 2] = 0.0
    
    # Phase 2: Hold tilt, increase thrust to ascend and move forward
    controls[phase1:phase1+phase2, 0] = hover_thrust + 1.5
    controls[phase1:phase1+phase2, 1] = 0.2  # Maintain some forward tilt
    controls[phase1:phase1+phase2, 2] = 0.0
    
    # Phase 3: Reduce tilt, settle near target
    controls[phase1+phase2:, 0] = hover_thrust + 0.5
    controls[phase1+phase2:, 1] = -0.3  # Counter-tilt to slow down
    controls[phase1+phase2:, 2] = 0.0
    
    # Save to CSV
    np.savetxt(output_csv, controls, delimiter=",", fmt="%.6f")
    print(f"[INFO] Generated {steps} control steps to reach ({y_target}, {z_target})")
    print(f"[INFO] Saved to: {output_csv}")
    print(f"[INFO] Phase 1 (tilt): {phase1} steps")
    print(f"[INFO] Phase 2 (fly): {phase2} steps")
    print(f"[INFO] Phase 3 (settle): {phase3} steps")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--y-target", type=float, default=5.0)
    parser.add_argument("--z-target", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--hover-thrust", type=float, default=6.454)
    parser.add_argument("--output", default="controls_to_target.csv")
    args = parser.parse_args()
    
    generate_to_target(args.y_target, args.z_target, args.steps, 
                      args.dt, args.hover_thrust, args.output)
