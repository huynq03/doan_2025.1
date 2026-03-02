# -*- coding: utf-8 -*-
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import re

# Angle helpers (handle +/-pi discontinuities without wrap-to-pi)
def unwrap_angle(angle_array):
    if angle_array is None:
        return None
    angle_array = np.asarray(angle_array)
    return np.unwrap(angle_array)


def unwrap_and_align(sim_angle, ref_angle):
    """Unwrap both arrays and align sim initial value to ref initial value.

    This removes +/-pi jumps and prevents constant 2*pi offsets without using wrap-to-pi.
    """
    sim_u = unwrap_angle(sim_angle)
    ref_u = unwrap_angle(ref_angle)
    if sim_u is None or ref_u is None or len(sim_u) == 0 or len(ref_u) == 0:
        return sim_u, ref_u
    sim_u = sim_u - sim_u[0] + ref_u[0]
    return sim_u, ref_u

# Helper: compute common error metrics for an error array (e = sim - ref)
def compute_metrics(e, sim=None, ref=None, t=None, tol=0.0):
    e = np.asarray(e)
    n = len(e)
    mae = np.mean(np.abs(e)) if n > 0 else np.nan
    rmse = np.sqrt(np.mean(e**2)) if n > 0 else np.nan
    max_abs = np.max(np.abs(e)) if n > 0 else np.nan
    std = np.std(e) if n > 0 else np.nan
    pct_within = (np.sum(np.abs(e) < tol) / n * 100) if n > 0 else np.nan

    dt = np.mean(np.diff(t)) if (t is not None and len(t) > 1) else 0.0
    IAE = np.sum(np.abs(e)) * dt
    ISE = np.sum(e**2) * dt
    ITAE = np.sum((t if t is not None else np.arange(n)) * np.abs(e)) * dt

    # sMAPE needs sim and ref arrays
    if sim is not None and ref is not None:
        denom = (np.abs(sim) + np.abs(ref) + 1e-9)
        sMAPE = np.mean(2 * np.abs(sim - ref) / denom) * 100
    else:
        sMAPE = np.nan

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'max_abs': float(max_abs),
        'std': float(std),
        'percent_within_tol': float(pct_within),
        'IAE': float(IAE),
        'ISE': float(ISE),
        'ITAE': float(ITAE),
        'sMAPE_percent': float(sMAPE)
    }

# Cấu hình
LG_VALUES = [0.25, 0.3, 0.35, 0.4, 0.45]
FILES_TO_MODIFY = ["chuyen_doi.py", "mo_phong.py"]
BACKUP_DIR = "backup_lg_test"
# Nếu True: dùng file minsnap_results/flat_outputs.csv hiện có, KHÔNG gọi qp5.py
USE_EXISTING_FLAT = True
# Giá trị tham chiếu để so sánh (sẽ dùng nếu có trong LG_VALUES)
REFERENCE_LG = 0.35

# Console output control
# If False: only print best/worst summary for trajectory + angles (RMSE & Max)
PRINT_DETAILED_METRICS = False

def backup_files():
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    for file in FILES_TO_MODIFY:
        if os.path.exists(file):
            shutil.copy2(file, os.path.join(BACKUP_DIR, file))

def restore_files():
    for file in FILES_TO_MODIFY:
        backup_path = os.path.join(BACKUP_DIR, file)
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, file)

def modify_lg_value(file_path, lg_value):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if file_path == "chuyen_doi.py":
        pattern = r'(L_g\s*=\s*)[\d.]+(\s*,\s*#)'
        content = re.sub(pattern, f'\\g<1>{lg_value}\\g<2>', content)
    
    elif file_path == "mo_phong.py":
        pattern_lp = r'(l_p,\s*l_q\s*=\s*)[\d.]+(\s*,\s*[\d.]+)'
        content = re.sub(pattern_lp, f'\\g<1>{lg_value}\\g<2>', content)
        pattern_Lg = r'(J_q,\s*J_g,\s*L_g\s*=\s*[\d.]+,\s*[\d.]+,\s*)[\d.]+'
        content = re.sub(pattern_Lg, f'\\g<1>{lg_value}', content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def regenerate_flat_outputs(lg_value):
    if USE_EXISTING_FLAT:
        print(f"  → Dùng file flat_outputs.csv hiện có (L_g={lg_value}m)...", end=" ")
        flat_path = "minsnap_results/flat_outputs.csv"
        if not os.path.exists(flat_path):
            print(f"✗ File không tồn tại")
            return False
        print("✓ (không tái tạo)")
        return True

    # Nếu không dùng file hiện có thì chạy qp5.py để tái tạo
    print(f"  → Tái tạo flat_outputs.csv (L_g={lg_value}m)...", end=" ")
    flat_path = "minsnap_results/flat_outputs.csv"
    old_mtime = os.path.getmtime(flat_path) if os.path.exists(flat_path) else None

    result = subprocess.run(["python", "qp5.py"], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"✗ Lỗi qp5.py")
        return False

    if not os.path.exists(flat_path):
        print(f"✗ File không tồn tại")
        return False

    new_mtime = os.path.getmtime(flat_path)
    if old_mtime and new_mtime == old_mtime:
        print(f"⚠️ File không cập nhật")
        return False

    print("✓")
    return True

def run_simulation(lg_value):
    output_csv = f"minsnap_results/ketqua_lg{lg_value:.2f}.csv"
    
    print(f"\n[L_g = {lg_value}m]")
    
    if not regenerate_flat_outputs(lg_value):
        return None
    
    print(f"  → Chạy simulation...", end=" ")
    result = subprocess.run(["python", "dieu_khien.py", "--simulate", "--save_csv", output_csv],
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓")
        return output_csv
    else:
        print("✗")
        return None

def load_trajectory(csv_file):
    if not os.path.exists(csv_file):
        return None
    df = pd.read_csv(csv_file)
    return {
        't': df['t'].values,
        'x_q': df['x_q'].values,
        'z_q': df['z_q'].values,
        'theta': df['theta'].values,
        'beta': df['beta'].values,
    }

def plot_comparison(trajectories, lg_values):
    colors = plt.cm.viridis(np.linspace(0, 1, len(lg_values)))

    # Figure 1: Quỹ đạo X-Z (riêng)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for i, (lg, traj) in enumerate(zip(lg_values, trajectories)):
        if traj is not None:
            ax1.plot(traj['x_q'], traj['z_q'], color=colors[i], linewidth=2, label=f'L_g = {lg:.2f} m')
    ax1.set_xlabel('x(m)', fontsize=12)
    ax1.set_ylabel('z(m)', fontsize=12)
    ax1.set_title('Các quỹ đạo chuyển động', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    fig1.tight_layout()

    # Figure 2: X và Z theo thời gian
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for i, (lg, traj) in enumerate(zip(lg_values, trajectories)):
        if traj is not None:
            ax2.plot(traj['t'], traj['x_q'], color=colors[i], linewidth=2, linestyle='-', label=f'X - {lg:.2f} m')
            ax2.plot(traj['t'], traj['z_q'], color=colors[i], linewidth=2, linestyle='--', label=f'Z - {lg:.2f} m')
    ax2.set_xlabel('Thời gian (s)', fontsize=12)
    ax2.set_ylabel('Vị trí (m)', fontsize=12)
    ax2.set_title('Vị trí x và z theo thời gian', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, ncol=2)
    fig2.tight_layout()

    # Figure 3: Theta và Beta
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for i, (lg, traj) in enumerate(zip(lg_values, trajectories)):
        if traj is not None:
            ax3.plot(traj['t'], np.rad2deg(traj['theta']), color=colors[i], linewidth=2, linestyle='-', label=f'θ - {lg:.2f} m')
            ax3.plot(traj['t'], np.rad2deg(traj['beta']), color=colors[i], linewidth=2, linestyle='--', label=f'β - {lg:.2f} m')
    ax3.set_xlabel('Thời gian (s)', fontsize=12)
    ax3.set_ylabel('Góc (°)', fontsize=12)
    ax3.set_title(r'Góc $\theta$ và $\beta$ theo thời gian', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8, ncol=2)
    fig3.tight_layout()

    # Figure 4: Sai số so với trường hợp tham chiếu (REFERENCE_LG nếu có)
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    # chọn chỉ mục tham chiếu: nếu có REFERENCE_LG trong lg_values dùng nó, không thì dùng phần tử cuối
    try:
        idx_ref = list(lg_values).index(REFERENCE_LG)
        ref_lg = REFERENCE_LG
    except ValueError:
        idx_ref = len(lg_values) - 1
        ref_lg = lg_values[idx_ref]

    if trajectories and trajectories[idx_ref] is not None:
        ref_traj = trajectories[idx_ref]
        for i, (lg, traj) in enumerate(zip(lg_values, trajectories)):
            if traj is not None and i != idx_ref:
                # đảm bảo kích thước mảng giống nhau trước khi tính lỗi
                error = np.sqrt((traj['x_q'] - ref_traj['x_q'])**2 + (traj['z_q'] - ref_traj['z_q'])**2)
                ax4.plot(traj['t'], error, color=colors[i], linewidth=2, label=f'{lg:.2f} m')
        ax4.set_xlabel('Thời gian (s)', fontsize=12)
        ax4.set_ylabel('Sai số (m)', fontsize=12)
        ax4.set_title(f'Sai số so với L_g={ref_lg:.2f} m', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    fig4.tight_layout()

    # Lưu từng figure riêng
    out1 = 'minsnap_results/lg_xz.png'
    out2 = 'minsnap_results/lg_xz_time.png'
    out3 = 'minsnap_results/lg_angles.png'
    out4 = 'minsnap_results/lg_error.png'
    try:
        fig1.savefig(out1, dpi=150, bbox_inches='tight')
        fig2.savefig(out2, dpi=150, bbox_inches='tight')
        fig3.savefig(out3, dpi=150, bbox_inches='tight')
        fig4.savefig(out4, dpi=150, bbox_inches='tight')
        print(f"\n✓ Đã lưu: {out1}, {out2}, {out3}, {out4}")
    except Exception as e:
        print(f"✗ Lỗi khi lưu ảnh: {e}")

    plt.show()

def print_statistics(trajectories, lg_values):
    import pandas as pd
    print("\n" + "="*70)
    print("Basic range statistics:")
    print(f"{'L_g (m)':<10} {'X_max':<10} {'X_min':<10} {'Z_max':<10} {'Z_min':<10} {'θ_max(°)':<12}")
    print("-"*70)
    for lg, traj in zip(lg_values, trajectories):
        if traj is not None:
            print(f"{lg:<10.2f} {traj['x_q'].max():<10.3f} {traj['x_q'].min():<10.3f} "
                  f"{traj['z_q'].max():<10.3f} {traj['z_q'].min():<10.3f} "
                  f"{np.rad2deg(np.abs(traj['theta']).max()):<12.3f}")

    # If there's a reference trajectory, compute per-variable error metrics against it
    try:
        idx_ref = list(lg_values).index(REFERENCE_LG)
        ref_traj = trajectories[idx_ref]
    except ValueError:
        idx_ref = None
        ref_traj = None

    metrics_rows = []
    if ref_traj is not None:
        pos_tol = 0.05  # meters
        ang_tol = np.deg2rad(3.0)  # radians (~3 degrees)

        # Per-L_g summary for slide-friendly reporting
        summary = []

        for lg, traj in zip(lg_values, trajectories):
            if traj is None:
                continue
            if lg == REFERENCE_LG:
                continue

            # align lengths if necessary
            n = min(len(traj['t']), len(ref_traj['t']))
            t = traj['t'][:n]

            # compute per-variable errors (sim - ref)
            ex = traj['x_q'][:n] - ref_traj['x_q'][:n]
            ez = traj['z_q'][:n] - ref_traj['z_q'][:n]
            theta_sim_u, theta_ref_u = unwrap_and_align(traj['theta'][:n], ref_traj['theta'][:n])
            beta_sim_u, beta_ref_u = unwrap_and_align(traj['beta'][:n], ref_traj['beta'][:n])
            etheta = theta_sim_u - theta_ref_u
            ebeta = beta_sim_u - beta_ref_u

            # Trajectory (position) error magnitude
            epos = np.sqrt(ex**2 + ez**2)
            pos_rmse = float(np.sqrt(np.mean(epos**2)))
            pos_max = float(np.max(epos))

            # Individual angle metrics (for reporting)
            theta_rmse = float(np.sqrt(np.mean(etheta**2)))
            theta_max = float(np.max(np.abs(etheta)))
            beta_rmse = float(np.sqrt(np.mean(ebeta**2)))
            beta_max = float(np.max(np.abs(ebeta)))

            summary.append({
                'L_g': float(lg),
                'pos_rmse': pos_rmse,
                'pos_max': pos_max,
                'theta_rmse': theta_rmse,
                'theta_max': theta_max,
                'beta_rmse': beta_rmse,
                'beta_max': beta_max
            })

            mx = compute_metrics(ex, sim=traj['x_q'][:n], ref=ref_traj['x_q'][:n], t=t, tol=pos_tol)
            mz = compute_metrics(ez, sim=traj['z_q'][:n], ref=ref_traj['z_q'][:n], t=t, tol=pos_tol)
            mtheta = compute_metrics(etheta, sim=theta_sim_u, ref=theta_ref_u, t=t, tol=ang_tol)
            mbeta = compute_metrics(ebeta, sim=beta_sim_u, ref=beta_ref_u, t=t, tol=ang_tol)

            if PRINT_DETAILED_METRICS:
                print("\n" + "-"*60)
                print(f"L_g = {lg:.2f} m (so với L_g ref = {REFERENCE_LG:.2f} m):")
                print(f"  X:   MAE={mx['mae']:.4e} m, RMSE={mx['rmse']:.4e} m, MaxAbs={mx['max_abs']:.4e} m, %within{pos_tol}m={mx['percent_within_tol']:.2f}%")
                print(f"  Z:   MAE={mz['mae']:.4e} m, RMSE={mz['rmse']:.4e} m, MaxAbs={mz['max_abs']:.4e} m, %within{pos_tol}m={mz['percent_within_tol']:.2f}%")
                print(f"  θ:   MAE={mtheta['mae']:.4e} rad ({np.rad2deg(mtheta['mae']):.3f}°), RMSE={mtheta['rmse']:.4e} rad")
                print(f"  β:   MAE={mbeta['mae']:.4e} rad ({np.rad2deg(mbeta['mae']):.3f}°), RMSE={mbeta['rmse']:.4e} rad")

            # collect rows for CSV export
            for var_name, mm in [('x_q', mx), ('z_q', mz), ('theta', mtheta), ('beta', mbeta)]:
                row = {
                    'L_g': float(lg),
                    'var': var_name,
                    'mae': mm['mae'],
                    'rmse': mm['rmse'],
                    'max_abs': mm['max_abs'],
                    'std': mm['std'],
                    'percent_within_tol': mm['percent_within_tol'],
                    'IAE': mm['IAE'],
                    'ISE': mm['ISE'],
                    'ITAE': mm['ITAE'],
                    'sMAPE_percent': mm['sMAPE_percent']
                }
                metrics_rows.append(row)

        # Slide-friendly summary: only best (min) and worst (max)
        if summary:
            best_pos = min(summary, key=lambda r: r['pos_rmse'])
            worst_pos = max(summary, key=lambda r: r['pos_rmse'])
            best_beta = min(summary, key=lambda r: r['beta_rmse'])
            worst_beta = max(summary, key=lambda r: r['beta_rmse'])

            print("\n" + "="*70)
            print("TÓM TẮT (chỉ 2 số/nhóm: RMSE và Max):")
            print(f"Ref: L_g = {REFERENCE_LG:.2f} m")

            print("\n[QUỸ ĐẠO |e_pos| = sqrt(e_x^2 + e_z^2)]")
            print(f"  Ít sai số nhất:  L_g={best_pos['L_g']:.2f}  RMSE={best_pos['pos_rmse']:.6f} m   Max={best_pos['pos_max']:.6f} m")
            print(f"  Nhiều sai số nhất: L_g={worst_pos['L_g']:.2f}  RMSE={worst_pos['pos_rmse']:.6f} m   Max={worst_pos['pos_max']:.6f} m")

            print("\n[GÓC β (beta)]")
            print(f"  Ít sai số nhất:  L_g={best_beta['L_g']:.2f}  RMSE={best_beta['beta_rmse']:.6f} rad  Max={best_beta['beta_max']:.6f} rad")
            print(f"  Nhiều sai số nhất: L_g={worst_beta['L_g']:.2f}  RMSE={worst_beta['beta_rmse']:.6f} rad  Max={worst_beta['beta_max']:.6f} rad")

        # export to CSV
        if metrics_rows:
            metrics_df = pd.DataFrame(metrics_rows)
            out_metrics = 'minsnap_results/test_lg_metrics.csv'
            try:
                metrics_df.to_csv(out_metrics, index=False)
                print(f"\n✓ Lưu metrics tổng hợp: {out_metrics}")
            except Exception as e:
                print(f"✗ Không thể lưu metrics: {e}")
    else:
        print("\nKhông tìm thấy trajectory tham chiếu để tính sai số (REFERENCE_LG).")

def main():
    print("="*70)
    print("SO SÁNH ẢNH HƯỞNG CỦA L_g")
    print("="*70)
    
    backup_files()
    
    trajectories = []
    result_files = []
    
    for lg in LG_VALUES:
        for file in FILES_TO_MODIFY:
            modify_lg_value(file, lg)
        
        result_file = run_simulation(lg)
        result_files.append(result_file)
        trajectories.append(load_trajectory(result_file) if result_file else None)
    
    restore_files()
    
    print("\n[Vẽ biểu đồ]")
    plot_comparison(trajectories, LG_VALUES)
    print_statistics(trajectories, LG_VALUES)
    
    print("\n" + "="*70)
    print("✓ HOÀN THÀNH")
    print("="*70)

if __name__ == "__main__":
    main()
