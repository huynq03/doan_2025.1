# -*- coding: utf-8 -*-
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import re

# Cấu hình
LG_VALUES = [0.25, 0.3, 0.35, 0.4, 0.45]
FILES_TO_MODIFY = ["chuyen_doi.py", "mo_phong.py"]
BACKUP_DIR = "backup_lg_test"
# Nếu True: dùng file minsnap_results/flat_outputs.csv hiện có, KHÔNG gọi qp5.py
USE_EXISTING_FLAT = True
# Giá trị tham chiếu để so sánh (sẽ dùng nếu có trong LG_VALUES)
REFERENCE_LG = 0.35

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
    print("\n" + "="*70)
    print(f"{'L_g (m)':<10} {'X_max':<10} {'X_min':<10} {'Z_max':<10} {'Z_min':<10} {'θ_max(°)':<12}")
    print("-"*70)
    for lg, traj in zip(lg_values, trajectories):
        if traj is not None:
            print(f"{lg:<10.2f} {traj['x_q'].max():<10.3f} {traj['x_q'].min():<10.3f} "
                  f"{traj['z_q'].max():<10.3f} {traj['z_q'].min():<10.3f} "
                  f"{np.rad2deg(np.abs(traj['theta']).max()):<12.3f}")

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
