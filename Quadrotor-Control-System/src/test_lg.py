# -*- coding: utf-8 -*-
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import re

# Cấu hình
LG_VALUES = [0.1, 0.2, 0.35]
FILES_TO_MODIFY = ["chuyen_doi.py", "mo_phong.py"]
BACKUP_DIR = "backup_lg_test"

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
    fig = plt.figure(figsize=(16, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(lg_values)))
    
    # Quỹ đạo X-Z
    ax1 = plt.subplot(2, 2, 1)
    for i, (lg, traj) in enumerate(zip(lg_values, trajectories)):
        if traj is not None:
            ax1.plot(traj['x_q'], traj['z_q'], color=colors[i], linewidth=2, 
                    label=f'L_g = {lg:.2f}m')
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Z (m)')
    ax1.set_title('Quỹ đạo XZ', fontweight='bold')
    ax1.grid(True, alpha=0.3); ax1.legend(); ax1.axis('equal')
    
    # X và Z theo thời gian
    ax2 = plt.subplot(2, 2, 2)
    for i, (lg, traj) in enumerate(zip(lg_values, trajectories)):
        if traj is not None:
            ax2.plot(traj['t'], traj['x_q'], color=colors[i], linewidth=2, linestyle='-',
                    label=f'X - {lg:.2f}m')
            ax2.plot(traj['t'], traj['z_q'], color=colors[i], linewidth=2, linestyle='--',
                    label=f'Z - {lg:.2f}m')
    ax2.set_xlabel('Thời gian (s)'); ax2.set_ylabel('Vị trí (m)')
    ax2.set_title('X và Z theo thời gian', fontweight='bold')
    ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8, ncol=2)
    
    # Theta và Beta
    ax3 = plt.subplot(2, 2, 3)
    for i, (lg, traj) in enumerate(zip(lg_values, trajectories)):
        if traj is not None:
            ax3.plot(traj['t'], np.rad2deg(traj['theta']), color=colors[i], linewidth=2, 
                    linestyle='-', label=f'θ - {lg:.2f}m')
            ax3.plot(traj['t'], np.rad2deg(traj['beta']), color=colors[i], linewidth=2, 
                    linestyle='--', label=f'β - {lg:.2f}m')
    ax3.set_xlabel('Thời gian (s)'); ax3.set_ylabel('Góc (độ)')
    ax3.set_title('Theta và Beta', fontweight='bold')
    ax3.grid(True, alpha=0.3); ax3.legend(fontsize=8, ncol=2)
    
    # Sai số
    ax4 = plt.subplot(2, 2, 4)
    if trajectories[-1] is not None:
        ref_traj = trajectories[-1]
        for i, (lg, traj) in enumerate(zip(lg_values, trajectories)):
            if traj is not None and i != len(trajectories)-1:
                error = np.sqrt((traj['x_q'] - ref_traj['x_q'])**2 + 
                              (traj['z_q'] - ref_traj['z_q'])**2)
                ax4.plot(traj['t'], error, color=colors[i], linewidth=2, label=f'{lg:.2f}m')
        ax4.set_xlabel('Thời gian (s)'); ax4.set_ylabel('Sai số (m)')
        ax4.set_title(f'Sai số so với L_g={lg_values[-1]:.2f}m', fontweight='bold')
        ax4.grid(True, alpha=0.3); ax4.legend()
    
    plt.tight_layout()
    output_file = 'minsnap_results/lg_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Đã lưu: {output_file}")
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
