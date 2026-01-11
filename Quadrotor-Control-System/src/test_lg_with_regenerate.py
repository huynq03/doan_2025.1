# -*- coding: utf-8 -*-
"""
Script để test các giá trị L_g khác nhau và so sánh quỹ đạo
PHIÊN BẢN CẢI TIẾN: Tái tạo flat_outputs.csv với L_g mới
"""
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

# ========== CẤU HÌNH ==========
# Danh sách các giá trị L_g để test (mét)
LG_VALUES = [0.1, 0.15, 0.25, 0.35]

# File cần sửa đổi
FILES_TO_MODIFY = [
    "chuyen_doi.py",
    "mo_phong.py"
]

# Thư mục backup
BACKUP_DIR = "backup_lg_test"

# ========== HÀM PHỤ ==========
def backup_files():
    """Sao lưu các file gốc"""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    
    for file in FILES_TO_MODIFY:
        if os.path.exists(file):
            shutil.copy2(file, os.path.join(BACKUP_DIR, file))
            print(f"✓ Đã backup: {file}")

def restore_files():
    """Khôi phục các file gốc"""
    for file in FILES_TO_MODIFY:
        backup_path = os.path.join(BACKUP_DIR, file)
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, file)
            print(f"✓ Đã khôi phục: {file}")

def modify_lg_value(file_path, lg_value):
    """Thay đổi giá trị L_g và l_p trong file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Tìm và thay thế L_g
    if file_path == "chuyen_doi.py":
        # Thay đổi trong PARAMS dict
        import re
        pattern = r'(L_g\s*=\s*)[\d.]+(\s*,\s*#)'
        replacement = f'\\g<1>{lg_value}\\g<2>'
        content = re.sub(pattern, replacement, content)
    
    elif file_path == "mo_phong.py":
        # Thay đổi l_p (chiều dài pendulum = L_g)
        import re
        pattern_lp = r'(l_p,\s*l_q\s*=\s*)[\d.]+(\s*,\s*[\d.]+)'
        replacement_lp = f'\\g<1>{lg_value}\\g<2>'
        content = re.sub(pattern_lp, replacement_lp, content)
        
        # Thay đổi L_g trong phần khai báo
        pattern_Lg = r'(J_q,\s*J_g,\s*L_g\s*=\s*[\d.]+,\s*[\d.]+,\s*)[\d.]+'
        replacement_Lg = f'\\g<1>{lg_value}'
        content = re.sub(pattern_Lg, replacement_Lg, content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  • Đã sửa {file_path}: L_g = l_p = {lg_value}")

def regenerate_flat_outputs(lg_value):
    """
    TÁI TẠO flat_outputs.csv với L_g mới
    Chạy qp5.py để tạo quỹ đạo minsnap với L_g đã được cập nhật
    """
    print(f"\n  🔄 Tái tạo flat_outputs.csv với L_g = {lg_value} m...")
    print(f"     Đang chạy: python qp5.py")
    
    # Chạy qp5.py để tạo flat_outputs.csv mới
    cmd = ["python", "qp5.py"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ✗ Lỗi khi chạy qp5.py:")
        print(result.stderr)
        return False
    
    print(f"  ✓ Đã tái tạo flat_outputs.csv thành công!")
    
    # Kiểm tra file có tồn tại không
    flat_outputs_path = "minsnap_results/flat_outputs.csv"
    if not os.path.exists(flat_outputs_path):
        print(f"  ✗ Không tìm thấy {flat_outputs_path} sau khi chạy qp5.py")
        return False
    
    return True

def run_simulation(lg_value):
    """Chạy simulation với giá trị L_g cho trước
    
    QUY TRÌNH:
    1. Tái tạo flat_outputs.csv với L_g mới (QUAN TRỌNG!)
    2. Đọc quỹ đạo tham chiếu từ flat_outputs.csv
    3. Chạy PD + Feedforward Controller
    4. Thêm nhiễu Gaussian vào đo lường
    5. Tích phân dynamics với JAX
    6. Lưu quỹ đạo THỰC TẾ vào CSV
    """
    output_csv = f"minsnap_results/ketqua_lg{lg_value:.2f}.csv"
    
    print(f"\n{'='*50}")
    print(f"Đang chạy simulation với L_g = {lg_value} m")
    print(f"{'='*50}")
    print(f"  → Bước 1: Tái tạo flat_outputs.csv với L_g={lg_value}")
    
    # Tái tạo flat_outputs với L_g mới
    if not regenerate_flat_outputs(lg_value):
        print(f"  ✗ Không thể tái tạo flat_outputs, bỏ qua L_g={lg_value}")
        return None
    
    print(f"  → Bước 2: Chạy controller với dynamics L_g={lg_value}")
    print(f"     • PD + Feedforward Controller")
    print(f"     • Nhiễu Gaussian (sigma=0.15)")
    print(f"     • Dynamics tích phân (JAX)")
    
    # Chạy dieu_khien.py
    cmd = ["python", "dieu_khien.py", "--simulate", "--save_csv", output_csv]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ Simulation thành công, kết quả lưu tại: {output_csv}")
        return output_csv
    else:
        print(f"✗ Lỗi khi chạy simulation:")
        print(result.stderr)
        return None

def load_trajectory(csv_file):
    """Đọc dữ liệu quỹ đạo từ file CSV"""
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
    """Vẽ biểu đồ so sánh các quỹ đạo"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Tạo colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(lg_values)))
    
    # 1. Quỹ đạo X-Z
    ax1 = plt.subplot(2, 2, 1)
    for i, (lg, traj) in enumerate(zip(lg_values, trajectories)):
        if traj is not None:
            ax1.plot(traj['x_q'], traj['z_q'], 
                    color=colors[i], linewidth=2, 
                    label=f'L_g = {lg:.2f}m')
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Z (m)', fontsize=12)
    ax1.set_title('Quỹ đạo XZ', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.axis('equal')
    
    # 2. X và Z theo thời gian (gộp)
    ax2 = plt.subplot(2, 2, 2)
    for i, (lg, traj) in enumerate(zip(lg_values, trajectories)):
        if traj is not None:
            ax2.plot(traj['t'], traj['x_q'], 
                    color=colors[i], linewidth=2, linestyle='-',
                    label=f'X - L_g={lg:.2f}m')
            ax2.plot(traj['t'], traj['z_q'], 
                    color=colors[i], linewidth=2, linestyle='--',
                    label=f'Z - L_g={lg:.2f}m')
    ax2.set_xlabel('Thời gian (s)', fontsize=12)
    ax2.set_ylabel('Vị trí (m)', fontsize=12)
    ax2.set_title('Vị trí X và Z theo thời gian', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, ncol=2)
    
    # 3. Theta và Beta theo thời gian (gộp)
    ax3 = plt.subplot(2, 2, 3)
    for i, (lg, traj) in enumerate(zip(lg_values, trajectories)):
        if traj is not None:
            ax3.plot(traj['t'], np.rad2deg(traj['theta']), 
                    color=colors[i], linewidth=2, linestyle='-',
                    label=f'θ - L_g={lg:.2f}m')
            ax3.plot(traj['t'], np.rad2deg(traj['beta']), 
                    color=colors[i], linewidth=2, linestyle='--',
                    label=f'β - L_g={lg:.2f}m')
    ax3.set_xlabel('Thời gian (s)', fontsize=12)
    ax3.set_ylabel('Góc (độ)', fontsize=12)
    ax3.set_title('Góc Theta và Beta theo thời gian', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8, ncol=2)
    
    # 4. Sai số từ giá trị gốc (L_g = 0.35)
    ax4 = plt.subplot(2, 2, 4)
    if trajectories[-1] is not None:  # Index cuối là L_g = 0.35
        ref_traj = trajectories[-1]
        for i, (lg, traj) in enumerate(zip(lg_values, trajectories)):
            if traj is not None and i != len(trajectories)-1:
                # Tính sai số Euclidean
                error = np.sqrt((traj['x_q'] - ref_traj['x_q'])**2 + 
                              (traj['z_q'] - ref_traj['z_q'])**2)
                ax4.plot(traj['t'], error, 
                        color=colors[i], linewidth=2,
                        label=f'L_g = {lg:.2f}m')
        ax4.set_xlabel('Thời gian (s)', fontsize=12)
        ax4.set_ylabel('Sai số (m)', fontsize=12)
        ax4.set_title('Sai số so với L_g = 0.35m', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9)
    
    plt.tight_layout()
    
    # Lưu hình
    output_file = 'minsnap_results/lg_comparison_regenerated.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Đã lưu biểu đồ so sánh: {output_file}")
    
    plt.show()

def print_statistics(trajectories, lg_values):
    """In thống kê so sánh"""
    print("\n" + "="*70)
    print("THỐNG KÊ SO SÁNH")
    print("="*70)
    print(f"{'L_g (m)':<10} {'X_max':<10} {'X_min':<10} {'Z_max':<10} {'Z_min':<10} {'Theta_max(°)':<15}")
    print("-"*70)
    
    for lg, traj in zip(lg_values, trajectories):
        if traj is not None:
            x_max = traj['x_q'].max()
            x_min = traj['x_q'].min()
            z_max = traj['z_q'].max()
            z_min = traj['z_q'].min()
            theta_max = np.rad2deg(np.abs(traj['theta']).max())
            
            print(f"{lg:<10.2f} {x_max:<10.3f} {x_min:<10.3f} {z_max:<10.3f} {z_min:<10.3f} {theta_max:<15.3f}")

# ========== MAIN ==========
def main():
    print("="*70)
    print("SCRIPT SO SÁNH L_g - PHIÊN BẢN CẢI TIẾN")
    print("="*70)
    print("🔄 TÁI TẠO flat_outputs.csv với từng L_g mới")
    print("="*70)
    print("\n⚠️  QUAN TRỌNG:")
    print("   Script này sẽ TÁI TẠO flat_outputs.csv với L_g mới")
    print("   → Quỹ đạo tham chiếu sẽ khác nhau cho mỗi L_g")
    print("   → Sẽ thấy SỰ KHÁC BIỆT RÕ RÀNG hơn nhiều!")
    print("\n❌ LÝ DO script cũ không thấy sự khác biệt:")
    print("   1. flat_outputs.csv được tạo với L_g=0.1m (cố định)")
    print("   2. Khi test L_g khác → chỉ dynamics thay đổi")
    print("   3. Controller đủ mạnh để BÙ TRỪ → quỹ đạo gần giống nhau")
    print("\n✅ GIẢI PHÁP:")
    print("   → Tái tạo flat_outputs.csv với L_g mới")
    print("   → Mỗi L_g có quỹ đạo tham chiếu riêng")
    print("   → Dynamics + Controller đều dùng L_g mới")
    print("="*70)
    
    # 1. Backup files gốc
    print("\n[Bước 1] Backup files gốc...")
    backup_files()
    
    # 2. Chạy simulation với từng giá trị L_g
    trajectories = []
    result_files = []
    
    for lg in LG_VALUES:
        print(f"\n[Bước 2.{LG_VALUES.index(lg)+1}] Test với L_g = {lg} m")
        
        # Sửa đổi giá trị L_g trong các file
        for file in FILES_TO_MODIFY:
            modify_lg_value(file, lg)
        
        # Chạy simulation (bao gồm tái tạo flat_outputs)
        result_file = run_simulation(lg)
        result_files.append(result_file)
        
        # Đọc kết quả
        traj = load_trajectory(result_file) if result_file else None
        trajectories.append(traj)
    
    # 3. Khôi phục files gốc
    print(f"\n[Bước 3] Khôi phục files gốc...")
    restore_files()
    
    # 4. Vẽ biểu đồ so sánh
    print(f"\n[Bước 4] Vẽ biểu đồ so sánh...")
    plot_comparison(trajectories, LG_VALUES)
    
    # 5. In thống kê
    print_statistics(trajectories, LG_VALUES)
    
    print("\n" + "="*70)
    print("✓ HOÀN THÀNH!")
    print("="*70)
    print(f"\nCác file kết quả đã được lưu tại thư mục 'minsnap_results/':")
    for i, (lg, file) in enumerate(zip(LG_VALUES, result_files)):
        if file:
            print(f"  {i+1}. L_g = {lg:.2f}m: {file}")
    print(f"\nBiểu đồ so sánh: minsnap_results/lg_comparison_regenerated.png")
    print(f"Files backup tại: {BACKUP_DIR}/")

if __name__ == "__main__":
    main()
