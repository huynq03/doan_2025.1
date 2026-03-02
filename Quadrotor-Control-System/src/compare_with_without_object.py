# compare_with_without_object.py
"""
Script so sánh kết quả mô phỏng giữa:
- Không có vật (ketqua.csv)
- Có vật 27g (ketqua_with_object.csv)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def compare_results():
    # Đường dẫn file
    file_without = "minsnap_results/ketqua.csv"
    file_with = "minsnap_results/ketqua_with_object.csv"
    
    # Kiểm tra tồn tại
    if not os.path.exists(file_without):
        print(f"[ERROR] Không tìm thấy file: {file_without}")
        print("Chạy mô phỏng gốc trước: python dieu_khien.py --simulate --save_csv minsnap_results/ketqua.csv")
        return
    
    if not os.path.exists(file_with):
        print(f"[ERROR] Không tìm thấy file: {file_with}")
        print("Chạy mô phỏng với vật trước: python test_with_object.py")
        return
    
    # Đọc dữ liệu
    df_without = pd.read_csv(file_without)
    df_with = pd.read_csv(file_with)
    
    print("="*70)
    print("SO SÁNH KẾT QUẢ: KHÔNG CÓ VẬT vs CÓ VẬT 27g")
    print("="*70)
    
    # Tính sai số
    print("\n1. SAI SỐ VỊ TRÍ TRUNG BÌNH:")
    print("-" * 50)
    
    # Sai số x
    err_x_without = np.abs(df_without['x_q'] - df_without['x_q'].iloc[-1]).mean()
    err_x_with = np.abs(df_with['x_q'] - df_with['x_q'].iloc[-1]).mean()
    print(f"   Sai số X (không vật): {err_x_without:.4f} m")
    print(f"   Sai số X (có vật):    {err_x_with:.4f} m")
    print(f"   Chênh lệch:           {abs(err_x_with - err_x_without):.4f} m ({((err_x_with/err_x_without - 1)*100):.2f}%)")
    
    # Sai số z
    err_z_without = np.abs(df_without['z_q'] - df_without['z_q'].iloc[-1]).mean()
    err_z_with = np.abs(df_with['z_q'] - df_with['z_q'].iloc[-1]).mean()
    print(f"\n   Sai số Z (không vật): {err_z_without:.4f} m")
    print(f"   Sai số Z (có vật):    {err_z_with:.4f} m")
    print(f"   Chênh lệch:           {abs(err_z_with - err_z_without):.4f} m ({((err_z_with/err_z_without - 1)*100):.2f}%)")
    
    print("\n2. THRUST (U1) TRUNG BÌNH:")
    print("-" * 50)
    u1_without = df_without['u1'].mean()
    u1_with = df_with['u1'].mean()
    print(f"   U1 (không vật): {u1_without:.4f} N")
    print(f"   U1 (có vật):    {u1_with:.4f} N")
    print(f"   Chênh lệch:     {u1_with - u1_without:.4f} N ({((u1_with/u1_without - 1)*100):.2f}%)")
    print(f"   (Lý thuyết: cộng thêm 0.027*9.81 = {0.027*9.81:.4f} N)")
    
    print("\n3. GÓC BETA:")
    print("-" * 50)
    beta_min_without = np.rad2deg(df_without['beta'].min())
    beta_min_with = np.rad2deg(df_with['beta'].min())
    print(f"   Beta min (không vật): {beta_min_without:.2f}°")
    print(f"   Beta min (có vật):    {beta_min_with:.2f}°")
    
    # Vẽ biểu đồ so sánh
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('So sánh: Không có vật vs Có vật 27g', fontsize=16, fontweight='bold')
    
    t_without = df_without['t']
    t_with = df_with['t']
    
    # X position
    axes[0, 0].plot(t_without, df_without['x_q'], 'b-', label='Không vật', linewidth=2)
    axes[0, 0].plot(t_with, df_with['x_q'], 'r--', label='Có vật 27g', linewidth=2)
    axes[0, 0].set_ylabel('x (m)')
    axes[0, 0].set_title('Vị trí X')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Z position
    axes[0, 1].plot(t_without, df_without['z_q'], 'b-', label='Không vật', linewidth=2)
    axes[0, 1].plot(t_with, df_with['z_q'], 'r--', label='Có vật 27g', linewidth=2)
    axes[0, 1].set_ylabel('z (m)')
    axes[0, 1].set_title('Vị trí Z')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Thrust u1
    axes[1, 0].plot(t_without, df_without['u1'], 'b-', label='Không vật', linewidth=2)
    axes[1, 0].plot(t_with, df_with['u1'], 'r--', label='Có vật 27g', linewidth=2)
    axes[1, 0].set_ylabel('u1 (N)')
    axes[1, 0].set_title('Lực đẩy U1')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Theta
    axes[1, 1].plot(t_without, np.rad2deg(df_without['theta']), 'b-', label='Không vật', linewidth=2)
    axes[1, 1].plot(t_with, np.rad2deg(df_with['theta']), 'r--', label='Có vật 27g', linewidth=2)
    axes[1, 1].set_ylabel('theta (°)')
    axes[1, 1].set_title('Góc Theta')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Beta
    axes[2, 0].plot(t_without, np.rad2deg(df_without['beta']), 'b-', label='Không vật', linewidth=2)
    axes[2, 0].plot(t_with, np.rad2deg(df_with['beta']), 'r--', label='Có vật 27g', linewidth=2)
    axes[2, 0].axhline(y=-20, color='green', linestyle=':', label='Ngưỡng gắp (-20°)')
    axes[2, 0].set_xlabel('Thời gian (s)')
    axes[2, 0].set_ylabel('beta (°)')
    axes[2, 0].set_title('Góc Beta (Gripper)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # XY trajectory
    axes[2, 1].plot(df_without['x_q'], df_without['z_q'], 'b-', label='Không vật', linewidth=2)
    axes[2, 1].plot(df_with['x_q'], df_with['z_q'], 'r--', label='Có vật 27g', linewidth=2)
    axes[2, 1].plot(df_without['x_q'].iloc[0], df_without['z_q'].iloc[0], 'go', markersize=10, label='Start')
    axes[2, 1].plot(df_without['x_q'].iloc[-1], df_without['z_q'].iloc[-1], 'rs', markersize=10, label='End')
    axes[2, 1].set_xlabel('x (m)')
    axes[2, 1].set_ylabel('z (m)')
    axes[2, 1].set_title('Quỹ đạo X-Z')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].axis('equal')
    
    plt.tight_layout()
    
    # Lưu hình
    output_file = "minsnap_results/comparison_with_without_object.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n4. BIỂU ĐỒ SO SÁNH:")
    print("-" * 50)
    print(f"   Đã lưu tại: {output_file}")
    
    plt.show()
    
    print("\n" + "="*70)

if __name__ == "__main__":
    compare_results()
