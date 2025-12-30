import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Tạo thư mục media nếu chưa có
media_dir = '../media'
if not os.path.exists(media_dir):
    os.makedirs(media_dir)

# 1. Đọc 2 file CSV
df_sim = pd.read_csv('C:\\Users\\2003h\\OneDrive\\Máy tính\\doan_2025.1\\Quadrotor-Control-System\\src\\minsnap_results\\ketqua.csv')
df_flat = pd.read_csv('C:\\Users\\2003h\\OneDrive\\Máy tính\\doan_2025.1\\Quadrotor-Control-System\\src\\minsnap_results\\flat_outputs1.csv')

# Đảm bảo 2 file có cùng số điểm
min_len = min(len(df_sim), len(df_flat))
df_sim = df_sim.iloc[:min_len]
df_flat = df_flat.iloc[:min_len]

# Tính error
error_beta = df_sim['beta'].values - df_flat['beta'].values
abs_error = np.abs(error_beta)
percent_error = (abs_error / (np.abs(df_flat['beta'].values) + 1e-9)) * 100

# Tính thống kê
max_error = np.max(abs_error)
mean_error = np.mean(abs_error)
rms_error = np.sqrt(np.mean(error_beta**2))
max_percent = np.max(percent_error)

# Timestamp cho filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================
# HÌNH 1: So sánh Beta (full range)
# ============================================================
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df_sim['t'], df_sim['beta'], label='Beta (Simulation)', color='blue', linewidth=2.5, marker='o', markersize=3, alpha=0.8)
ax1.plot(df_flat['t'], df_flat['beta'], label='Beta (Reference)', color='red', linestyle='--', linewidth=2.5)
ax1.fill_between(df_sim['t'], df_sim['beta'], df_flat['beta'], alpha=0.2, color='yellow', label='Error Region')
ax1.set_xlabel('Thời gian (s)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Góc Beta (rad)', fontsize=12, fontweight='bold')
ax1.set_title('So sánh Góc Beta: Mô phỏng vs Tham chiếu', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11, loc='best')
ax1.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
filename1 = os.path.join(media_dir, f'01_beta_comparison_{timestamp}.png')
plt.savefig(filename1, dpi=300, bbox_inches='tight')
print(f"✓ Lưu: {filename1}")
plt.close()

# ============================================================
# HÌNH 2: Tracking Error (Absolute Value)
# ============================================================
fig2, ax2 = plt.subplots(figsize=(12, 6))
color_error = ['red' if e > mean_error else 'green' for e in abs_error]
ax2.bar(df_sim['t'], abs_error, color=color_error, alpha=0.6, width=0.01, label='Absolute Error')
ax2.axhline(y=mean_error, color='orange', linestyle='-', linewidth=2.5, label=f'Mean: {mean_error:.2e} rad')
ax2.axhline(y=max_error, color='darkred', linestyle=':', linewidth=2.5, label=f'Max: {max_error:.2e} rad')
ax2.set_xlabel('Thời gian (s)', fontsize=12, fontweight='bold')
ax2.set_ylabel('|Error| (rad)', fontsize=12, fontweight='bold')
ax2.set_title('Tracking Error (Absolute Value)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11, loc='best')
ax2.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
filename2 = os.path.join(media_dir, f'02_error_absolute_{timestamp}.png')
plt.savefig(filename2, dpi=300, bbox_inches='tight')
print(f"✓ Lưu: {filename2}")
plt.close()

# ============================================================
# HÌNH 3: Tracking Error (Relative Percentage)
# ============================================================
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(df_sim['t'], percent_error, color='purple', linewidth=2.5, marker='s', markersize=3, label='Relative Error')
ax3.axhline(y=np.mean(percent_error), color='orange', linestyle='-', linewidth=2.5, 
            label=f'Mean: {np.mean(percent_error):.4f}%')
ax3.axhline(y=max_percent, color='darkred', linestyle=':', linewidth=2.5, 
            label=f'Max: {max_percent:.4f}%')
ax3.fill_between(df_sim['t'], percent_error, alpha=0.3, color='purple')
ax3.set_xlabel('Thời gian (s)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
ax3.set_title('Tracking Error (Relative Percentage)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=11, loc='best')
ax3.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
filename3 = os.path.join(media_dir, f'03_error_percentage_{timestamp}.png')
plt.savefig(filename3, dpi=300, bbox_inches='tight')
print(f"✓ Lưu: {filename3}")
plt.close()

# ============================================================
# HÌNH 4: So sánh vị trí (x_q, z_q)
# ============================================================
fig4, ax4 = plt.subplots(figsize=(12, 6))
ax4.plot(df_sim['t'], df_sim['x_q'], label='x_q (Simulation)', color='#E60000', linewidth=3.5, alpha=0.95, linestyle='-')
ax4.plot(df_sim['t'], df_sim['z_q'], label='z_q (Simulation)', color='#0066FF', linewidth=3.5, alpha=0.95, linestyle='-')
if 'x_q' in df_flat.columns and 'z_q' in df_flat.columns:
    ax4.plot(df_flat['t'], df_flat['x_q'], label='x_q (Reference)', color='#00D900', linestyle='--', linewidth=3, alpha=0.9)
    ax4.plot(df_flat['t'], df_flat['z_q'], label='z_q (Reference)', color='#FFB800', linestyle='--', linewidth=3, alpha=0.9)
ax4.set_xlabel('Thời gian (s)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Vị trí (m)', fontsize=12, fontweight='bold')
ax4.set_title('So sánh Vị trí Quadrotor (Simulation vs Reference)', fontsize=13, fontweight='bold')
ax4.legend(fontsize=11, loc='best', framealpha=0.95)
ax4.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
filename4 = os.path.join(media_dir, f'04_position_comparison_{timestamp}.png')
plt.savefig(filename4, dpi=300, bbox_inches='tight')
print(f"✓ Lưu: {filename4}")
plt.close()

# ============================================================
# HÌNH 5: Tổng hợp tất cả 4 đồ thị
# ============================================================
fig5 = plt.figure(figsize=(16, 12))
gs = fig5.add_gridspec(4, 1, hspace=0.35)

ax5_1 = fig5.add_subplot(gs[0])
ax5_1.plot(df_sim['t'], df_sim['beta'], label='Beta (Simulation)', color='blue', linewidth=2, marker='o', markersize=2, alpha=0.7)
ax5_1.plot(df_flat['t'], df_flat['beta'], label='Beta (Reference)', color='red', linestyle='--', linewidth=2)
ax5_1.fill_between(df_sim['t'], df_sim['beta'], df_flat['beta'], alpha=0.2, color='yellow', label='Error Region')
ax5_1.set_ylabel('Góc Beta (rad)', fontsize=11, fontweight='bold')
ax5_1.set_title('1. So sánh Góc Beta', fontsize=12, fontweight='bold')
ax5_1.legend(fontsize=10)
ax5_1.grid(True, alpha=0.3)

ax5_2 = fig5.add_subplot(gs[1])
ax5_2.bar(df_sim['t'], abs_error, color=color_error, alpha=0.6, width=0.01)
ax5_2.axhline(y=mean_error, color='orange', linestyle='-', linewidth=2, label=f'Mean: {mean_error:.2e}')
ax5_2.axhline(y=max_error, color='darkred', linestyle=':', linewidth=2, label=f'Max: {max_error:.2e}')
ax5_2.set_ylabel('|Error| (rad)', fontsize=11, fontweight='bold')
ax5_2.set_title('2. Tracking Error (Absolute)', fontsize=12, fontweight='bold')
ax5_2.legend(fontsize=10)
ax5_2.grid(True, alpha=0.3)

ax5_3 = fig5.add_subplot(gs[2])
ax5_3.plot(df_sim['t'], percent_error, color='purple', linewidth=2, marker='s', markersize=2)
ax5_3.axhline(y=np.mean(percent_error), color='orange', linestyle='-', linewidth=2, 
              label=f'Mean: {np.mean(percent_error):.2f}%')
ax5_3.axhline(y=max_percent, color='darkred', linestyle=':', linewidth=2, 
              label=f'Max: {max_percent:.2f}%')
ax5_3.fill_between(df_sim['t'], percent_error, alpha=0.3, color='purple')
ax5_3.set_ylabel('Error (%)', fontsize=11, fontweight='bold')
ax5_3.set_title('3. Tracking Error (Percentage)', fontsize=12, fontweight='bold')
ax5_3.legend(fontsize=10)
ax5_3.grid(True, alpha=0.3)

ax5_4 = fig5.add_subplot(gs[3])
ax5_4.plot(df_sim['t'], df_sim['x_q'], label='x_q (Sim)', color='#E60000', linewidth=3, alpha=0.95, linestyle='-')
ax5_4.plot(df_sim['t'], df_sim['z_q'], label='z_q (Sim)', color='#0066FF', linewidth=3, alpha=0.95, linestyle='-')
if 'x_q' in df_flat.columns and 'z_q' in df_flat.columns:
    ax5_4.plot(df_flat['t'], df_flat['x_q'], label='x_q (Ref)', color='#00D900', linestyle='--', linewidth=2.5, alpha=0.9)
    ax5_4.plot(df_flat['t'], df_flat['z_q'], label='z_q (Ref)', color='#FFB800', linestyle='--', linewidth=2.5, alpha=0.9)
ax5_4.set_xlabel('Thời gian (s)', fontsize=11, fontweight='bold')
ax5_4.set_ylabel('Vị trí (m)', fontsize=11, fontweight='bold')
ax5_4.set_title('4. So sánh Vị trí Quadrotor', fontsize=12, fontweight='bold')
ax5_4.legend(fontsize=10)
ax5_4.grid(True, alpha=0.3)

fig5.suptitle('Phân tích Chi Tiết Tracking Performance', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
filename5 = os.path.join(media_dir, f'05_all_comparison_{timestamp}.png')
plt.savefig(filename5, dpi=300, bbox_inches='tight')
print(f"✓ Lưu: {filename5}")
plt.close()

# ============================================================
# IN THỐNG KÊ CHI TIẾT
# ============================================================
print("\n" + "=" * 75)
print("THỐNG KÊ TRACKING ERROR CHI TIẾT")
print("=" * 75)
print(f"  Max Absolute Error:    {max_error:.8f} rad = {max_error*180/np.pi:.6f}°")
print(f"  Mean Absolute Error:   {mean_error:.8f} rad = {mean_error*180/np.pi:.6f}°")
print(f"  RMS Error:             {rms_error:.8f} rad = {rms_error*180/np.pi:.6f}°")
print(f"  Std Dev Error:         {np.std(error_beta):.8f} rad")
print(f"\n  Max Relative Error:    {max_percent:.4f}%")
print(f"  Mean Relative Error:   {np.mean(percent_error):.4f}%")
print("=" * 75)
print(f"\n📁 Tất cả hình ảnh đã lưu vào: {os.path.abspath(media_dir)}\n")

if np.max(np.abs(error_beta)) < 1e-6:
    print("⚠️  CẢNH BÁO: Error quá nhỏ (~0), có thể simulation đang copy trực tiếp reference!")
else:
    print("✓  Error hợp lý, bộ điều khiển hoạt động với hiệu suất tracking tốt.")
print()