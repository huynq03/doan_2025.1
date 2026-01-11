import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Đọc dữ liệu từ file CSV
data = pd.read_csv('minsnap_results/ketqua.csv')

# Lấy dữ liệu quỹ đạo x và z
x = data['x_q'].values
z = data['z_q'].values
t = data['t'].values

# Tạo figure với 3 biểu đồ
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Biểu đồ 1: Quỹ đạo X-Z (2D)
axes[0].plot(x, z, 'b-', linewidth=2, label='Quỹ đạo')
axes[0].plot(x[0], z[0], 'go', markersize=10, label='Điểm bắt đầu')
axes[0].plot(x[-1], z[-1], 'ro', markersize=10, label='Điểm kết thúc')
axes[0].set_xlabel('X (m)', fontsize=12)
axes[0].set_ylabel('Z (m)', fontsize=12)
axes[0].set_title('Quỹ đạo XZ', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()
axes[0].axis('equal')

# Biểu đồ 2: X theo thời gian
axes[1].plot(t, x, 'r-', linewidth=2)
axes[1].set_xlabel('Thời gian (s)', fontsize=12)
axes[1].set_ylabel('X (m)', fontsize=12)
axes[1].set_title('Vị trí X theo thời gian', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Biểu đồ 3: Z theo thời gian
axes[2].plot(t, z, 'g-', linewidth=2)
axes[2].set_xlabel('Thời gian (s)', fontsize=12)
axes[2].set_ylabel('Z (m)', fontsize=12)
axes[2].set_title('Vị trí Z theo thời gian', fontsize=13, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('minsnap_results/quydao_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Quỹ đạo đã được vẽ và lưu vào: minsnap_results/quydao_visualization.png")
print(f"✓ Tổng số điểm dữ liệu: {len(t)}")
print(f"✓ Thời gian mô phỏng: {t[0]:.4f} - {t[-1]:.4f} (s)")
print(f"✓ Phạm vi X: [{x.min():.4f}, {x.max():.4f}] (m)")
print(f"✓ Phạm vi Z: [{z.min():.4f}, {z.max():.4f}] (m)")
