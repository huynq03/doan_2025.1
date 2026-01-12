import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Cấu hình gains để vẽ
GAIN_CONFIGS = [
    {"name": "Thấp", "file": "minsnap_results/ketqua_gains_Thấp.csv", "color": "#2E86DE", "linestyle": "-"},
    {"name": "Vừa", "file": "minsnap_results/ketqua_gains_Vừa.csv", "color": "#EE5A6F", "linestyle": "-"},
    {"name": "Cao", "file": "minsnap_results/ketqua_gains_Cao.csv", "color": "#26DE81", "linestyle": "-"}
]

# Đọc dữ liệu từ các file CSV
trajectories = []
for config in GAIN_CONFIGS:
    if os.path.exists(config['file']):
        data = pd.read_csv(config['file'])
        trajectories.append({
            'name': config['name'],
            'x': data['x_q'].values,
            'z': data['z_q'].values,
            't': data['t'].values,
            'color': config['color'],
            'linestyle': config['linestyle']
        })
        print(f"✓ Đọc {config['name']}: {len(data)} điểm")
    else:
        print(f"✗ Không tìm thấy: {config['file']}")

if not trajectories:
    print("Lỗi: Không có dữ liệu nào!")
    exit(1)

# Tạo figure với 2 biểu đồ
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Biểu đồ 1: Quỹ đạo X-Z (2D)
for traj in trajectories:
    axes[0].plot(traj['x'], traj['z'], color=traj['color'], linestyle=traj['linestyle'], 
                 linewidth=3, label=traj['name'], alpha=0.85)
# Điểm bắt đầu và kết thúc (dùng trajectory đầu tiên)
axes[0].plot(trajectories[0]['x'][0], trajectories[0]['z'][0], 'go', markersize=5, label='Start')
axes[0].plot(trajectories[0]['x'][-1], trajectories[0]['z'][-1], 'ro', markersize=5, label='End')
axes[0].set_xlabel('X (m)', fontsize=13)
axes[0].set_ylabel('Z (m)', fontsize=13)
axes[0].set_title('Quỹ đạo với 3 cấu hình gains', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=11)
axes[0].axis('equal')

# Biểu đồ 2: X và Z theo thời gian
for traj in trajectories:
    axes[1].plot(traj['t'], traj['x'], color=traj['color'], linestyle=traj['linestyle'], 
                 linewidth=3, label=f"X - {traj['name']}", alpha=0.85)
    axes[1].plot(traj['t'], traj['z'], color=traj['color'], linestyle=':', 
                 linewidth=3, label=f"Z - {traj['name']}", alpha=0.85)
axes[1].set_xlabel('Thời gian (s)', fontsize=13)
axes[1].set_ylabel('Vị trí (m)', fontsize=13)
axes[1].set_title('Vị trí x và z theo thời gian', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=9, ncol=2)

plt.tight_layout()
plt.savefig('minsnap_results/quydao_3gains_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Quỹ đạo 3 gains đã được vẽ và lưu vào: minsnap_results/quydao_3gains_visualization.png")
for traj in trajectories:
    print(f"\n[{traj['name']}]")
    print(f"  Số điểm: {len(traj['t'])}")
    print(f"  Thời gian: {traj['t'][0]:.4f} - {traj['t'][-1]:.4f} (s)")
    print(f"  X: [{traj['x'].min():.4f}, {traj['x'].max():.4f}] (m)")
    print(f"  Z: [{traj['z'].min():.4f}, {traj['z'].max():.4f}] (m)")
