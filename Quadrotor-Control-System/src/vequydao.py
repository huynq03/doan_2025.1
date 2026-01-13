import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Cấu hình gains để vẽ
GAIN_CONFIGS = [
    {"name": "Thấp", "file": "minsnap_results/ketqua_gains_Thấp.csv", "color": "#2E86DE", "linestyle": "-"},
    {"name": "Vừa", "file": "minsnap_results/ketqua_gains_Vừa.csv", "color": "#EE5A6F", "linestyle": "-"},
    {"name": "Cao", "file": "minsnap_results/ketqua_gains_Cao.csv", "color": "#26DE81", "linestyle": "-"},
    {"name": "Rất cao", "file": "minsnap_results/ketqua_gains_ratcao.csv", "color": "#6A1F9C", "linestyle": "-"}
]

# Đọc dữ liệu từ các file CSV
trajectories = []
for config in GAIN_CONFIGS:
    if os.path.exists(config['file']):
        data = pd.read_csv(config['file'])
        
        # Kiểm tra các cột có sẵn
        print(f"\n[{config['name']}] Các cột có sẵn: {list(data.columns)}")
        
        # Lấy theta từ cột 'beta' hoặc 'theta' nếu có
        theta_col = 'theta' if 'theta' in data.columns else ('beta' if 'beta' in data.columns else None)
        theta_data = data[theta_col].values if theta_col else None
        
        trajectories.append({
            'name': config['name'],
            'x': data['x_q'].values,
            'z': data['z_q'].values,
            't': data['t'].values,
            'theta': theta_data,
            'beta': data['beta'].values if 'beta' in data.columns else None,
            'color': config['color'],
            'linestyle': config['linestyle']
        })
        print(f"✓ Đọc {config['name']}: {len(data)} điểm")
    else:
        print(f"✗ Không tìm thấy: {config['file']}")

if not trajectories:
    print("Lỗi: Không có dữ liệu nào!")
    exit(1)

# Đọc điểm đầu và cuối từ flat_outputs.csv
flat_outputs_file = 'minsnap_results/flat_outputs.csv'
if os.path.exists(flat_outputs_file):
    flat_data = pd.read_csv(flat_outputs_file)
    print(f"\n[flat_outputs.csv] Các cột có sẵn: {list(flat_data.columns)}")
    
    start_x, start_z = flat_data['x_q'].iloc[0], flat_data['z_q'].iloc[0]
    end_x, end_z = flat_data['x_q'].iloc[-1], flat_data['z_q'].iloc[-1]
    flat_t = flat_data['t'].values if 't' in flat_data.columns else None
    flat_beta = flat_data['beta'].values if 'beta' in flat_data.columns else None
    flat_theta = flat_data['theta'].values if 'theta' in flat_data.columns else None
    print(f"✓ Đọc điểm đầu/cuối từ {flat_outputs_file}")
else:
    print(f"✗ Không tìm thấy {flat_outputs_file}, dùng điểm từ trajectory đầu tiên")
    start_x, start_z = trajectories[0]['x'][0], trajectories[0]['z'][0]
    end_x, end_z = trajectories[0]['x'][-1], trajectories[0]['z'][-1]
    flat_t = None
    flat_beta = None
    flat_theta = None

# Biểu đồ 1: Quỹ đạo X-Z (2D)
fig1 = plt.figure(figsize=(8, 6))
for traj in trajectories:
    plt.plot(traj['x'], traj['z'], color=traj['color'], linestyle=traj['linestyle'], 
             linewidth=3, label=traj['name'], alpha=0.85)
# Điểm bắt đầu và kết thúc (từ flat_outputs.csv)
plt.plot(start_x, start_z, 'go', markersize=8, label='Start', zorder=5)
plt.plot(end_x, end_z, 'ro', markersize=8, label='End', zorder=5)
plt.xlabel('X (m)', fontsize=13)
plt.ylabel('Z (m)', fontsize=13)
plt.title('Quỹ đạo với 3 cấu hình gains', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.axis('equal')
plt.tight_layout()
plt.savefig('minsnap_results/quydao_3gains_xz.png', dpi=150, bbox_inches='tight')
 

# Biểu đồ 2: X và Z theo thời gian
fig2 = plt.figure(figsize=(10, 6))
for traj in trajectories:
    plt.plot(traj['t'], traj['x'], color=traj['color'], linestyle=traj['linestyle'], 
             linewidth=3, label=f"X - {traj['name']}", alpha=0.85)
    plt.plot(traj['t'], traj['z'], color=traj['color'], linestyle=':', 
             linewidth=3, label=f"Z - {traj['name']}", alpha=0.85)
plt.xlabel('Thời gian (s)', fontsize=13)
plt.ylabel('Vị trí (m)', fontsize=13)
plt.title('Vị trí x và z theo thời gian', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig('minsnap_results/quydao_3gains_time.png', dpi=150, bbox_inches='tight')
 

# Biểu đồ 3: So sánh Beta (từ flat_outputs) và Theta (từ gains) theo thời gian
fig3 = plt.figure(figsize=(10, 6))
if flat_t is not None and flat_beta is not None:
    plt.plot(flat_t, flat_beta, color='red', linestyle='--', linewidth=2.5, label='Beta (mong muốn)', alpha=0.85)
if flat_t is not None and flat_theta is not None:
    plt.plot(flat_t, flat_theta, color='red', linestyle='-', linewidth=2.5, label='Theta (mong muốn)', alpha=0.75)

for traj in trajectories:
    if traj['theta'] is not None:
        plt.plot(traj['t'], traj['theta'], color=traj['color'], linestyle=traj['linestyle'], 
                 linewidth=2.5, label=f"Theta - {traj['name']}", alpha=0.75)
    # Plot Beta for this trajectory if available
    if traj['beta'] is not None:
        plt.plot(traj['t'], traj['beta'], color=traj['color'], linestyle='-', 
                 linewidth=2.0, label=f"Beta - {traj['name']}", alpha=0.75)

plt.xlabel('Thời gian (s)', fontsize=13)
plt.ylabel('Góc (rad)', fontsize=13)
plt.title('So sánh Beta và Theta theo thời gian', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, ncol=2)
plt.tight_layout()
plt.savefig('minsnap_results/quydao_3gains_beta_theta.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ So sánh Beta-Theta đã được lưu vào: minsnap_results/quydao_3gains_beta_theta.png")

print(f"\n✓ Quỹ đạo X-Z đã được lưu vào: minsnap_results/quydao_3gains_xz.png")
print(f"✓ Quỹ đạo theo thời gian đã được lưu vào: minsnap_results/quydao_3gains_time.png")
print(f"✓ So sánh Beta-Theta đã được lưu vào: minsnap_results/quydao_3gains_beta_theta.png")
for traj in trajectories:
    print(f"\n[{traj['name']}]")
    print(f"  Số điểm: {len(traj['t'])}")
    print(f"  Thời gian: {traj['t'][0]:.4f} - {traj['t'][-1]:.4f} (s)")
    print(f"  X: [{traj['x'].min():.4f}, {traj['x'].max():.4f}] (m)")
    print(f"  Z: [{traj['z'].min():.4f}, {traj['z'].max():.4f}] (m)")
