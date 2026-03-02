import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# Console output control
# If False: still computes and exports vequydao_metrics.csv, but also prints a short
# best/worst summary (trajectory + beta) that is easy to paste into slides.
PRINT_DETAILED_METRICS = False


# Angle helpers (handle +/-pi discontinuities without wrap-to-pi)
def unwrap_angle(angle_array):
    if angle_array is None:
        return None
    angle_array = np.asarray(angle_array)
    return np.unwrap(angle_array)


def unwrap_and_align(sim_angle, ref_angle):
    """Unwrap both arrays and align sim initial value to ref initial value."""
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

# === Tính sai số so với tham chiếu (flat_outputs nếu có) ===
print("\nTính sai số (so với tham chiếu):")
metrics_rows = []
pos_tol = 0.05  # meters
ang_tol = np.deg2rad(3.0)  # radians

# Determine reference: use flat_outputs if available, else use first trajectory as reference
use_flat_ref = (flat_t is not None and flat_beta is not None and flat_theta is not None)
if use_flat_ref:
    ref_t = flat_t
    ref_x = flat_data['x_q'].values
    ref_z = flat_data['z_q'].values
    ref_theta = flat_theta
    ref_beta = flat_beta
    ref_label = 'flat_outputs'
else:
    # fallback: use first trajectory as reference if more than one
    if len(trajectories) > 0:
        ref = trajectories[0]
        ref_t = ref['t']
        ref_x = ref['x']
        ref_z = ref['z']
        ref_theta = ref['theta'] if ref['theta'] is not None else None
        ref_beta = ref['beta'] if ref['beta'] is not None else None
        ref_label = ref['name']
    else:
        ref_t = None
        ref_x = None
        ref_z = None
        ref_theta = None
        ref_beta = None
        ref_label = None

for traj in trajectories:
    name = traj['name']
    # skip comparing reference to itself when using first trajectory as ref
    if (not use_flat_ref) and name == ref_label:
        continue

    # align by minimum length
    if ref_t is None:
        print(f"Không có tham chiếu phù hợp để so sánh với {name}")
        continue

    n = min(len(traj['t']), len(ref_t))
    t = traj['t'][:n]

    rx = ref_x[:n]
    rz = ref_z[:n]
    rtheta = ref_theta[:n] if (ref_theta is not None) else None
    rbeta = ref_beta[:n] if (ref_beta is not None) else None

    ex = traj['x'][:n] - rx
    ez = traj['z'][:n] - rz
    if traj['theta'] is not None and rtheta is not None:
        theta_sim_u, theta_ref_u = unwrap_and_align(traj['theta'][:n], rtheta)
        etheta = theta_sim_u - theta_ref_u
    else:
        theta_sim_u, theta_ref_u, etheta = None, None, None

    if traj['beta'] is not None and rbeta is not None:
        beta_sim_u, beta_ref_u = unwrap_and_align(traj['beta'][:n], rbeta)
        ebeta = beta_sim_u - beta_ref_u
    else:
        beta_sim_u, beta_ref_u, ebeta = None, None, None

    mx = compute_metrics(ex, sim=traj['x'][:n], ref=rx, t=t, tol=pos_tol)
    mz = compute_metrics(ez, sim=traj['z'][:n], ref=rz, t=t, tol=pos_tol)
    mtheta = compute_metrics(etheta, sim=theta_sim_u, ref=theta_ref_u, t=t, tol=ang_tol) if etheta is not None else None
    mbeta = compute_metrics(ebeta, sim=beta_sim_u, ref=beta_ref_u, t=t, tol=ang_tol) if ebeta is not None else None

    if PRINT_DETAILED_METRICS:
        print("\n" + "-"*60)
        print(f"So sánh {name} vs {ref_label}:")
        print(f"  X: MAE={mx['mae']:.4e} m, RMSE={mx['rmse']:.4e} m, MaxAbs={mx['max_abs']:.4e} m, %within{pos_tol}m={mx['percent_within_tol']:.2f}%")
        print(f"  Z: MAE={mz['mae']:.4e} m, RMSE={mz['rmse']:.4e} m, MaxAbs={mz['max_abs']:.4e} m, %within{pos_tol}m={mz['percent_within_tol']:.2f}%")
        if mtheta is not None:
            print(f"  θ: MAE={mtheta['mae']:.4e} rad ({np.rad2deg(mtheta['mae']):.3f}°), RMSE={mtheta['rmse']:.4e} rad")
        if mbeta is not None:
            print(f"  β: MAE={mbeta['mae']:.4e} rad ({np.rad2deg(mbeta['mae']):.3f}°), RMSE={mbeta['rmse']:.4e} rad")

    # === Slide-friendly summary metrics ===
    # Trajectory (position) error magnitude
    epos = np.sqrt(ex**2 + ez**2)
    pos_rmse = float(np.sqrt(np.mean(epos**2)))
    pos_max = float(np.max(epos))

    # Beta metrics (if available)
    if ebeta is not None:
        beta_rmse = float(np.sqrt(np.mean(ebeta**2)))
        beta_max = float(np.max(np.abs(ebeta)))
    else:
        beta_rmse = np.nan
        beta_max = np.nan

    metrics_rows.append({
        'ref': ref_label,
        'name': name,
        'var': 'pos_mag',
        'mae': float(np.mean(np.abs(epos))),
        'rmse': pos_rmse,
        'max_abs': pos_max,
        'std': float(np.std(epos)),
        'percent_within_tol': float(np.sum(epos < pos_tol) / len(epos) * 100) if len(epos) > 0 else np.nan,
        'IAE': float(np.sum(np.abs(epos)) * (np.mean(np.diff(t)) if len(t) > 1 else 0.0)),
        'ISE': float(np.sum(epos**2) * (np.mean(np.diff(t)) if len(t) > 1 else 0.0)),
        'ITAE': float(np.sum(t * np.abs(epos)) * (np.mean(np.diff(t)) if len(t) > 1 else 0.0)),
        'sMAPE_percent': np.nan
    })

    if ebeta is not None:
        metrics_rows.append({
            'ref': ref_label,
            'name': name,
            'var': 'beta_summary',
            'mae': float(np.mean(np.abs(ebeta))),
            'rmse': float(beta_rmse),
            'max_abs': float(beta_max),
            'std': float(np.std(ebeta)),
            'percent_within_tol': float(np.sum(np.abs(ebeta) < ang_tol) / len(ebeta) * 100) if len(ebeta) > 0 else np.nan,
            'IAE': float(np.sum(np.abs(ebeta)) * (np.mean(np.diff(t)) if len(t) > 1 else 0.0)),
            'ISE': float(np.sum(ebeta**2) * (np.mean(np.diff(t)) if len(t) > 1 else 0.0)),
            'ITAE': float(np.sum(t * np.abs(ebeta)) * (np.mean(np.diff(t)) if len(t) > 1 else 0.0)),
            'sMAPE_percent': np.nan
        })

    # collect for CSV
    for var_name, mm in [('x', mx), ('z', mz)]:
        metrics_rows.append({
            'ref': ref_label,
            'name': name,
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
        })
    if mtheta is not None:
        metrics_rows.append({
            'ref': ref_label,
            'name': name,
            'var': 'theta',
            'mae': mtheta['mae'],
            'rmse': mtheta['rmse'],
            'max_abs': mtheta['max_abs'],
            'std': mtheta['std'],
            'percent_within_tol': mtheta['percent_within_tol'],
            'IAE': mtheta['IAE'],
            'ISE': mtheta['ISE'],
            'ITAE': mtheta['ITAE'],
            'sMAPE_percent': mtheta['sMAPE_percent']
        })
    if mbeta is not None:
        metrics_rows.append({
            'ref': ref_label,
            'name': name,
            'var': 'beta',
            'mae': mbeta['mae'],
            'rmse': mbeta['rmse'],
            'max_abs': mbeta['max_abs'],
            'std': mbeta['std'],
            'percent_within_tol': mbeta['percent_within_tol'],
            'IAE': mbeta['IAE'],
            'ISE': mbeta['ISE'],
            'ITAE': mbeta['ITAE'],
            'sMAPE_percent': mbeta['sMAPE_percent']
        })

# === Print best/worst summary (trajectory + beta) ===
summary_pos = [r for r in metrics_rows if r.get('var') == 'pos_mag' and not np.isnan(r.get('rmse', np.nan))]
summary_beta = [r for r in metrics_rows if r.get('var') == 'beta_summary' and not np.isnan(r.get('rmse', np.nan))]

if summary_pos:
    best_pos = min(summary_pos, key=lambda r: r['rmse'])
    worst_pos = max(summary_pos, key=lambda r: r['rmse'])

    print("\n" + "="*70)
    print("TÓM TẮT (chỉ 2 số/nhóm: RMSE và Max):")
    print(f"Ref: {best_pos['ref']}")
    print("\n[QUỸ ĐẠO |e_pos| = sqrt(e_x^2 + e_z^2)]")
    print(f"  Ít sai số nhất:  {best_pos['name']}  RMSE={best_pos['rmse']:.6f} m   Max={best_pos['max_abs']:.6f} m")
    print(f"  Nhiều sai số nhất: {worst_pos['name']}  RMSE={worst_pos['rmse']:.6f} m   Max={worst_pos['max_abs']:.6f} m")

if summary_beta:
    best_beta = min(summary_beta, key=lambda r: r['rmse'])
    worst_beta = max(summary_beta, key=lambda r: r['rmse'])
    print("\n[GÓC β (beta)]")
    print(f"  Ít sai số nhất:  {best_beta['name']}  RMSE={best_beta['rmse']:.6f} rad  Max={best_beta['max_abs']:.6f} rad")
    print(f"  Nhiều sai số nhất: {worst_beta['name']}  RMSE={worst_beta['rmse']:.6f} rad  Max={worst_beta['max_abs']:.6f} rad")

# Save metrics CSV
if metrics_rows:
    import pandas as pd
    out_metrics = 'minsnap_results/vequydao_metrics.csv'
    try:
        pd.DataFrame(metrics_rows).to_csv(out_metrics, index=False)
        print(f"\n✓ Lưu metrics: {out_metrics}")
    except Exception as e:
        print(f"✗ Không thể lưu metrics: {e}")
