# -*- coding: utf-8 -*-
"""
Script để vẽ 4 đồ thị từ các file kết quả CSV
- Đồ thị 1: Quỹ đạo X-Z
- Đồ thị 2: X và Z theo thời gian
- Đồ thị 3: Góc theta và beta theo thời gian
- Đồ thị 4: Sai số so với trường hợp tham chiếu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# Cấu hình
# Thư mục chứa các file CSV kết quả
RESULTS_DIR = "minsnap_results"
# Danh sách các file CSV cụ thể cần vẽ
CSV_FILES = [
    "ketqua_lg0.25.csv",
    "ketqua_lg0.30.csv",
    "ketqua.csv",           # L_g = 0.35 m (vừa vẽ vừa làm tham chiếu)
    "ketqua_lg0.40.csv",
    "ketqua_lg0.45.csv"
]
# File tham chiếu để tính sai số
REFERENCE_FILE = "ketqua.csv"  # L_g = 0.35 m
# Thư mục lưu đồ thị
OUTPUT_DIR = "minsnap_results"

# Console output control
PRINT_DETAILED_METRICS = False  # Nếu False: chỉ in best/worst summary


# ========== CÁC HÀM HELPER TÍNH SAI SỐ ==========

def unwrap_angle(angle_array):
    """Xử lý góc để tránh discontinuities tại +/-pi"""
    if angle_array is None:
        return None
    angle_array = np.asarray(angle_array)
    return np.unwrap(angle_array)


def unwrap_and_align(sim_angle, ref_angle):
    """Unwrap cả hai mảng góc và align giá trị ban đầu của sim với ref
    
    Điều này loại bỏ +/-pi jumps và ngăn chặn offset 2*pi không cần thiết
    """
    sim_u = unwrap_angle(sim_angle)
    ref_u = unwrap_angle(ref_angle)
    if sim_u is None or ref_u is None or len(sim_u) == 0 or len(ref_u) == 0:
        return sim_u, ref_u
    sim_u = sim_u - sim_u[0] + ref_u[0]
    return sim_u, ref_u


def compute_metrics(e, sim=None, ref=None, t=None, tol=0.0):
    """Tính các chỉ số sai số cho mảng error (e = sim - ref)
    
    Returns:
        dict: Chứa MAE, RMSE, max_abs, std, percent_within_tol, IAE, ISE, ITAE, sMAPE
    """
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

    # sMAPE cần mảng sim và ref
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


# ========== KẾT THÚC HÀM HELPER ==========

def load_csv_file(filepath):
    """Load dữ liệu từ file CSV"""
    if not os.path.exists(filepath):
        print(f"⚠️  File không tồn tại: {filepath}")
        return None
    
    try:
        df = pd.read_csv(filepath)
        required_cols = ['t', 'x_q', 'z_q', 'theta', 'beta']
        
        if not all(col in df.columns for col in required_cols):
            print(f"⚠️  File thiếu cột cần thiết: {filepath}")
            return None
        
        return {
            't': df['t'].values,
            'x_q': df['x_q'].values,
            'z_q': df['z_q'].values,
            'theta': df['theta'].values,
            'beta': df['beta'].values,
            'filename': os.path.basename(filepath)
        }
    except Exception as e:
        print(f"✗ Lỗi khi đọc file {filepath}: {e}")
        return None

def extract_label_from_filename(filename):
    """Trích xuất nhãn từ tên file (ví dụ: ketqua_lg0.35.csv -> 0.35)"""
    import re
    # Xử lý file ketqua.csv đặc biệt (là L_g = 0.35 m)
    if filename == "ketqua.csv":
        return "L_g = 0.35 m"
    match = re.search(r'lg([\d.]+)', filename)
    if match:
        return f"L_g = {match.group(1)} m"
    return filename.replace('.csv', '')

def plot_4_graphs(data_list, reference_idx=None):
    """
    Vẽ 4 đồ thị so sánh
    
    Parameters:
    -----------
    data_list : list of dict
        Danh sách các dict chứa dữ liệu từ CSV
    reference_idx : int, optional
        Index của dữ liệu tham chiếu để tính sai số (vẫn vẽ lên đồ thị)
    """
    if not data_list or len(data_list) == 0:
        print("✗ Không có dữ liệu để vẽ")
        return
    
    # Tạo màu sắc cho mỗi đường
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_list)))
    
    # ========== ĐỒ THỊ 1: QUỸ ĐẠO X-Z ==========
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for i, data in enumerate(data_list):
        if data is not None:
            label = extract_label_from_filename(data['filename'])
            ax1.plot(data['x_q'], data['z_q'], 
                    color=colors[i], linewidth=2, label=label)
    
    ax1.set_xlabel('x (m)', fontsize=12)
    ax1.set_ylabel('z (m)', fontsize=12)
    ax1.set_title('Các quỹ đạo chuyển động', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    fig1.tight_layout()
    
    # ========== ĐỒ THỊ 2: X VÀ Z THEO THỜI GIAN ==========
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for i, data in enumerate(data_list):
        if data is not None:
            label = extract_label_from_filename(data['filename'])
            ax2.plot(data['t'], data['x_q'], 
                    color=colors[i], linewidth=2, linestyle='-', 
                    label=f'X - {label}')
            ax2.plot(data['t'], data['z_q'], 
                    color=colors[i], linewidth=2, linestyle='--', 
                    label=f'Z - {label}')
    
    ax2.set_xlabel('Thời gian (s)', fontsize=12)
    ax2.set_ylabel('Vị trí (m)', fontsize=12)
    ax2.set_title('Vị trí x và z theo thời gian', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, ncol=2)
    fig2.tight_layout()
    
    # ========== ĐỒ THỊ 3: GÓC THETA VÀ BETA ==========
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for i, data in enumerate(data_list):
        if data is not None:
            label = extract_label_from_filename(data['filename'])
            ax3.plot(data['t'], np.rad2deg(data['theta']), 
                    color=colors[i], linewidth=2, linestyle='-', 
                    label=f'θ - {label}')
            ax3.plot(data['t'], np.rad2deg(data['beta']), 
                    color=colors[i], linewidth=2, linestyle='--', 
                    label=f'β - {label}')
    
    ax3.set_xlabel('Thời gian (s)', fontsize=12)
    ax3.set_ylabel('Góc (°)', fontsize=12)
    ax3.set_title(r'Góc $\theta$ và $\beta$ theo thời gian', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8, ncol=2)
    fig3.tight_layout()
    
    # ========== ĐỒ THỊ 4: SAI SỐ SO VỚI THAM CHIẾU ==========
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    
    # Sử dụng dữ liệu tham chiếu nếu có
    if reference_idx is not None and reference_idx < len(data_list):
        ref_data = data_list[reference_idx]
        ref_label = extract_label_from_filename(ref_data['filename'])
        
        for i, data in enumerate(data_list):
            if data is not None and i != reference_idx:
                # Tính sai số vị trí (Euclidean distance)
                n = min(len(data['t']), len(ref_data['t']))
                error = np.sqrt(
                    (data['x_q'][:n] - ref_data['x_q'][:n])**2 + 
                    (data['z_q'][:n] - ref_data['z_q'][:n])**2
                )
                label = extract_label_from_filename(data['filename'])
                ax4.plot(data['t'][:n], error, 
                        color=colors[i], linewidth=2, label=label)
        
        ax4.set_xlabel('Thời gian (s)', fontsize=12)
        ax4.set_ylabel('Sai số vị trí (m)', fontsize=12)
        ax4.set_title(f'Sai số so với {ref_label}', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'Không có dữ liệu tham chiếu', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    
    fig4.tight_layout()
    
    # ========== LƯU CÁC ĐỒ THỊ ==========
    output_files = {
        'fig1': os.path.join(OUTPUT_DIR, 'plot_xz_trajectory.png'),
        'fig2': os.path.join(OUTPUT_DIR, 'plot_xz_time.png'),
        'fig3': os.path.join(OUTPUT_DIR, 'plot_angles.png'),
        'fig4': os.path.join(OUTPUT_DIR, 'plot_error.png')
    }
    
    try:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        fig1.savefig(output_files['fig1'], dpi=150, bbox_inches='tight')
        fig2.savefig(output_files['fig2'], dpi=150, bbox_inches='tight')
        fig3.savefig(output_files['fig3'], dpi=150, bbox_inches='tight')
        fig4.savefig(output_files['fig4'], dpi=150, bbox_inches='tight')
        
        print("\n✓ Đã lưu các đồ thị:")
        for key, filepath in output_files.items():
            print(f"  • {filepath}")
    except Exception as e:
        print(f"✗ Lỗi khi lưu đồ thị: {e}")
    
    # Hiển thị đồ thị
    plt.show()

def print_summary(data_list):
    """In thống kê tóm tắt"""
    print("\n" + "="*80)
    print("THỐNG KÊ TÓM TẮT")
    print("="*80)
    print(f"{'File':<30} {'X_min':<10} {'X_max':<10} {'Z_min':<10} {'Z_max':<10} {'|θ|_max (°)':<12}")
    print("-"*80)
    
    for data in data_list:
        if data is not None:
            filename = data['filename']
            x_min = np.min(data['x_q'])
            x_max = np.max(data['x_q'])
            z_min = np.min(data['z_q'])
            z_max = np.max(data['z_q'])
            theta_max = np.rad2deg(np.max(np.abs(data['theta'])))
            
            print(f"{filename:<30} {x_min:<10.3f} {x_max:<10.3f} {z_min:<10.3f} {z_max:<10.3f} {theta_max:<12.3f}")
    
    print("="*80)


def compute_error_metrics(data_list, reference_idx=None):
    """Tính toán các chỉ số sai số so với tham chiếu
    
    Parameters:
    -----------
    data_list : list of dict
        Danh sách các dict chứa dữ liệu
    reference_idx : int, optional
        Index của dữ liệu tham chiếu
    
    Returns:
    --------
    metrics_rows : list of dict
        Danh sách các metrics cho từng biến
    """
    if reference_idx is None or reference_idx >= len(data_list):
        print("\n⚠️  Không có dữ liệu tham chiếu để tính sai số")
        return []
    
    ref_data = data_list[reference_idx]
    if ref_data is None:
        print("\n⚠️  Dữ liệu tham chiếu không hợp lệ")
        return []
    
    ref_label = extract_label_from_filename(ref_data['filename'])
    
    # Tolerance
    pos_tol = 0.05  # meters
    ang_tol = np.deg2rad(3.0)  # radians (~3 degrees)
    
    metrics_rows = []
    
    print("\n" + "="*80)
    print(f"TÍNH SAI SỐ SO VỚI THAM CHIẾU: {ref_label}")
    print("="*80)
    
    for i, data in enumerate(data_list):
        if data is None or i == reference_idx:
            continue
        
        name = extract_label_from_filename(data['filename'])
        
        # Align lengths
        n = min(len(data['t']), len(ref_data['t']))
        t = data['t'][:n]
        
        # Compute position errors
        ex = data['x_q'][:n] - ref_data['x_q'][:n]
        ez = data['z_q'][:n] - ref_data['z_q'][:n]
        
        # Compute angle errors (unwrap and align first)
        theta_sim_u, theta_ref_u = unwrap_and_align(data['theta'][:n], ref_data['theta'][:n])
        etheta = theta_sim_u - theta_ref_u
        
        beta_sim_u, beta_ref_u = unwrap_and_align(data['beta'][:n], ref_data['beta'][:n])
        ebeta = beta_sim_u - beta_ref_u
        
        # Compute metrics for each variable
        mx = compute_metrics(ex, sim=data['x_q'][:n], ref=ref_data['x_q'][:n], t=t, tol=pos_tol)
        mz = compute_metrics(ez, sim=data['z_q'][:n], ref=ref_data['z_q'][:n], t=t, tol=pos_tol)
        mtheta = compute_metrics(etheta, sim=theta_sim_u, ref=theta_ref_u, t=t, tol=ang_tol)
        mbeta = compute_metrics(ebeta, sim=beta_sim_u, ref=beta_ref_u, t=t, tol=ang_tol)
        
        # Position magnitude error
        epos = np.sqrt(ex**2 + ez**2)
        pos_rmse = float(np.sqrt(np.mean(epos**2)))
        pos_max = float(np.max(epos))
        
        # Print detailed metrics if enabled
        if PRINT_DETAILED_METRICS:
            print("\n" + "-"*60)
            print(f"So sánh: {name}")
            print(f"  X: MAE={mx['mae']:.4e} m, RMSE={mx['rmse']:.4e} m, MaxAbs={mx['max_abs']:.4e} m")
            print(f"  Z: MAE={mz['mae']:.4e} m, RMSE={mz['rmse']:.4e} m, MaxAbs={mz['max_abs']:.4e} m")
            print(f"  θ: MAE={mtheta['mae']:.4e} rad ({np.rad2deg(mtheta['mae']):.3f}°), RMSE={mtheta['rmse']:.4e} rad")
            print(f"  β: MAE={mbeta['mae']:.4e} rad ({np.rad2deg(mbeta['mae']):.3f}°), RMSE={mbeta['rmse']:.4e} rad")
            print(f"  Position magnitude: RMSE={pos_rmse:.4e} m, Max={pos_max:.4e} m")
        
        # Store metrics for CSV
        # Position magnitude summary
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
        
        # Individual variables
        for var_name, mm in [('x', mx), ('z', mz), ('theta', mtheta), ('beta', mbeta)]:
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
    
    # Print best/worst summary
    summary_pos = [r for r in metrics_rows if r.get('var') == 'pos_mag']
    summary_beta = [r for r in metrics_rows if r.get('var') == 'beta']
    summary_x = [r for r in metrics_rows if r.get('var') == 'x']
    summary_z = [r for r in metrics_rows if r.get('var') == 'z']
    summary_theta = [r for r in metrics_rows if r.get('var') == 'theta']
    
    print("\n" + "="*80)
    print("CHI TIẾT SAI SỐ (RMSE & Max) SO VỚI THAM CHIẾU")
    print("="*80)
    
    # Bảng tổng hợp cho tất cả các trường hợp
    if summary_pos:
        print("\n[1. QUỸ ĐẠO - Position Magnitude: |e_pos| = sqrt(e_x^2 + e_z^2)]")
        print(f"{'L_g':<20} {'RMSE (m)':<15} {'Max (m)':<15}")
        print("-"*50)
        for r in summary_pos:
            print(f"{r['name']:<20} {r['rmse']:<15.6f} {r['max_abs']:<15.6f}")
        
        best_pos = min(summary_pos, key=lambda r: r['rmse'])
        worst_pos = max(summary_pos, key=lambda r: r['rmse'])
        print("\n  → Tốt nhất:  " + f"{best_pos['name']:<20} RMSE={best_pos['rmse']:.6f} m   Max={best_pos['max_abs']:.6f} m")
        print("  → Tệ nhất:   " + f"{worst_pos['name']:<20} RMSE={worst_pos['rmse']:.6f} m   Max={worst_pos['max_abs']:.6f} m")
    
    if summary_x:
        print("\n[2. VỊ TRÍ X - e_x]")
        print(f"{'L_g':<20} {'RMSE (m)':<15} {'Max (m)':<15}")
        print("-"*50)
        for r in summary_x:
            print(f"{r['name']:<20} {r['rmse']:<15.6f} {r['max_abs']:<15.6f}")
    
    if summary_z:
        print("\n[3. VỊ TRÍ Z - e_z]")
        print(f"{'L_g':<20} {'RMSE (m)':<15} {'Max (m)':<15}")
        print("-"*50)
        for r in summary_z:
            print(f"{r['name']:<20} {r['rmse']:<15.6f} {r['max_abs']:<15.6f}")
    
    if summary_theta:
        print("\n[4. GÓC THETA - e_θ]")
        print(f"{'L_g':<20} {'RMSE (rad)':<15} {'Max (rad)':<15} {'RMSE (°)':<15} {'Max (°)':<15}")
        print("-"*80)
        for r in summary_theta:
            print(f"{r['name']:<20} {r['rmse']:<15.6f} {r['max_abs']:<15.6f} "
                  f"{np.rad2deg(r['rmse']):<15.3f} {np.rad2deg(r['max_abs']):<15.3f}")
    
    if summary_beta:
        print("\n[5. GÓC BETA - e_β]")
        print(f"{'L_g':<20} {'RMSE (rad)':<15} {'Max (rad)':<15} {'RMSE (°)':<15} {'Max (°)':<15}")
        print("-"*80)
        for r in summary_beta:
            print(f"{r['name']:<20} {r['rmse']:<15.6f} {r['max_abs']:<15.6f} "
                  f"{np.rad2deg(r['rmse']):<15.3f} {np.rad2deg(r['max_abs']):<15.3f}")
        
        best_beta = min(summary_beta, key=lambda r: r['rmse'])
        worst_beta = max(summary_beta, key=lambda r: r['rmse'])
        print("\n  → Tốt nhất:  " + f"{best_beta['name']:<20} RMSE={best_beta['rmse']:.6f} rad ({np.rad2deg(best_beta['rmse']):.3f}°)  "
              f"Max={best_beta['max_abs']:.6f} rad ({np.rad2deg(best_beta['max_abs']):.3f}°)")
        print("  → Tệ nhất:   " + f"{worst_beta['name']:<20} RMSE={worst_beta['rmse']:.6f} rad ({np.rad2deg(worst_beta['rmse']):.3f}°)  "
              f"Max={worst_beta['max_abs']:.6f} rad ({np.rad2deg(worst_beta['max_abs']):.3f}°)")
    
    print("="*80)
    
    return metrics_rows

def main():
    """Hàm chính"""
    print("="*80)
    print("VẼ 4 ĐỒ THỊ SO SÁNH KẾT QUẢ")
    print("="*80)
    
    # Tạo danh sách đường dẫn đầy đủ cho các file CSV
    csv_files = [os.path.join(RESULTS_DIR, f) for f in CSV_FILES]
    
    print(f"\n✓ Sẽ vẽ {len(csv_files)} file CSV:")
    for f in csv_files:
        if os.path.exists(f):
            is_ref = " (tham chiếu)" if os.path.basename(f) == REFERENCE_FILE else ""
            print(f"  ✓ {os.path.basename(f)}{is_ref}")
        else:
            print(f"  ✗ {os.path.basename(f)} (không tồn tại)")
    
    # Load dữ liệu từ các file
    print("\n[Đang load dữ liệu...]")
    data_list = []
    reference_idx = None
    
    for i, filepath in enumerate(csv_files):
        data = load_csv_file(filepath)
        if data is not None:
            data_list.append(data)
            print(f"  ✓ Đã load: {os.path.basename(filepath)}")
            
            # Kiểm tra xem có phải file tham chiếu không
            if REFERENCE_FILE and os.path.basename(filepath) == REFERENCE_FILE:
                reference_idx = len(data_list) - 1
        else:
            data_list.append(None)
    
    # Lọc các dữ liệu None
    data_list = [d for d in data_list if d is not None]
    
    if not data_list:
        print("✗ Không có dữ liệu hợp lệ để vẽ")
        return
    
    # In thống kê
    print_summary(data_list)
    
    # Tính sai số so với tham chiếu
    metrics_rows = compute_error_metrics(data_list, reference_idx)
    
    # Lưu metrics vào CSV
    if metrics_rows:
        metrics_file = os.path.join(OUTPUT_DIR, 'plot_results_metrics.csv')
        try:
            pd.DataFrame(metrics_rows).to_csv(metrics_file, index=False)
            print(f"\n✓ Đã lưu metrics vào: {metrics_file}")
        except Exception as e:
            print(f"\n✗ Lỗi khi lưu metrics: {e}")
    
    # Vẽ đồ thị
    print("\n[Đang vẽ đồ thị...]")
    plot_4_graphs(data_list, reference_idx=reference_idx)
    
    print("\n" + "="*80)
    print("✓ HOÀN THÀNH")
    print("="*80)

if __name__ == "__main__":
    main()
