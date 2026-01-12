# -*- coding: utf-8 -*-
import subprocess
import os
import shutil
import re

# Cấu hình gains để test - Để thấy sai lệch rõ nhất
GAIN_CONFIGS = [
    {"name": "Low", "multiplier": 0.5},   # Rất thấp - sai lệch lớn nhất
    {"name": "Medium", "multiplier": 1.0},    # Baseline
    {"name": "High", "multiplier": 5.0}     # Rất cao - tracking tốt nhất
]

# Gains gốc (từ dieu_khien.py - hiện tại)
BASE_GAINS = {
    'kpx': 1.2,
    'kdx': 0.6,
    'kpz': 10.0,
    'kdz': 5.5,
    'kp_theta': 6.0,
    'kd_theta': 2.5,
    'kp_beta': 2.0,
    'kd_beta': 0.6
}

FILE_TO_MODIFY = "dieu_khien.py"
BACKUP_DIR = "backup_gains_lg_test"

def backup_files():
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    if os.path.exists(FILE_TO_MODIFY):
        shutil.copy2(FILE_TO_MODIFY, os.path.join(BACKUP_DIR, FILE_TO_MODIFY))

def restore_files():
    backup_path = os.path.join(BACKUP_DIR, FILE_TO_MODIFY)
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, FILE_TO_MODIFY)

def modify_gains(multiplier):
    """Điều chỉnh gains trong dieu_khien.py theo multiplier"""
    with open(FILE_TO_MODIFY, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Tính gains mới
    gains = {k: v * multiplier for k, v in BASE_GAINS.items()}
    
    # Pattern để tìm class Gains (class variables: "kpx: float = ")
    patterns = {
        'kpx': (r'(kpx:\s*float\s*=\s*)[\d.]+', f'\\g<1>{gains["kpx"]:.1f}'),
        'kdx': (r'(kdx:\s*float\s*=\s*)[\d.]+', f'\\g<1>{gains["kdx"]:.1f}'),
        'kpz': (r'(kpz:\s*float\s*=\s*)[\d.]+', f'\\g<1>{gains["kpz"]:.1f}'),
        'kdz': (r'(kdz:\s*float\s*=\s*)[\d.]+', f'\\g<1>{gains["kdz"]:.1f}'),
        'kp_theta': (r'(kp_theta:\s*float\s*=\s*)[\d.]+', f'\\g<1>{gains["kp_theta"]:.1f}'),
        'kd_theta': (r'(kd_theta:\s*float\s*=\s*)[\d.]+', f'\\g<1>{gains["kd_theta"]:.1f}'),
        'kp_beta': (r'(kp_beta:\s*float\s*=\s*)[\d.]+', f'\\g<1>{gains["kp_beta"]:.1f}'),
        'kd_beta': (r'(kd_beta:\s*float\s*=\s*)[\d.]+', f'\\g<1>{gains["kd_beta"]:.1f}')
    }
    
    for gain_name, (pattern, replacement) in patterns.items():
        content = re.sub(pattern, replacement, content)
    
    with open(FILE_TO_MODIFY, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return gains

def run_simulation(config_name, multiplier):
    output_csv = f"minsnap_results/ketqua_gains_{config_name}.csv"
    
    print(f"\n[{config_name} (x{multiplier:.2f})]")
    
    # Điều chỉnh gains
    print(f"  → Điều chỉnh gains...", end=" ")
    gains = modify_gains(multiplier)
    print("✓")
    
    # Chạy simulation
    print(f"  → Chạy simulation...", end=" ")
    result = subprocess.run(["python", "dieu_khien.py", "--simulate", "--save_csv", output_csv],
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓")
        return output_csv, gains
    else:
        print("✗")
        # In stdout và stderr để debug
        if result.stdout:
            print(f"  STDOUT: {result.stdout[-150:]}")
        if result.stderr:
            print(f"  STDERR: {result.stderr[-150:]}")
        return None, None

def print_statistics(all_gains, config_names):
    print("\n" + "="*80)
    print(f"{'Config':<10} {'kpx':<8} {'kdx':<8} {'kpz':<8} {'kdz':<8} "
          f"{'kp_θ':<8} {'kd_θ':<8} {'kp_β':<8} {'kd_β':<8}")
    print("-"*80)
    
    for name, gains in zip(config_names, all_gains):
        if gains:
            print(f"{name:<10} {gains['kpx']:<8.1f} {gains['kdx']:<8.1f} "
                  f"{gains['kpz']:<8.1f} {gains['kdz']:<8.1f} "
                  f"{gains['kp_theta']:<8.1f} {gains['kd_theta']:<8.1f} "
                  f"{gains['kp_beta']:<8.1f} {gains['kd_beta']:<8.1f}")

def main():
    print("="*80)
    print("TEST GAINS (KP, KD) - LƯU KẾT QUẢ CSV")
    print("="*80)
    
    backup_files()
    
    config_names = []
    all_gains = []
    result_files = []
    
    for config in GAIN_CONFIGS:
        result_file, gains = run_simulation(config['name'], config['multiplier'])
        config_names.append(config['name'])
        all_gains.append(gains)
        result_files.append(result_file)
    
    restore_files()
    
    print_statistics(all_gains, config_names)
    
    print("\n" + "="*80)
    print("KẾT QUẢ ĐÃ LƯU:")
    for name, file in zip(config_names, result_files):
        if file:
            print(f"  {name:<10} → {file}")
    print("\n✓ HOÀN THÀNH - Dùng vequydao.py để vẽ đồ thị")
    print("="*80)

if __name__ == "__main__":
    main()
