import csv

# Đọc file gốc và tạo file mới với phi=0, beta=0
input_file = "controls_to_5_5.csv"
output_file = "controls_to_5_5_zero_angles.csv"

with open(input_file, 'r') as fin, open(output_file, 'w', newline='') as fout:
    reader = csv.DictReader(fin)
    writer = csv.writer(fout)
    
    # Ghi header
    writer.writerow(['time', 'u1', 'u2', 'u3', 'y', 'z', 'phi', 'beta'])
    
    # Đọc và ghi từng dòng với phi=0, beta=0
    count = 0
    for row in reader:
        writer.writerow([
            row['time'],
            row['u1'],
            row['u2'],
            row['u3'],
            row['y'],
            row['z'],
            0.0,  # phi = 0
            0.0   # beta = 0
        ])
        count += 1

print(f"✓ Đã tạo file {output_file} với {count} dòng")
print(f"  - Giữ nguyên: time, u1, u2, u3, y, z")
print(f"  - Đặt về 0: phi, beta")
