# Bộ Điều Khiển Quadrotor (PD + FF)

Thư mục này chứa các file cần thiết để chạy PD + Feed-forward controller cho quadrotor + gripper.

## Cấu trúc

```
src/
├── dieu_khien.py        # Bộ điều khiển chính (PD + FF)
├── chuyen_doi.py        # Ánh xạ đầu ra phẳng → lệnh điều khiển
├── mo_phong.py          # Động lực học và trực quan hóa
├── flat_outputs.csv     # Dữ liệu quỹ đạo đầu vào
├── minsnap_results/     # Thư mục output simulation
└── archive/             # Các file cũ không dùng
```

## Chạy Controller

### Mô phỏng cơ bản:
```bash
python dieu_khien.py --flat_csv flat_outputs.csv --simulate --save_csv minsnap_results
```

### Với animation:
```bash
python dieu_khien.py --flat_csv flat_outputs.csv --simulate --save_csv minsnap_results --animate
```

### Tùy chọn beta_sign:
```bash
# Nếu plant dùng CCW dương (mặc định)
python dieu_khien.py --beta_sign 1 --simulate

# Nếu plant dùng CW dương
python dieu_khien.py --beta_sign -1 --simulate
```

## Dependencies

**File phụ thuộc:**
- `chuyen_doi.py`: Cung cấp `recover_inputs_from_flat()` và `PARAMS`
- `mo_phong.py`: Cung cấp `jax_dynamics_matrix()` cho dynamics và `animate()` cho visualization
- `flat_outputs.csv`: Quỹ đạo tham chiếu (t, x_q, z_q, beta)

**Thư viện Python:**
- numpy
- pandas
- jax
- matplotlib

## Output

Kết quả simulation được lưu vào `minsnap_results/ketqua.csv` với các cột:
- t, u1, u2, u3, tau
- x_q, xdot_q, z_q, zdot_q
- theta, theta_dot, beta, beta_dot
