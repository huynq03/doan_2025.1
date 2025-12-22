# Hệ Thống Điều Khiển Nắm Bắt Lấy Cảm Hứng Từ Chim Cho Quadrotor

Dự án này hiện thực hóa lập kế hoạch quỹ đạo và điều khiển cho quadrotor phẳng (chuyển động trong mặt phẳng x–z) gắn cơ cấu kẹp (gripper) lấy cảm hứng từ cơ chế nắm bắt của chim, dựa trên bài báo ["Avian-Inspired Grasping for Quadrotor Micro UAVs"](https://doi.org/10.1115/DETC2013-12842) (Thomas và cộng sự, IDETC'13).

## Tổng Quan Dự Án

Hệ thống cho phép quadrotor trang bị gripper thực hiện thao tác nắm bắt tự động, theo nguyên lý phẳng vi phân (differential flatness) cho việc tạo quỹ đạo và điều khiển PD + feed-forward cho việc thực thi.

### Động Học/Hệ Phương Trình Hệ Thống

Hệ quadrotor–gripper phẳng hoạt động trong mặt phẳng x–z với vector trạng thái:
- **Trạng thái quadrotor:** [x_q, ẋ_q, z_q, ż_q, θ, θ̇] – vị trí, vận tốc, góc thân và vận tốc góc
- **Trạng thái gripper:** [β, β̇] – góc gripper và vận tốc góc

Hệ có tổng cộng 8 trạng thái và 3 tín hiệu điều khiển: u₁ (lực đẩy/thrust), u₃ (mô-men pitch), τ (mô-men gripper).

### Nền Tảng Toán Học

Động lực học hệ dựa trên cơ học Lagrange với các phương trình chính sau (trích từ IDETC'13):

$$\begin{align}
m_s \ddot{x}_s &= u_1 \sin \theta \\
m_s \ddot{z}_s &= u_1 \cos \theta - m_s g
\end{align}$$

Trong đó:
- m_q = 0,500 kg (khối lượng quadrotor)
- m_g = 0,158 kg (khối lượng gripper)
- m_s = m_q + m_g (khối lượng toàn hệ)
- J_q = 1,2×10⁻² kg·m² (mô-men quán tính quadrotor)
- J_g = 1,0×10⁻³ kg·m² (mô-men quán tính gripper)
- L_g = 0,105 m (chiều dài tay gripper)
- g = 9,81 m/s² (gia tốc trọng trường)

**Quan hệ động học của gripper:**
```
x_g = x_q + L_g cos(β)
z_g = z_q - L_g sin(β)
```

## Các Thành Phần Hiện Thực

1. **Lập Kế Hoạch Quỹ Đạo (dựa trên QP)**
   - Tối ưu đa thức 2 đoạn (bậc 7)
   - Ràng buộc waypoint cho các thao tác nhặt/đặt
   - Ràng buộc mềm cho line-of-sight và định hình góc gripper

2. **Ánh Xạ Phẳng Vi Phân (Differential Flatness Mapping)**
   - Ánh xạ các đầu ra phẳng (x_q, z_q, β) → tín hiệu điều khiển (u₁, u₃, τ)
   - Tính toán các hạng số feed-forward: u₁ᵈ, u₃ᵈ, τᵈ, θᵈ, θ̇ᵈ
   - Dựa trên các phương trình (16)–(31) trong bài IDETC'13

3. **Bộ Điều Khiển PD + Feed-Forward**
   - Kiến trúc điều khiển phân tầng (cascade)
   - Điều khiển vị trí → lệnh góc thân → điều khiển mô-men
   - Điều khiển góc gripper với mô-men đầu ra
   - Xử lý quy ước dấu cho các dạng triển khai hệ vật lý khác nhau

## Chi Tiết Kỹ Thuật

### Các Thành Phần Chính

#### Hệ Điều Khiển Lõi (`src/`)
- **`dieu_khien.py`** – Bộ điều khiển PD + feed-forward chính
  - Điều khiển phân tầng vị trí/góc thân/gripper
  - Hỗ trợ chuyển đổi quy ước dấu của `beta`
  - Tích hợp mô phỏng `mo_phong`
  
  - **`chuyen_doi.py`** – Bộ ánh xạ phẳng vi phân
  - Hiện thực các phương trình (16)–(31) từ bài báo
  - Ánh xạ (x_q, z_q, β) → (u₁, u₃, τ, θ, θ̇)
  - Vi phân bằng sai phân hữu hạn
  
- **`mo_phong.py`** – Mô phỏng động lực học và trực quan hóa
  - Tích hợp động học bằng JAX
  - Hỗ trợ hoạt họa trực quan quỹ đạo

#### Lập Kế Hoạch Quỹ Đạo (`src/archive/`)
- **`qp5.py`** – Trình tối ưu quỹ đạo minimum-snap (phiên bản hiện tại)
  - Ràng buộc waypoint (bắt đầu → nhặt → kết thúc)
  - Giới hạn 2 phía cho góc gripper (beta_min ≤ β ≤ beta_max)
  - Ràng buộc mềm cho line-of-sight và định hình gripper
  - Xuất: `flat_outputs.csv` (mặc định, cấu hình qua `--out_csv`)
- **`qp4.py`** – Phiên bản thay thế với tính năng tương tự
  - Xuất: `flat_outputs1.csv`

### Kiến Trúc Điều Khiển

Đầu ra phẳng (x_q, z_q, β)  →  [chuyen_doi.py]  →  Feed-forward (u₁ᵈ, u₃ᵈ, τᵈ, θᵈ)
                                                           ↓
Trạng thái hệ (đo lường)     →  [dieu_khien.py] →  Hiệu chỉnh PD  →  Lệnh cuối cùng
                                                           ↓
                                                   [mo_phong.py] Mô phỏng

**Hệ số khuếch đại điều khiển (chỉnh trong `dieu_khien.py`):**
```python
kpx = 1.2        # K khuếch đại P vị trí ngang
kdx = 0.6        # K khuếch đại D vị trí ngang
kpz = 10.0       # K khuếch đại P độ cao
kdz = 5.5        # K khuếch đại D độ cao
kp_theta = 6.0   # K khuếch đại P góc thân
kd_theta = 2.5   # K khuếch đại D góc thân
kp_beta = 4.0    # K khuếch đại P góc gripper
kd_beta = 1.2    # K khuếch đại D góc gripper
```

### Công Nghệ Sử Dụng
- **NumPy** – Tính toán số
- **Pandas** – Xử lý dữ liệu CSV
- **JAX** – Vi phân tự động và động lực học nhanh
- **Matplotlib** – Trực quan hóa và hoạt họa quỹ đạo

## Bắt Đầu

### Yêu Cầu Cài Đặt

#### Cách 1: Sử dụng Conda (Khuyến nghị)

**Bước 1: Tạo môi trường conda mới**
```bash
conda create -n quad2d python=3.9
```

**Bước 2: Kích hoạt môi trường**
```bash
conda activate quad2d
```

**Bước 3: Cài đặt các thư viện cần thiết**
```bash
# Di chuyển vào thư mục dự án
cd Quadrotor-Control-System

# Cài đặt từ file requirements.txt
pip install -r requirements.txt
```

**Bước 4: Chạy code**
```bash
cd src
python dieu_khien.py --simulate --save_csv minsnap_results
```

**Lưu ý:** Mỗi lần mở terminal mới, cần kích hoạt môi trường:
```bash
conda activate quad2d
```

Để thoát môi trường:
```bash
conda deactivate
```

#### Cách 2: Cài đặt trực tiếp (không dùng môi trường ảo)
```bash
pip install numpy pandas jax jaxlib matplotlib
```

### Phụ Thuộc
- **Tệp chính:** `dieu_khien.py`, `chuyen_doi.py`, `mo_phong.py`, `flat_outputs.csv`
- **Thư viện:** NumPy, Pandas, JAX/JAXlib, Matplotlib
- **Đầu vào phẳng:** `flat_outputs.csv` chứa các cột: `t, x_q, z_q, beta`

### Cấu Trúc Thư Mục
```
Quadrotor-Control-System/
├── src/
│   ├── dieu_khien.py        # Bộ điều khiển chính
│   ├── chuyen_doi.py        # Ánh xạ phẳng vi phân
│   ├── mo_phong.py          # Mô phỏng
│   ├── flat_outputs.csv     # Dữ liệu quỹ đạo
│   ├── minsnap_results/     # Thư mục kết quả
│   └── archive/             # Trình QP & các file cũ
├── slide/                   # Tài liệu dự án
└── README.md
```

### Chạy Bộ Điều Khiển

#### 1. Sinh Quỹ Đạo (Tùy chọn – đã có sẵn)
```bash
cd src/archive

# Dùng qp5.py (khuyến nghị – sinh flat_outputs.csv)
python qp5.py

# Hoặc dùng qp4.py (sinh flat_outputs1.csv)
python qp4.py
```

**Lưu ý:** `dieu_khien.py` dùng `flat_outputs.csv` theo mặc định (được sinh bởi `qp5.py`). Để dùng đầu ra từ `qp4.py`, chỉ định `--flat_csv ../minsnap_results/flat_outputs1.csv`.

#### 2. Chạy Mô Phỏng Bộ Điều Khiển
```bash
cd src

# Mô phỏng cơ bản
python dieu_khien.py --flat_csv flat_outputs.csv --simulate --save_csv minsnap_results

# Kèm hoạt họa
python dieu_khien.py --simulate --save_csv minsnap_results --animate

# Thay đổi quy ước dấu cho beta (nếu cần)
python dieu_khien.py --beta_sign -1 --simulate --save_csv minsnap_results
```

#### 3. Tùy Chọn Dòng Lệnh
```
--flat_csv FLAT_CSV       CSV quỹ đạo đầu vào (mặc định: flat_outputs.csv)
--beta_sign {1,-1}        Quy ước dấu góc beta: +1=CCW, -1=CW (mặc định: 1)
--simulate                Bật mô phỏng với mo_phong.py
--save_csv PATH           Lưu kết quả ra CSV (thư mục hoặc đường dẫn file)
--animate                 Hiển thị hoạt họa trong khi mô phỏng
```

### Kết Quả Đầu Ra
- Mặc định lưu tại `src/minsnap_results/ketqua.csv` với các cột:
  - `t, u1, u2, u3, tau`
  - `x_q, xdot_q, z_q, zdot_q`
  - `theta, theta_dot, beta, beta_dot`

### File Đầu Vào/Đầu Ra

**Đầu vào:** `flat_outputs.csv`
```
t, x_q, z_q, beta, xd_q, zd_q, betad
```

**Đầu ra:** `minsnap_results/ketqua.csv`
```
t, u1, u2, u3, tau,
x_q, xdot_q, z_q, zdot_q,
theta, theta_dot, beta, beta_dot
```

## Chi Tiết Hiện Thực

### Lập Kế Hoạch Quỹ Đạo (QP)
- **Hàm mục tiêu:** giảm thiểu đạo hàm bậc 4 (snap) của vị trí
- **Ràng buộc:** 
  - Bằng nhau: vị trí tại waypoint, vận tốc/ gia tốc bằng 0 tại biên
  - Mềm: line-of-sight trước khi nhặt, định hình góc gripper sau khi nhặt
- **Phương pháp:** giải hệ KKT cho bài toán QP có ràng buộc

### Luật Điều Khiển
**Vị trí → Góc thân:**
```
θ_cmd = arcsin(Kp_x·e_x + Kd_x·ė_x) + θᵈ
```

**Độ cao:**
```
u₁ = Kp_z·e_z + Kd_z·ė_z + u₁ᵈ
```

**Góc thân:**
```
u₃ = Kp_θ·e_θ + Kd_θ·ė_θ + u₃ᵈ
```

**Gripper:**
```
τ = Kp_β·e_β + Kd_β·ė_β + τᵈ
```

### Quy Ước Dấu Cho Beta
Bộ điều khiển hỗ trợ các quy ước dấu khác nhau của hệ vật lý:
- **+1 (theo bài báo):** chiều dương quay ngược chiều kim đồng hồ (CCW)
- **-1 (thay thế):** chiều dương quay cùng chiều kim đồng hồ (CW)

Việc ánh xạ tự động đảm bảo hiệu chỉnh PD chính xác bất kể quy ước dấu của hệ.

## Kết Quả

Hệ thống chứng minh thành công:
- ✅ Quỹ đạo minimum-snap mượt mà
- ✅ Bám vị trí và góc thân ổn định
- ✅ Phối hợp chuyển động gripper trong khi bay
- ✅ Bù feed-forward chính xác

## Tài Liệu Tham Khảo

1. Thomas, J., Loianno, G., Polin, J., Sreenath, K., & Kumar, V. (2013). "Avian-Inspired Grasping for Quadrotor Micro UAVs". *ASME 2013 International Design Engineering Technical Conferences*. DOI: [10.1115/DETC2013-12842](https://doi.org/10.1115/DETC2013-12842)

2. Mellinger, D., & Kumar, V. (2011). "Minimum snap trajectory generation and control for quadrotors". *2011 IEEE International Conference on Robotics and Automation*. DOI: [10.1109/ICRA.2011.5980409](https://doi.org/10.1109/ICRA.2011.5980409)
