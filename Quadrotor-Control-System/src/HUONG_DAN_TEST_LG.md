# HƯỚNG DẪN SO SÁNH ẢNH HƯỞNG CỦA L_g

## Quy trình kiểm tra sự khác biệt quỹ đạo khi thay đổi L_g

### Cách 1: Tự động (Khuyến nghị) ✅

Chạy script tự động test nhiều giá trị L_g:

```bash
python test_lg_comparison.py
```

Script này sẽ:
1. ✓ Backup các file gốc (chuyen_doi.py, mo_phong.py)
2. ✓ Tự động thay đổi L_g với các giá trị: [0.08, 0.10, 0.12, 0.15, 0.20] m
3. ✓ Chạy simulation cho mỗi giá trị
4. ✓ Lưu kết quả vào: ketqua_lg0.08.csv, ketqua_lg0.10.csv, ...
5. ✓ Vẽ biểu đồ so sánh 6 khía cạnh:
   - Quỹ đạo XZ
   - X theo thời gian
   - Z theo thời gian
   - Góc nghiêng theta
   - Góc cánh tay beta
   - Sai số so với L_g chuẩn
6. ✓ Khôi phục lại files gốc
7. ✓ In thống kê so sánh

**Kết quả**: Biểu đồ so sánh lưu tại `minsnap_results/lg_comparison.png`

---

### Cách 2: Thủ công (Chi tiết hơn)

#### Bước 1: Thay đổi L_g và l_p trong các file

**File 1: chuyen_doi.py** (dòng 11)
```python
PARAMS = dict(
    m_q = 0.500,
    m_g = 0.158,
    J_q = 0.15,
    J_g = 0.001,
    L_g = 0.15,       # ← THAY ĐỔI GIÁ TRỊ NÀY (ví dụ: 0.15)
    g   = 9.81,
)
```

**File 2: mo_phong.py** (dòng 11-12)
```python
l_p, l_q = 0.15, 0.2  # ← THAY ĐỔI l_p (giá trị đầu) = L_g
J_q, J_g, L_g = 0.15, 0.001, 0.15  # ← THAY ĐỔI L_g (giá trị cuối)
```

**Lưu ý:** l_p và L_g phải có cùng giá trị!

#### Bước 2: Chạy simulation
```bash
python dieu_khien.py --simulate --save_csv minsnap_results\ketqua_lg0.15.csv
```

#### Bước 3: Vẽ quỹ đạo
```bash
python vequydao.py
```

#### Bước 4: Lặp lại với các giá trị L_g khác

Ví dụ: Test L_g = 0.08, 0.10, 0.12, 0.15, 0.20

#### Bước 5: So sánh thủ công
- Mở các file CSV và so sánh số liệu
- Hoặc dùng script riêng để vẽ nhiều quỹ đạo trên cùng biểu đồ

---

## Ý nghĩa của L_g và l_p

**L_g**: Chiều dài cánh tay gripper trong dynamics (m) - khoảng cách từ tâm quadrotor đến gripper, được sử dụng trong các phương trình động lực học

**l_p**: Chiều dài cánh tay trong visualization (m) - chiều dài hiển thị của pendulum/gripper trong animation

**Lưu ý quan trọng**: L_g và l_p phải có cùng giá trị để đảm bảo tính nhất quán giữa mô phỏng và hiển thị!

**Ảnh hưởng khi thay đổi L_g:**
- ↑ L_g lớn hơn → Cánh tay dài hơn → Mômen quán tính lớn hơn → Khó điều khiển hơn
- ↓ L_g nhỏ hơn → Cánh tay ngắn hơn → Linh hoạt hơn → Dễ điều khiển

**Phạm vi hợp lý:** 0.05m - 0.25m (cho quadrotor micro UAV)

---

## Các metrics để so sánh

1. **Sai số quỹ đạo**: Độ lệch giữa quỹ đạo mong muốn và thực tế
2. **Góc nghiêng theta**: Góc nghiêng của quadrotor
3. **Góc cánh tay beta**: Góc dao động của gripper
4. **Năng lượng điều khiển**: Tổng lực đẩy và mômen xoắn
5. **Thời gian ổn định**: Thời gian để hệ thống ổn định

---

## Ghi chú

- Files backup được lưu trong thư mục `backup_lg_test/`
- Kết quả simulation lưu trong `minsnap_results/`
- Có thể chỉnh sửa danh sách giá trị L_g trong file `test_lg_comparison.py` (dòng 14)
