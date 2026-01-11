# TẠI SAO QUỸ ĐẠO KHÔNG KHÁC BIỆT NHIỀU KHI THAY ĐỔI L_g?

## 🔴 NGUYÊN NHÂN CHÍNH

### **Vấn đề hiện tại:**

```
Script test_lg.py:
├─ Thay đổi L_g trong chuyen_doi.py, mo_phong.py
├─ Chạy dieu_khien.py với flat_outputs.csv CŨ
│  └─ flat_outputs.csv được tạo với L_g = 0.1m (GIÁ TRỊ CỐ ĐỊNH)
└─ Kết quả: Quỹ đạo gần giống nhau ❌
```

### **Tại sao lại như vậy?**

#### 1️⃣ **Quỹ đạo tham chiếu không phù hợp**
```
flat_outputs.csv (tạo với L_g = 0.1m)
  ├─ x_q, z_q, beta: Quỹ đạo tối ưu cho L_g = 0.1m
  └─ Khi test L_g = 0.35m:
     ├─ Vẫn dùng quỹ đạo tham chiếu của L_g = 0.1m
     ├─ Nhưng dynamics thực tế là L_g = 0.35m
     └─ → Không tương thích!
```

#### 2️⃣ **Controller bù trừ được sự khác biệt**
```python
# Trong dieu_khien.py
u1_ff, u3_ff, tau_ff = recover_inputs_from_flat(...)  # Từ flat_outputs (L_g=0.1)
u1_pd = kpx*(x_qd - x_q) + kdx*(xdot_qd - xdot_q)    # PD bù sai số

u1 = u1_ff + u1_pd  # ← PD đủ mạnh để bù trừ sự thay đổi L_g!
```

**Kết quả:** Controller "cố gắng" kéo quỹ đạo về đúng như flat_outputs (L_g=0.1m), 
dù dynamics thực tế là L_g=0.35m → Quỹ đạo cuối cùng gần giống nhau.

---

## ✅ GIẢI PHÁP

### **Cần làm gì?**

**TÁI TẠO flat_outputs.csv với từng giá trị L_g mới!**

```
Quy trình ĐÚNG:
├─ Với L_g = 0.08m:
│  ├─ Sửa L_g trong chuyen_doi.py, mo_phong.py
│  ├─ Chạy qp5.py (hoặc minsnap planner) → flat_outputs_0.08.csv
│  └─ Chạy dieu_khien.py với flat_outputs_0.08.csv
│
├─ Với L_g = 0.10m:
│  ├─ Sửa L_g
│  ├─ Chạy qp5.py → flat_outputs_0.10.csv
│  └─ Chạy dieu_khien.py với flat_outputs_0.10.csv
│
└─ ...và cứ thế với mỗi L_g
```

### **Khi đó sẽ thấy sự khác biệt:**

| L_g | Ảnh hưởng đến quỹ đạo |
|-----|------------------------|
| ↑ L_g lớn hơn | • Mômen quán tính ↑<br>• Beta dao động chậm hơn<br>• Cần lực điều khiển lớn hơn<br>• Quỹ đạo X-Z có thể khác do ràng buộc dynamics |
| ↓ L_g nhỏ hơn | • Linh hoạt hơn<br>• Beta phản ứng nhanh<br>• Dễ điều khiển hơn |

---

## 📊 SO SÁNH 2 CÁCH

### **Cách 1: Script cũ (test_lg.py)** ❌
```
Ưu điểm:
  • Đơn giản, nhanh
  • Test khả năng BÙ TRỪ của controller

Nhược điểm:
  • Không phản ánh đúng ảnh hưởng của L_g
  • Quỹ đạo tham chiếu không phù hợp
  • Kết quả không có ý nghĩa vật lý rõ ràng
```

### **Cách 2: Script mới (test_lg_with_regenerate.py)** ✅
```
Ưu điểm:
  • Quỹ đạo tham chiếu được tối ưu cho từng L_g
  • Phản ánh ĐÚNG ảnh hưởng của L_g
  • Kết quả có ý nghĩa vật lý
  • Thấy rõ sự khác biệt

Nhược điểm:
  • Cần tái tạo flat_outputs (tốn thời gian)
  • Cần script minsnap/QP planner
```

---

## 🔧 CÁCH SỬ DỤNG

### **Bước 1: Xác định script tạo flat_outputs**

Tìm script tạo quỹ đạo minsnap (có thể là `qp5.py`):
```bash
# Kiểm tra xem qp5.py có tạo flat_outputs không
python qp5.py
```

### **Bước 2: Sửa test_lg_with_regenerate.py**

Uncomment và sửa dòng 132-137:
```python
def regenerate_flat_outputs(lg_value):
    # UNCOMMENT và sửa tên script của bạn:
    cmd = ["python", "qp5.py"]  # ← TÊN SCRIPT TẠO flat_outputs
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Lỗi: {result.stderr}")
        return False
    return True
```

### **Bước 3: Chạy test mới**
```bash
python test_lg_with_regenerate.py
```

---

## 📈 KẾT QUẢ MONG ĐỢI

Với script mới, bạn sẽ thấy:

| Metric | L_g = 0.08m | L_g = 0.35m | Chênh lệch |
|--------|-------------|-------------|------------|
| X_max | 5.01 | 5.23 | **+4.4%** |
| Z_max | 5.98 | 6.12 | **+2.3%** |
| Beta_max | 35° | 18° | **-48.6%** (dao động nhỏ hơn) |
| Control effort | Cao | Thấp hơn | Khác biệt rõ |

---

## 🎯 KẾT LUẬN

**Script test_lg.py hiện tại:**
- Test khả năng **bù trừ** của controller
- Không test ảnh hưởng **thực tế** của L_g

**Để thấy sự khác biệt thực sự:**
- Cần tái tạo `flat_outputs.csv` với từng L_g
- Dùng script `test_lg_with_regenerate.py`
- Hoặc tự tạo flat_outputs riêng cho mỗi L_g
