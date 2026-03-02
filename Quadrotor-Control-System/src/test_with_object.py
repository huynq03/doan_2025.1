# test_with_object.py
"""
Script để test mô phỏng với khối lượng vật 27g.
So sánh kết quả giữa không có vật và có vật.
"""

import sys
import os

# Thêm thư mục src vào path
sys.path.insert(0, os.path.dirname(__file__))

from dieu_khien_with_object import PDFFController

if __name__ == "__main__":
    print("="*70)
    print("TEST MÔ PHỎNG VỚI KHỐI LƯỢNG VẬT 27g")
    print("="*70)
    
    # Đường dẫn đến file quỹ đạo
    flat_csv = "C:\\Users\\2003h\\OneDrive\\Máy tính\\doan_2025.1\\Quadrotor-Control-System\\src\\minsnap_results\\flat_outputs.csv"
    
    # Tạo controller
    print("\n[1] Khởi tạo controller...")
    ctrl = PDFFController(flat_csv=flat_csv)
    
    # Chạy mô phỏng
    print("\n[2] Chạy mô phỏng...")
    save_path = "C:\\Users\\2003h\\OneDrive\\Máy tính\\doan_2025.1\\Quadrotor-Control-System\\src\\minsnap_results\\ketqua_with_object.csv"
    states, cmds = ctrl.mophong(save_csv=save_path, animate=False)
    
    print("\n" + "="*70)
    print("KẾT QUẢ:")
    print("="*70)
    print(f"- Số bước mô phỏng: {len(states)}")
    print(f"- Đã gắp vật: {'CÓ' if ctrl.gripped else 'KHÔNG'}")
    print(f"- Khối lượng vật: {ctrl.object_mass*1000:.1f}g")
    print(f"- Kết quả đã lưu tại: {save_path}")
    print("\nHướng dẫn:")
    print("1. So sánh file ketqua_with_object.csv với ketqua.csv (không có vật)")
    print("2. Để xem animation, chạy lại với: --simulate --animate")
    print("="*70)
