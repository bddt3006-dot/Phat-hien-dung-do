import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rule_engine import ParkingRuleEngine

def test_passing_object_does_not_hijack():
    config = {
        'rules': {
            'time_threshold': 5.0,
            'movement_tolerance': 30.0,
            'roi': [[0, 0], [500, 0], [500, 500], [0, 500]]
        }
    }
    engine = ParkingRuleEngine(config)

    # 1. Xe ID 1 đỗ tại (100, 100) lúc t=0.0s
    engine.update([(1, 100, 100)], current_time=0.0)
    engine.update([(1, 100, 100)], current_time=2.0)
    assert engine.history[1]['start_time'] == 0.0

    # 2. Xe đạp (ID 2) đi ngang qua sát cạnh xe ID 1 ở t=3.0s (tọa độ 110, 100)
    # Cả hai xe CÙNG XUẤT HIỆN trong frame
    engine.update([(1, 100, 100), (2, 110, 100)], current_time=3.0)
    assert 1 in engine.history, "Xe 1 không được bị xóa"
    assert engine.history[1]['start_time'] == 0.0, "Thời gian của Xe 1 không được bị reset"
    assert 2 in engine.history, "Xe 2 có bộ đếm riêng của nó"

    # 3. Xe đạp ID 2 chạy tiếp đi xa ở t=4.0s (180, 100), Xe 1 vẫn ở (100, 100)
    engine.update([(1, 100, 100), (2, 180, 100)], current_time=4.0)
    assert engine.history[1]['start_time'] == 0.0, "Thời gian của Xe 1 vẫn giữ nguyên 0.0s"

    # 4. Xe đạp ID 2 rời khỏi ROI ở t=5.5s, chỉ còn Xe 1 ở (100, 100)
    # Xe 1 lúc này đã đỗ được 5.5s (lớn hơn ngưỡng 5.0s) -> Phải vi phạm!
    viols = engine.update([(1, 100, 100)], current_time=5.5)
    assert len(viols) == 1, "Xe 1 phải bị bắt vi phạm!"
    assert viols[0]['track_id'] == 1
    assert viols[0]['duration'] == 5.5
    print("Test Passed: Xe dap di ngang qua KHONG cuop hoac reset thoi gian cua xe dang do!")

if __name__ == '__main__':
    test_passing_object_does_not_hijack()
