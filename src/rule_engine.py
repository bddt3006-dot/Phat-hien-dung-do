import cv2
import numpy as np
from collections import deque

class ParkingRuleEngine:
    """
    Engine phân tích hành vi dừng đỗ phương tiện thông minh & chống báo sai xe đang chạy:
    - Tách biệt hoàn toàn trạng thái ĐANG CHẠY (Moving) và ĐANG DỪNG ĐỖ (Stopped/Stationary).
    - CƠ CHẾ XÁC NHẬN DỪNG LIÊN TỤC (Debounce 1.2s): Xe bắt buộc phải đứng yên liên tục trong ít nhất 
      1.2 giây mới bắt đầu kích hoạt trạng thái dừng đỗ. Xe đang chạy qua dù có bị chậm lại hoặc 
      detector rung 1-2 frames cũng TUYỆT ĐỐI KHÔNG bị gắn nhãn dừng đỗ hay đếm giây!
    - KHÓA ĐIỂM NEO VỊ TRÍ (Anchor Locking): Điểm neo đỗ được cố định tại vị trí dừng thực tế, không bị trôi 
      theo xe, giúp nhận diện chính xác khoảnh khắc xe lăn bánh rời đi để lập tức hủy timer.
    """
    def __init__(self, config):
        self.time_threshold = float(config['rules'].get('time_threshold', 5.0))
        self.movement_tolerance = float(config['rules'].get('movement_tolerance', 25.0))
        self.roi_polygon = np.array(config['rules'].get('roi', []), np.int32)
        
        # Ngưỡng vận tốc xác định xe đứng yên (pixel/giây)
        # Xe đứng yên dao động nhẹ 1-3 px do detector jitter (~3-6 px/s ở 25-30 FPS)
        self.stop_speed_threshold = 10.0
        
        # Ngưỡng dịch chuyển tịnh tiến tối đa trong cửa sổ quan sát (pixel)
        self.stop_dist_threshold = 12.0
        
        # THỜI GIAN XÁC NHẬN DỪNG LIÊN TỤC (giây): Xe phải đứng yên liên tục đủ 1.2s mới tính là đỗ
        self.min_stationary_seconds = 1.2
        
        # Cửa sổ thời gian tính vận tốc tức thời (giây)
        self.speed_window_seconds = 0.8
        
        # Thời gian ân hạn khi xe bị khuất (mất track do xe khác đi ngang qua)
        self.max_missing_seconds = 5.0
        
        # Thời gian ân hạn khi xe tạm thời ra ngoài ROI
        self.roi_grace_seconds = 2.0
        
        # Lưu trữ trạng thái xe: {track_id: dict}
        self.history = {}

    @staticmethod
    def compute_iou(box1, box2):
        """Tính Intersection over Union (IoU) giữa 2 bounding boxes [x1, y1, x2, y2]"""
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        inter_w = max(0.0, xB - xA)
        inter_h = max(0.0, yB - yA)
        inter_area = inter_w * inter_h
        
        area1 = max(1e-5, (box1[2] - box1[0]) * (box1[3] - box1[1]))
        area2 = max(1e-5, (box2[2] - box2[0]) * (box2[3] - box2[1]))
        iou = inter_area / float(area1 + area2 - inter_area)
        return iou

    def bbox_touches_roi(self, x1, y1, x2, y2):
        """Kiểm tra bbox có chạm hoặc nằm trong ROI không"""
        if len(self.roi_polygon) < 3:
            return True
        
        check_points = [
            (x1, y1), (x2, y1), (x1, y2), (x2, y2),
            ((x1+x2)/2, y1), ((x1+x2)/2, y2),
            (x1, (y1+y2)/2), (x2, (y1+y2)/2),
            ((x1+x2)/2, (y1+y2)/2)
        ]
        for pt in check_points:
            if cv2.pointPolygonTest(self.roi_polygon, (float(pt[0]), float(pt[1])), False) >= 0:
                return True
        
        for pt in self.roi_polygon:
            px, py = float(pt[0]), float(pt[1])
            if x1 <= px <= x2 and y1 <= py <= y2:
                return True
        
        return False

    def is_vehicle_in_roi(self, x1, y1, x2, y2, cx, cy):
        """Kiểm tra vị trí thực tế của phương tiện đối với vùng ROI"""
        if len(self.roi_polygon) < 3:
            return True
        # Điểm tiếp xúc mặt đường (đáy giữa bbox)
        if cv2.pointPolygonTest(self.roi_polygon, (float(cx), float(y2)), False) >= 0:
            return True
        # Tâm xe
        if cv2.pointPolygonTest(self.roi_polygon, (float(cx), float(cy)), False) >= 0:
            return True
        # Phần diện tích bbox tiếp xúc ROI
        return self.bbox_touches_roi(x1, y1, x2, y2)

    def update(self, tracks, current_time):
        """
        tracks: list of (track_id, cx, cy, x1, y1, x2, y2)
        Trả về: list of violations [{'track_id': id, 'duration': dur, 'is_new': bool}]
        """
        violations = []
        all_current_ids = set()
        parsed_tracks = []

        for item in tracks:
            if len(item) == 7:
                tid, cx, cy, bx1, by1, bx2, by2 = item
            elif len(item) == 3:
                tid, cx, cy = item
                bx1, by1, bx2, by2 = cx - 20, cy - 20, cx + 20, cy + 20
            else:
                continue
            all_current_ids.add(tid)
            parsed_tracks.append((tid, float(cx), float(cy), float(bx1), float(by1), float(bx2), float(by2)))

        for track_id, cx, cy, bx1, by1, bx2, by2 in parsed_tracks:
            cur_pos = (cx, cy)
            cur_box = (bx1, by1, bx2, by2)
            in_roi = self.is_vehicle_in_roi(bx1, by1, bx2, by2, cx, cy)

            # Trường hợp track_id chưa có trong history
            if track_id not in self.history:
                # Kiểm tra Spatial Re-ID: chỉ kế thừa xe đỗ cũ nếu xe đỗ cũ thực sự đứng yên tại vị trí đó
                best_match_id = None
                best_match_iou = 0.0

                for old_id, old_rec in self.history.items():
                    if old_id in all_current_ids:
                        continue
                    if not old_rec.get('is_stopped'):
                        continue

                    missing_dur = current_time - old_rec['last_seen_time']
                    if missing_dur > self.max_missing_seconds:
                        continue

                    iou = self.compute_iou(cur_box, old_rec['anchor_box'])
                    dist = np.hypot(cur_pos[0] - old_rec['anchor_pos'][0],
                                    cur_pos[1] - old_rec['anchor_pos'][1])

                    # Chỉ kế thừa nếu vị trí trùng khớp cao (IoU >= 0.4 hoặc lệch < 15px)
                    if (iou >= 0.40 or dist <= 15.0) and iou > best_match_iou:
                        best_match_iou = iou
                        best_match_id = old_id

                if best_match_id is not None:
                    # Kế thừa đúng kỷ lục của xe đỗ cũ
                    inherited_rec = self.history.pop(best_match_id)
                    inherited_rec['last_seen_time'] = current_time
                    inherited_rec['outside_roi_start'] = None
                    inherited_rec['anchor_box'] = cur_box
                    inherited_rec['pos_history'].append((current_time, cx, cy))
                    self.history[track_id] = inherited_rec
                else:
                    # Xe mới xuất hiện: mặc định coi là đang chuyển động (is_stopped = False)
                    self.history[track_id] = {
                        'pos_history': deque([(current_time, cx, cy)], maxlen=45),
                        'anchor_pos': cur_pos,
                        'anchor_box': cur_box,
                        'is_stopped': False,
                        'stationary_start_time': None,
                        'stopped_confirmed_time': None,
                        'last_seen_time': current_time,
                        'outside_roi_start': None,
                        'is_violation': False,
                        'is_new_violation': False,
                        'dwell_time': 0.0,
                        'speed': 999.0,
                        'in_roi': in_roi
                    }

            rec = self.history[track_id]
            rec['last_seen_time'] = current_time
            rec['in_roi'] = in_roi
            rec['pos_history'].append((current_time, cx, cy))

            # Giữ lịch sử vị trí trong cửa sổ quan sát speed_window_seconds
            pos_hist = rec['pos_history']
            while pos_hist and (current_time - pos_hist[0][0]) > self.speed_window_seconds:
                pos_hist.popleft()

            # Tính toán vận tốc tức thời (pixels/giây) và độ dời tịnh tiến
            dt = current_time - pos_hist[0][0]
            if dt >= 0.35: # Cần ít nhất 0.35s để đo chính xác vận tốc
                dx = cx - pos_hist[0][1]
                dy = cy - pos_hist[0][2]
                net_dist = np.hypot(dx, dy)
                speed = net_dist / dt
            else:
                net_dist = 999.0
                speed = 999.0
            rec['speed'] = speed

            # ==============================================================
            # ĐÁNH GIÁ TRẠNG THÁI: XE TRONG VÙNG ROI HAY NGOÀI VÙNG ROI
            # ==============================================================
            if in_roi:
                rec['outside_roi_start'] = None

                # Điều kiện xe đứng yên: vận tốc nhỏ (< 10 px/s) VÀ độ dịch chuyển nhỏ (< 12 px)
                is_currently_stationary = (speed <= self.stop_speed_threshold and net_dist <= self.stop_dist_threshold)

                if not rec['is_stopped']:
                    # XE ĐANG CHẠY:
                    if is_currently_stationary:
                        # Xe bắt đầu có dấu hiệu đứng yên -> Bắt đầu đếm thời gian xác nhận (Debounce)
                        if rec['stationary_start_time'] is None:
                            rec['stationary_start_time'] = current_time
                            rec['anchor_pos'] = cur_pos
                            rec['anchor_box'] = cur_box

                        # Kiểm tra xem xe đã đứng yên liên tục đủ min_stationary_seconds chưa
                        stationary_duration = current_time - rec['stationary_start_time']
                        if stationary_duration >= self.min_stationary_seconds:
                            # ĐÃ ĐỨNG YÊN LIÊN TỤC ĐỦ 1.2 GIÂY -> CHÍNH THỨC XÁC NHẬN DỪNG ĐỖ
                            rec['is_stopped'] = True
                            rec['stopped_confirmed_time'] = rec['stationary_start_time']
                            rec['dwell_time'] = stationary_duration
                        else:
                            # Đang trong thời gian ân hạn kiểm chứng -> Vẫn coi là xe chạy bình thường
                            rec['dwell_time'] = 0.0
                    else:
                        # Xe vẫn đang di chuyển bình thường -> Hủy bộ đếm xác nhận dừng
                        rec['stationary_start_time'] = None
                        rec['stopped_confirmed_time'] = None
                        rec['is_stopped'] = False
                        rec['dwell_time'] = 0.0
                        rec['anchor_pos'] = cur_pos
                        rec['anchor_box'] = cur_box

                else:
                    # XE ĐÃ XÁC NHẬN DỪNG ĐỖ TỪ TRƯỚC:
                    # Kiểm tra xem xe có lăn bánh di chuyển rời đi không:
                    # Khoảng cách dịch chuyển so với vị trí neo đỗ ban đầu
                    dist_from_anchor = np.hypot(cx - rec['anchor_pos'][0], cy - rec['anchor_pos'][1])
                    
                    # Nếu xe di chuyển xa hơn dung sai HOẶC vận tốc tăng lên đáng kể
                    if dist_from_anchor > self.movement_tolerance or speed > (self.stop_speed_threshold * 1.4):
                        # XE ĐÃ LĂN BÁNH DI CHUYỂN RỜI ĐI -> LẬP TỨC HỦY TRẠNG THÁI ĐỖ
                        rec['is_stopped'] = False
                        rec['stationary_start_time'] = None
                        rec['stopped_confirmed_time'] = None
                        rec['dwell_time'] = 0.0
                        rec['is_violation'] = False
                        rec['anchor_pos'] = cur_pos
                        rec['anchor_box'] = cur_box
                    else:
                        # VẪN ĐANG DỪNG ĐỖ TẠI ĐÂY -> TÍNH THỜI GIAN ĐỖ
                        dwell_time = current_time - rec['stopped_confirmed_time']
                        rec['dwell_time'] = dwell_time

                        # Kiểm tra vượt ngưỡng vi phạm dừng đỗ
                        if dwell_time >= self.time_threshold:
                            if not rec['is_violation']:
                                rec['is_violation'] = True
                                rec['is_new_violation'] = True
                            else:
                                rec['is_new_violation'] = False

                            violations.append({
                                'track_id': track_id,
                                'duration': dwell_time,
                                'is_new': rec['is_new_violation']
                            })

            else:
                # Xe đang ở NGOÀI vùng ROI -> Tuyệt đối không tính thời gian dừng đỗ
                rec['is_stopped'] = False
                rec['stationary_start_time'] = None
                rec['stopped_confirmed_time'] = None
                rec['dwell_time'] = 0.0
                rec['is_violation'] = False

                if rec['outside_roi_start'] is None:
                    rec['outside_roi_start'] = current_time

        # Dọn dẹp các xe mất tích quá lâu (> max_missing_seconds) hoặc rời ROI quá lâu (> roi_grace_seconds)
        expired = [
            tid for tid, rec in self.history.items()
            if (tid not in all_current_ids and (current_time - rec['last_seen_time']) > self.max_missing_seconds)
            or (rec.get('outside_roi_start') and (current_time - rec['outside_roi_start']) > self.roi_grace_seconds)
        ]
        for tid in expired:
            del self.history[tid]

        return violations
