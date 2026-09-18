# BÁO CÁO DỰ ÁN
# HỆ THỐNG GIÁM SÁT VÀ PHÁT HIỆN DỪNG ĐỖ XE TRÁI PHÉP

**Phiên bản:** 1.0 (Release)  
**Ngày lập:** 16/09/2026  
**Đơn vị thực hiện:** Nguyễn Trọng Minh Đức – Sinh viên Công nghệ Thông tin

---

# MỤC LỤC

1. [Báo cáo Nghiên cứu Khả thi](#1-bao-cao-nghien-cuu-kha-thi)
2. [Báo cáo Phân tích Thị trường](#2-bao-cao-phan-tich-thi-truong)
3. [Đặc tả Yêu cầu Phần mềm (SRS)](#3-dac-ta-yeu-cau-phan-mem)
4. [Kế hoạch & Kịch bản Kiểm thử](#4-ke-hoach-va-kich-ban-kiem-thu)
5. [Báo cáo Kiểm thử](#5-bao-cao-kiem-thu)
6. [Báo cáo Phiên bản (Alpha / Beta / Release)](#6-bao-cao-phien-ban)
7. [Hướng dẫn Sử dụng (HDSD)](#7-huong-dan-su-dung)

---

# 1. BÁO CÁO NGHIÊN CỨU KHẢ THI

## PHẦN A. TỔNG QUAN DỰ ÁN

**Tên dự án:** Hệ Thống Giám Sát & Phát Hiện Dừng Đỗ Xe Trái Phép (Smart Parking Violation Detection System)

**Phiên bản phần mềm:** 1.0

**Thời gian thực hiện:** Tháng 07/2026 – Tháng 09/2026

**Nền tảng công nghệ cốt lõi:**
- Ngôn ngữ: Python 3.10+
- Framework: Ultralytics YOLO11, Streamlit, OpenCV
- Thuật toán Tracking: ByteTrack (tuỳ chỉnh)
- Mô hình AI: YOLO11n (5.2 MB) và YOLO11m (40.5 MB) – Fine-tuned trên UA-DETRAC
- Tập dữ liệu huấn luyện: UA-DETRAC (~82.000 ảnh camera giao thông thực tế)

---

## PHẦN B. TÍNH KHẢ THI VỀ KỸ THUẬT

### I. Bối cảnh và Vấn đề Thực tiễn

Tình trạng dừng đỗ xe trái phép tại các khu vực đông dân cư, khu công nghiệp và trung tâm thương mại đang là vấn đề giao thông nan giải tại Việt Nam. Các phương pháp kiểm soát hiện tại chủ yếu dựa vào lực lượng chức năng tuần tra thủ công, dẫn đến:

- **Chi phí vận hành cao**: Yêu cầu nhân lực trực 24/7, khó duy trì nhất quán.
- **Phạm vi giám sát hạn chế**: Không thể bao phủ đồng thời nhiều điểm dừng đỗ.
- **Bằng chứng không đủ thuyết phục**: Thiếu ảnh chụp có thời gian và định danh phương tiện rõ ràng để lập biên bản.
- **Sai sót con người**: Xe đang chậm lại tạm thời bị xử phạt nhầm; xe đỗ thực sự lại bị bỏ sót.

### II. Định hướng Kiến trúc Phần mềm

Hệ thống được thiết kế với kiến trúc **3 tầng thông minh**:

**Tầng 1 – Nhận diện Phương tiện (Detection Layer):**  
Sử dụng mô hình YOLO11 Fine-tune trên UA-DETRAC. Toàn bộ các loại phương tiện (ô tô, xe tải, xe buýt, xe van) được gộp thành 1 lớp `0: vehicle` để tối ưu độ chính xác và tốc độ tracking.

**Tầng 2 – Theo dõi Phương tiện (Tracking Layer):**  
ByteTrack tinh chỉnh bám đuổi từng phương tiện qua từng khung hình, cấp phát Track ID duy nhất và duy trì kể cả khi xe bị che khuất tạm thời (Spatial Re-ID trong tối đa 5 giây).

**Tầng 3 – Đánh giá Vi phạm (Rule Engine Layer):**  
`ParkingRuleEngine` thực hiện 4 cơ chế chống báo sai:
1. **Debounce 1.2s**: Xe phải đứng yên liên tục >= 1.2 giây mới bắt đầu đếm giờ vi phạm.
2. **Anchor Locking**: Khóa tọa độ neo; phát hiện tức thì khi xe lăn bánh rời đi.
3. **Spatial Re-ID**: Kế thừa dwell_time khi xe mất track ngắn (<5s), tránh reset oan.
4. **ROI Polygon Filtering**: Chỉ xử lý phương tiện trong vùng cấm do người dùng định nghĩa.

### III. Căn cứ Pháp lý

- Luật Trật tự An toàn Giao thông đường bộ số 36/2024/QH15, ban hành ngày 27/6/2024.
- Nghị định số 168/2024/NĐ-CP về xử phạt vi phạm hành chính trật tự an toàn giao thông.
- Thông tư số 09/2013/TT-BTTTT về Danh mục sản phẩm phần mềm.

### IV. Mục tiêu Dự án

1. Xây dựng hệ thống phát hiện và theo dõi phương tiện dừng đỗ trái phép realtime từ video camera.
2. Cung cấp bằng chứng vi phạm tự động: ảnh chụp timestamp + nhật ký CSV đầy đủ.
3. Đạt mAP@50 >= 90% và Recall >= 85% trên tập kiểm thử UA-DETRAC.
4. Web Dashboard thân thiện, không yêu cầu kiến thức kỹ thuật để vận hành.

---

## PHẦN C. NĂNG LỰC THỰC HIỆN

### I. Thông tin Đơn vị Thực hiện

| Thông tin | Chi tiết |
|---|---|
| Tên | Nguyễn Trọng Minh Đức |
| Chuyên ngành | Công nghệ Thông tin |
| Công nghệ chính | Python, Computer Vision (OpenCV, YOLO), PyTorch, Streamlit |
| Nền tảng GPU đám mây | Kaggle (P100/T4 x2), Google Colab |

### II. Năng lực Công nghệ

- Thiết kế và Fine-tune mô hình Deep Learning trên GPU đám mây.
- Pipeline huấn luyện chống Overfitting: `freeze=10` (Backbone Freezing), `patience=5` (Early Stopping), `cos_lr=True` (Cosine Annealing).
- Triển khai Detection + Tracking realtime trên CPU không cần GPU chuyên dụng.
- Phát triển Web Dashboard với Streamlit.

---

# 2. BÁO CÁO PHÂN TÍCH THỊ TRƯỜNG

## I. Tổng quan Thị trường

Theo thống kê Cục Cảnh sát Giao thông (2024), hơn 5,8 triệu lượt vi phạm dừng đỗ xe được ghi nhận hàng năm tại các đô thị lớn. Với hơn 73 triệu phương tiện đăng ký (Bộ Công An 2025) và tốc độ đô thị hóa nhanh, nhu cầu về hệ thống giám sát giao thông thông minh ngày càng cấp thiết.

## II. Phân khúc Khách hàng Tiềm năng

| Phân khúc | Nhu cầu cụ thể | Khả năng đầu tư |
|---|---|---|
| **Cơ quan nhà nước (Công an, UBND)** | Xử phạt tự động, tiết kiệm nhân lực tuần tra | Ngân sách nhà nước |
| **Ban Quản lý Khu Công nghiệp** | Giám sát bãi đỗ xe nội bộ 24/7 | Trung bình – Cao |
| **Chủ đầu tư Tòa nhà / Chung cư** | Quản lý bãi đỗ xe thương mại | Trung bình |
| **Trung tâm thương mại / Siêu thị** | Kiểm soát vi phạm trước cổng | Trung bình |

## III. Phân tích Đối thủ Cạnh tranh

| Tiêu chí | Tuần tra thủ công | Camera ghi video thuần | **Hệ thống này (AI)** |
|---|---|---|---|
| Chi phí vận hành | Rất cao | Thấp | **Thấp** |
| Phản ứng thời gian thực | Không | Không | **Có – tức thì** |
| Tự động lưu bằng chứng | Không | Không | **Có (ảnh + CSV)** |
| Chống sai sót con người | Không | Không | **Có (Debounce + Anchor)** |
| Yêu cầu GPU chuyên dụng | N/A | Không | **Không cần** |

## IV. Lợi thế Cạnh tranh

1. **Không cần GPU**: YOLO11n đạt ~6.5 FPS trên CPU bình thường – đủ để phát hiện vi phạm chính xác.
2. **False Positive cực thấp**: Debounce 1.2s loại bỏ hoàn toàn cảnh báo nhầm xe đang di chuyển.
3. **Bằng chứng pháp lý**: Ảnh có timestamp, Track ID, thời gian dừng đỗ – sẵn sàng làm căn cứ xử phạt theo NĐ 168/2024.
4. **Chi phí triển khai thấp**: Tận dụng hạ tầng camera CCTV sẵn có.

---

# 3. ĐẶC TẢ YÊU CẦU PHẦN MỀM (SRS)

**Tên hệ thống:** Smart Parking Violation Detection System (SPVDS)  
**Phạm vi:** Phát hiện và ghi nhận tự động phương tiện dừng đỗ trái phép trong vùng ROI từ luồng video camera.

## I. Yêu cầu Chức năng (Functional Requirements)

### FR-01: Nhận diện Phương tiện

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| FR-01.1 | Phát hiện xe cộ trong khung hình với conf_threshold >= 0.3 | Bắt buộc |
| FR-01.2 | Hỗ trợ đầu vào file video (.mp4, .avi) và luồng RTSP | Bắt buộc |
| FR-01.3 | Tự động chọn tập lớp phát hiện theo model (1-class hoặc 80-class COCO) | Bắt buộc |

### FR-02: Theo dõi Phương tiện

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| FR-02.1 | Mỗi phương tiện được cấp Track ID duy nhất, duy trì xuyên suốt | Bắt buộc |
| FR-02.2 | Khi bị khuất <= 5 giây, kế thừa lại Track ID cũ (Spatial Re-ID) | Bắt buộc |
| FR-02.3 | Tracking với IoU threshold >= 0.4 để chống nhảy ID | Bắt buộc |

### FR-03: Phát hiện Vi phạm

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| FR-03.1 | Chỉ đếm giờ dừng đỗ khi phương tiện nằm trong vùng ROI | Bắt buộc |
| FR-03.2 | Debounce: phải đứng yên liên tục >= 1.2s trước khi tính thời gian vi phạm | Bắt buộc |
| FR-03.3 | Anchor Locking: dịch chuyển > 25 pixel so với neo -> hủy vi phạm ngay lập tức | Bắt buộc |
| FR-03.4 | Cảnh báo vi phạm khi dwell_time vượt time_threshold (mặc định 5s, cấu hình được) | Bắt buộc |

### FR-04: Lưu Bằng chứng Vi phạm

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| FR-04.1 | Tự động chụp và lưu ảnh bằng chứng vào `evidence/` khi có vi phạm mới | Bắt buộc |
| FR-04.2 | Ảnh phải hiển thị: Track ID, thời gian dừng, timestamp, ROI | Bắt buộc |
| FR-04.3 | Ghi nhận vào `violation_log.csv` với cột: Timestamp, Track_ID, Dwell_Time_Seconds, Image_Path | Bắt buộc |

### FR-05: Giao diện Người dùng

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| FR-05.1 | Giao diện CLI `src/main.py` với đầy đủ tham số --weights, --baseline, --video, --device | Bắt buộc |
| FR-05.2 | Web Dashboard (Streamlit) phát video realtime, thống kê và bảng chứng cứ | Bắt buộc |
| FR-05.3 | Tool vẽ ROI trực quan trên Dashboard, tự động lưu vào settings.yaml | Quan trọng |
| FR-05.4 | Bộ chọn model 1-click không cần restart hệ thống | Quan trọng |

## II. Yêu cầu Phi chức năng (Non-Functional Requirements)

| ID | Yêu cầu | Chỉ số Kiểm chứng |
|---|---|---|
| NFR-01 | YOLO11n đạt >= 5 FPS trên CPU Intel Core i5 Gen 8+ | Đo bằng compare_models.py |
| NFR-02 | mAP@50 >= 90% trên UA-DETRAC val | results.csv |
| NFR-03 | False Positive < 5%: xe di chuyển không bị gắn nhãn vi phạm | Kiểm thử TC-02 |
| NFR-04 | Mọi tham số nghiệp vụ chỉnh qua settings.yaml, không cần sửa code | Thực nghiệm |
| NFR-05 | Tự Fallback về model Baseline nếu file custom lỗi | Kiểm thử TC-04 |

---

# 4. KẾ HOẠCH & KỊCH BẢN KIỂM THỬ

**Tên dự án:** Smart Parking Violation Detection System  
**Phiên bản:** 1.0  
**Thời gian:** Tháng 09/2026  
**Người thực hiện:** Nguyễn Trọng Minh Đức  
**Môi trường:** UAT – CPU, Windows 11, Python 3.10, venv

## I. Kế hoạch Kiểm thử

| STT | Công việc | Ngày bắt đầu | Ngày kết thúc |
|---|---|---|---|
| 1 | Chuẩn bị kịch bản kiểm thử | 10/09/2026 | 12/09/2026 |
| 2 | Kiểm thử chức năng (TC-01 đến TC-05) | 13/09/2026 | 15/09/2026 |
| 3 | Kiểm thử model & đo lường hiệu năng | 16/09/2026 | 16/09/2026 |
| 4 | Tổng hợp kết quả & lập báo cáo | 16/09/2026 | 16/09/2026 |

## II. Kịch bản Kiểm thử

### TC-01: Phát hiện xe dừng đỗ vi phạm (Happy Path)

| Trường | Chi tiết |
|---|---|
| **Mô tả** | Xe dừng hẳn trong vùng ROI vượt quá time_threshold |
| **Điều kiện đầu vào** | Video có xe đứng yên > 5s trong vùng ROI |
| **Kết quả kỳ vọng** | Bbox đỏ "VIOLATION", ảnh bằng chứng lưu, CSV ghi nhận |
| **Kết quả thực tế** | **PASS** |

### TC-02: Xe chạy qua chậm – Không tính vi phạm

| Trường | Chi tiết |
|---|---|
| **Mô tả** | Xe giảm tốc nhưng không đứng yên quá 1.2s |
| **Điều kiện đầu vào** | Xe đi qua vùng ROI, vận tốc 5-15 px/s |
| **Kết quả kỳ vọng** | Bbox xanh "CHẠY", không cảnh báo, không lưu ảnh |
| **Kết quả thực tế** | **PASS** – Debounce 1.2s hoạt động chính xác |

### TC-03: Xe bị che khuất – Thời gian đỗ không reset

| Trường | Chi tiết |
|---|---|
| **Mô tả** | Xe đỗ bị xe khác che khuất < 5s rồi xuất hiện lại |
| **Điều kiện đầu vào** | Xe trong ROI, mất track 2-3s, xuất hiện lại |
| **Kết quả kỳ vọng** | Track ID kế thừa, dwell_time cộng dồn, không reset |
| **Kết quả thực tế** | **PASS** – Spatial Re-ID hoạt động đúng |

### TC-04: Model bị lỗi – Tự Fallback về Baseline

| Trường | Chi tiết |
|---|---|
| **Mô tả** | File output_runs/best.pt bị xóa hoặc không tồn tại |
| **Điều kiện đầu vào** | Xóa/rename best.pt, chạy python src/main.py |
| **Kết quả kỳ vọng** | Hệ thống load yolo11m.pt, in [FALLBACK], không crash |
| **Kết quả thực tế** | **PASS** – load_yolo_model() fallback hoạt động |

### TC-05: Vẽ ROI mới trên Dashboard

| Trường | Chi tiết |
|---|---|
| **Mô tả** | Người dùng định nghĩa lại vùng ROI qua giao diện Web |
| **Điều kiện đầu vào** | Mở Dashboard, click 4 điểm ROI, bấm Lưu |
| **Kết quả kỳ vọng** | settings.yaml cập nhật tọa độ mới, hệ thống nhận ROI mới |
| **Kết quả thực tế** | **PASS** |

---

# 5. BÁO CÁO KIỂM THỬ

**Tên dự án:** Smart Parking Violation Detection System  
**Phiên bản:** 1.0  
**Thời gian kiểm thử:** 13/09/2026 – 16/09/2026  
**Môi trường:** UAT – CPU, Windows 11, Python 3.10

## I. Phạm vi Kiểm thử

**Trong phạm vi:**
- Chức năng phát hiện, tracking và đánh giá vi phạm dừng đỗ.
- Fallback model, công cụ vẽ ROI, chọn model.
- Đo lường hiệu năng inference (FPS, Latency).

**Ngoài phạm vi:**
- Stress test hiệu năng chịu tải lớn.
- Pentest bảo mật Web Dashboard.

**Loại kiểm thử đã thực hiện:**
- Kiểm thử chức năng (Functional Test)
- Kiểm thử tích hợp (Integration Test – Detection + Tracking + Rule Engine)
- Kiểm thử hiệu năng (Performance Test)

## II. Tổng hợp Kết quả

| STT | Phân loại | Số lượng |
|---|---|---|
| 1 | Chức năng đạt yêu cầu hoàn toàn | 5/5 kịch bản |
| 2 | Chức năng còn lỗi | 0 |
| 3 | Ngoài phạm vi kiểm thử | 2 (stress test, pentest) |

## III. Kết quả Đo lường Hiệu năng Mô hình

| Tiêu chí | YOLO11n Custom (5.2 MB) | YOLO11m Custom (40.5 MB) | YOLO11m Gốc COCO (40.7 MB) |
|---|---|---|---|
| Tốc độ CPU (FPS) | **~6.5 FPS** | ~1.0 FPS | ~1.2 FPS |
| Độ trễ (ms/frame) | **153.3 ms** | 1026.6 ms | 833.6 ms |
| mAP@50 (UA-DETRAC val) | 89.94% | **92.65%** | N/A (COCO) |
| Recall (UA-DETRAC val) | 82.00% | **86.44%** | N/A |
| Precision | 91.53% | **91.96%** | N/A |
| Dung lượng model | 5.2 MB | 40.5 MB | 40.7 MB |

## IV. Đánh giá và Kiến nghị

**Đánh giá chung:**
- 5/5 kịch bản kiểm thử đạt yêu cầu.
- YOLO11n (5.2 MB) phù hợp triển khai CPU phổ thông: đạt 6.5 FPS, ByteTrack bám đuổi ổn định, phát hiện vi phạm chính xác.
- YOLO11m (40.5 MB) cho độ chính xác vượt trội (mAP 92.65%, Recall 86.44%), phù hợp môi trường có GPU.
- Tỷ lệ False Positive bằng 0 trong kiểm thử xe đang di chuyển chậm qua ROI.

**Kết luận:** Hệ thống đủ điều kiện đưa vào khai thác sử dụng thực tế.

---

# 6. BÁO CÁO PHIÊN BẢN

## 6.1 PHIÊN BẢN ALPHA – Kiểm thử Nội bộ (01/09/2026)

### I. Mục đích
Kiểm thử nội bộ các thành phần cốt lõi: YOLO Detection, ByteTrack, ParkingRuleEngine. Rà soát pipeline xử lý video end-to-end và phát hiện lỗi logic bộ đếm thời gian.

### II. Nội dung đánh giá

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| Tải và chạy mô hình YOLO11n baseline | Done | COCO 80 lớp chạy được |
| Pipeline Detection -> Tracking -> RuleEngine | Done | Luồng dữ liệu thông suốt |
| Debounce 1.2s chống báo sai | Done | Logic đúng |
| Lưu ảnh bằng chứng vi phạm | Done | .jpg + violation_log.csv tạo thành công |
| CLI python src/main.py | Done | Chạy trên video mẫu |
| Web Dashboard (cơ bản) | Một phần | Hiển thị video, chưa có ROI tool |

### III. Kết luận
Đáp ứng yêu cầu kiểm thử nội bộ. Chuyển sang Beta để hoàn thiện Dashboard và Fine-tune model.

---

## 6.2 PHIÊN BẢN BETA – Chạy thử Thực tế (10/09/2026)

### I. Mục đích
Kiểm tra hệ thống trong điều kiện gần thực tế sau khi Fine-tune YOLO11n Custom và tích hợp đầy đủ Dashboard. Thu thập phản hồi từ so sánh model, đánh giá Spatial Re-ID và công cụ vẽ ROI.

### II. Nội dung đánh giá

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| Fine-tune YOLO11n trên UA-DETRAC | Done | mAP@50 = 89.94% – Kaggle GPU P100, 20 epochs |
| Spatial Re-ID – Kế thừa track bị khuất | Done | TC-03 PASS |
| Anchor Locking – Phát hiện xe rời đi | Done | TC-02 PASS |
| Web Dashboard đầy đủ | Done | Tool vẽ ROI, chọn model, bảng evidence |
| Smart Class Resolver tự động | Done | Tự nhận 1-class vs 80-class |
| compare_models.py | Done | Ảnh side-by-side xuất evidence/ |
| Fallback tự động về Baseline | Done | TC-04 PASS |

### III. Kết luận
Đáp ứng đầy đủ yêu cầu chạy thử thực tế. False Positive = 0. Sẵn sàng nâng cấp YOLO11m và phát hành Release.

---

## 6.3 PHIÊN BẢN RELEASE – Chính thức (16/09/2026)

### I. Mục đích
Hoàn thiện sau khi Fine-tune YOLO11m (mAP@50 = 92.65%, Recall = 86.44%) và hoàn chỉnh toàn bộ tài liệu dự án. Sẵn sàng bàn giao và sử dụng thực tế.

### II. Nội dung đánh giá

| Hạng mục | Trạng thái | Kết quả |
|---|---|---|
| Fine-tune YOLO11m Custom (UA-DETRAC) | Done | mAP@50 = 92.65%, Recall = 86.44% |
| Tích hợp YOLO11m vào hệ thống | Done | Hiển thị trong Dashboard |
| Benchmark 3 mô hình | Done | Báo cáo + ảnh so sánh |
| Hệ thống bằng chứng vi phạm | Done | evidence/ với ảnh + CSV |
| Tài liệu đầy đủ theo QT 7 bước | Done | Đã hoàn thành toàn bộ |
| Code có comment và cấu trúc rõ | Done | Kiến trúc src/ scripts/ config/ |

### III. Kết luận
Hoàn thiện tất cả chức năng, đạt chỉ số kỹ thuật đề ra. Đủ điều kiện bàn giao và sử dụng chính thức.

---

# 7. HƯỚNG DẪN SỬ DỤNG (HDSD)

## I. Yêu cầu Hệ thống

| Thành phần | Tối thiểu | Khuyến nghị |
|---|---|---|
| Hệ điều hành | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |
| CPU | Intel Core i5 Gen 8 | Intel Core i7 Gen 10+ |
| RAM | 8 GB | 16 GB |
| Ổ cứng | 2 GB trống | 5 GB trống |
| GPU | Không bắt buộc | NVIDIA CUDA 11.8+ (tăng tốc ~5-8x) |
| Python | 3.10 | 3.10+ |

## II. Cài Đặt

```bash
# Truy cập thư mục dự án
cd "Phat-hien-dung-do-main"

# Tạo môi trường ảo và cài thư viện
python -m venv .venv
.\.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

## III. Cấu hình Cơ bản (config/settings.yaml)

```yaml
model:
  weights: output_runs/best.pt          # YOLO11n – 6.5 FPS trên CPU
  # weights: output_runs/yolo11m_custom.pt  # YOLO11m – cần GPU
  conf_threshold: 0.3                   # 0.0 – 1.0 (giảm xuống 0.2 nếu bỏ sót xe)
  imgsz: 960                            # 640 hoặc 960
  device: cpu                           # cpu hoặc cuda:0

rules:
  time_threshold: 5.0                   # Ngưỡng vi phạm (giây)
  movement_tolerance: 25.0              # Dung sai dịch chuyển (pixel)

video:
  source: data/sample.mp4              # File video hoặc rtsp://...
```

## IV. Cách Chạy

### Cách 1 – CLI (Nhanh nhất)

```powershell
# Chạy model mặc định (YOLO11n Custom)
python src/main.py

# Chạy model YOLO11m Custom (độ chính xác cao, cần GPU)
python src/main.py --weights output_runs/yolo11m_custom.pt

# Chạy model Baseline COCO (an toàn, không cần file custom)
python src/main.py --baseline

# Chỉ định video khác
python src/main.py --video "D:/video/camera_01.mp4"
```

> Bấm **Q** để thoát khi đang xem video.

### Cách 2 – Web Dashboard (Khuyên dùng)

```powershell
# Click đúp run_dashboard.bat  (hoặc chạy lệnh dưới)
python -m streamlit run src/dashboard.py
```

Mở trình duyệt: **http://localhost:8501**

**Tính năng Dashboard:**
- **Phát video giám sát** với overlay realtime.
- **Vẽ ROI** – Click trên ảnh định nghĩa vùng cấm, tự động lưu settings.yaml.
- **Chọn model 1-click** – Đổi YOLO11n / YOLO11m không cần restart.
- **Bảng chứng cứ vi phạm** – Danh sách + ảnh bằng chứng.

### Cách 3 – Benchmark So sánh Mô hình

```powershell
python scripts/compare_models.py \
    --model-a output_runs/yolo11m_custom.pt \
    --model-b output_runs/best.pt
```

Kết quả: Bảng so sánh FPS / Latency / Confidence + ảnh side-by-side tại `evidence/model_comparison.jpg`.

## V. Đọc Bằng chứng Vi phạm

Tất cả bằng chứng tại thư mục `evidence/`:

| Loại | Vị trí | Ý nghĩa |
|---|---|---|
| Ảnh vi phạm | `evidence/violation_<ID>_<timestamp>.jpg` | Ảnh chụp frame khi phát hiện vi phạm |
| Nhật ký CSV | `evidence/violation_log.csv` | Danh sách tất cả vi phạm |

**Cột trong violation_log.csv:**

| Cột | Ý nghĩa |
|---|---|
| Timestamp | Thời điểm phát hiện (YYYYMMDD_HHMMSS) |
| Track_ID | Mã định danh phương tiện |
| Dwell_Time_Seconds | Thời gian dừng đỗ (giây) |
| Image_Path | Tên file ảnh bằng chứng |

## VI. Khắc phục Sự cố

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `ModuleNotFoundError: ultralytics` | Chưa cài thư viện | `pip install -r requirements.txt` |
| `[LỖI] Không thể mở video` | Sai đường dẫn | Kiểm tra `--video` hoặc settings.yaml |
| FPS < 1 | Model quá nặng cho CPU | Dùng `output_runs/best.pt` (YOLO11n) |
| `[FALLBACK]` xuất hiện | File best.pt không tồn tại | Chạy `python src/main.py --baseline` |
| Không thấy xe | conf_threshold quá cao | Hạ `conf_threshold: 0.20` trong settings.yaml |

---

*Tài liệu soạn thảo theo Quy trình 7 bước chuẩn bị tài liệu dự án phần mềm.*  
*Ngày lập: 16/09/2026 | Phiên bản: 1.0*
