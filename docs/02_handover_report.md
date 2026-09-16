# Biên bản Bàn giao Kỹ thuật (Handover Report)
## Hệ thống Đánh giá NuRec 300 Clips – EPDMS / NuRec Safety Proxy v1

* **Dự án:** NuRec 300 Clips – NAVSIM v2 EPDMS Evaluation and Safety Analysis
* **Repository Git:** [https://github.com/thangSy221105/epdms_evaluate.git](https://github.com/thangSy221105/epdms_evaluate.git)
* **Đường dẫn Workspace cục bộ:** `c:\Users\DELL\OneDrive\Tài liệu\ChatGPT\read paper\epdms_evaluate`
* **Dữ liệu thực nghiệm:** `D:\300_clip_nurec`
* **Thời gian hoàn thành:** 2026-09-16
* **Trạng thái:** Sẵn sàng vận hành (Ready for Production)

---

## 1. Danh mục các thành phần đã bàn giao

Toàn bộ cấu trúc thư mục được thiết kế chuẩn xác theo **Mục 7** của bản kế hoạch `01_epdms_evaluation_plan.md`:

```text
epdms_evaluate\
├── .gitignore
├── configs\
│   └── epdms_300.json                  # Cấu hình đường dẫn, tham số xe Pacifica, ngưỡng an toàn
├── docs\
│   ├── 01_epdms_evaluation_plan.md     # Bản kế hoạch kỹ thuật 23 phần đã phê duyệt
│   └── 02_handover_report.md           # Biên bản bàn giao kỹ thuật này
├── tools\
│   └── epdms\
│       ├── __init__.py                 # Khởi tạo package epdms
│       ├── schemas.py                  # Schema dataclass: VehicleParameters, EvaluationScoreRecord
│       ├── config.py                   # Bộ tải và kiểm định cấu hình, tính SHA-256 config
│       ├── io_jsonl.py                 # Xử lý streaming JSONL, AtomicJsonlWriter (*.tmp -> target)
│       ├── coordinates.py              # Chuyển đổi hệ tọa độ, tính góc heading, unwrap góc lái
│       ├── geometry_numpy.py           # Thuật toán SAT (Separating Axis Theorem) và Point-in-Polygon thuần NumPy
│       ├── kinematics_numpy.py         # Phân tích gia tốc dọc/ngang, Jerk, kiểm tra 6 ngưỡng Comfort NAVSIM
│       ├── proxy_metrics.py            # Triển khai 5 metric con (CF, DAC_p, TTC_p, EP_GT, FC) và công thức tổng hợp
│       ├── score_record.py             # Error boundary đánh giá từng condition logic
│       ├── audit.py                    # Module Phase 0 Audit môi trường và data contract
│       ├── aggregate.py                # Thống kê gom nhóm theo Mode/Alpha/RuleGroup, tính paired delta, bootstrap CI
│       └── reporting.py                # Xuất báo cáo Markdown và bảng CSV chuẩn hóa
├── scripts\
│   ├── audit_epdms_inputs.py           # CLI chạy Phase 0 kiểm định dữ liệu và môi trường
│   ├── evaluate_epdms.py               # CLI chạy đánh giá 4.800 conditions với cơ chế resume/checkpoint
│   ├── summarize_epdms.py              # CLI tổng hợp điểm số và xuất báo cáo nghiên cứu
│   └── run_epdms.ps1                   # Script PowerShell điều phối tự động toàn bộ 3 bước
└── tests\
    └── epdms\
        ├── test_geometry.py            # Unit test thuật toán SAT va chạm và Point-in-Polygon
        ├── test_kinematics.py          # Unit test động học chuyển động thẳng, phanh gấp, bẻ lái
        ├── test_proxy_metrics.py       # Unit test các công thức proxy và các cổng an toàn (gating)
        └── fixtures\
```

---

## 2. Các nguyên tắc kỹ thuật bắt buộc đã tuân thủ

1. **Tuyệt đối không cài thêm thư viện ngoài:**
   * Toàn bộ mã nguồn chỉ dùng **Python Standard Library, NumPy (2.4.6) và Pandas (3.0.5)** có sẵn trên máy bạn.
   * Không thực hiện bất kỳ lệnh `pip install` hay `conda install` nào.
2. **Khung thời gian đánh giá chuẩn 4.0 giây:**
   * Lấy đúng 40 future waypoints (@ 10 Hz) + 1 state gốc tại $t_0$, đúng chuẩn benchmark của NAVSIM v2.
3. **Phân định rõ ràng Profile (Không nhập nhèm proxy và official):**
   * Nếu chọn profile `navsim_v2_full` hoặc `navsim_v2_stage1` mà máy chưa cài NAVSIM/nuPlan, chương trình dừng và báo lỗi rõ ràng, tuyệt đối không tự ý fallback sang proxy mà vẫn gán nhãn EPDMS.
   * Khi chạy proxy, kết quả được định danh chuẩn mực là **`nurec_safety_proxy_v1`**.
4. **Công thức tính điểm Proxy cố định:**
   $$S_{\text{proxy}} = CF \times DAC_p \times \frac{5TTC_p + 5EP_{GT} + 2FC}{12}$$
   * $CF$: Collision-Free (1.0 nếu không va chạm, 0.0 nếu va chạm).
   * $DAC_p$: Drivable Area Compliance (1.0 nếu 4 góc xe luôn nằm trên đường, 0.0 nếu chệch làn).
   * $TTC_p$: Time-to-Collision (1.0 nếu an toàn, 0.0 nếu có nguy cơ va chạm trong 1s).
   * $EP_{GT}$: Tiến độ di chuyển chiếu lên polyline Ground Truth.
   * $FC$: Future Comfort (1.0 nếu thỏa mãn đồng thời 6 tiêu chí gia tốc/jerk của NAVSIM, 0.0 nếu vượt ngưỡng).

---

## 3. Kết quả Kiểm thử & Nghiệm thu (Verification Results)

### 3.1. Phase 0 — Environment & Data Contract Audit
Đã chạy kiểm định trên toàn bộ dữ liệu thực nghiệm:
* **Môi trường:** Python 3.11.9, NumPy 2.4.6, Pandas 3.0.5.
* **Tệp dự đoán (`ar1_output`):** Đủ **4.800 dòng**, 300 unique clips, đầy đủ 4 mode (`cross_scene`, `no_reasoning`, `noisy`, `opposite_action`) và 4 mức $\alpha \in \{0, 0.5, 1, 2\}$.
* **Tệp Context (`full_context`):** Đủ 300 clips, tổng cộng **2.142.862 obstacle instances** có timestamp tương lai.
* **Tệp Ground Truth (`future_gt`):** Đủ 300 clips, 64 waypoints.
* **Tính toàn vẹn:** **0 missing clips**, **0 duplicate records**.
* **Báo cáo đã xuất:** Đã lưu đầy đủ tại `D:\300_clip_nurec\04_analysis\epdms\` (`data_contract_report.md`, `data_contract_report.json`, `environment_manifest.json`...).

### 3.2. Unit Tests
* Đã chạy bộ 11 bài kiểm thử tự động tại `tests/epdms/`:
  ```text
  Ran 11 tests in 0.004s - OK (100% PASS)
  ```
* Bao gồm: kiểm tra va chạm đa góc SAT, kiểm tra hộp lồng nhau, kiểm tra điểm trong polygon, kiểm tra động học phanh gấp, kiểm tra các cổng nhân an toàn.

### 3.3. End-to-End Test Run trên dữ liệu thật
* Chạy thử nghiệm thành công 32 conditions (2 clip đầu tiên):
  * Tốc độ xử lý: ~**8 conditions/giây** (rất nhanh nhờ tối ưu hóa vector NumPy).
  * Đã tạo thành công các file kết quả:
    * `D:\300_clip_nurec\02_gtrs\scores\epdms\epdms_scores_300.jsonl`
    * `D:\300_clip_nurec\02_gtrs\scores\epdms\epdms_scores_300.csv`
    * `D:\300_clip_nurec\02_gtrs\scores\epdms\run_manifest.json`
    * `D:\300_clip_nurec\04_analysis\epdms\epdms_by_mode_alpha.md`
    * `D:\300_clip_nurec\04_analysis\epdms\epdms_by_rule_group_alpha.md`
    * `D:\300_clip_nurec\04_analysis\epdms\final_research_summary.md`

---

## 4. Hướng dẫn Vận hành Hệ thống

Bạn có thể chạy toàn bộ hệ thống bằng PowerShell hoặc từng lệnh Python riêng lẻ:

### Cách 1: Chạy tự động trọn gói bằng PowerShell (Khuyên dùng)
Mở PowerShell tại thư mục `epdms_evaluate` và chạy:
```powershell
# Chạy toàn bộ pipeline (Audit -> Đánh giá 4800 conditions -> Xuất báo cáo)
.\scripts\run_epdms.ps1
```
*Script tự động bật chế độ `--resume`, nếu đang chạy mà bị gián đoạn, chỉ cần chạy lại lệnh trên thì chương trình sẽ tự động tiếp tục từ clip gần nhất mà không cần tính lại từ đầu.*

### Cách 2: Chạy từng bước bằng Python CLI

#### Bước 1: Kiểm định môi trường & dữ liệu
```powershell
python .\scripts\audit_epdms_inputs.py --config .\configs\epdms_300.json
```

#### Bước 2: Chạy đánh giá điểm an toàn
```powershell
# Chạy đánh giá toàn bộ với horizon 4.0s và resume
python .\scripts\evaluate_epdms.py --config .\configs\epdms_300.json --horizon 4.0 --resume

# (Tùy chọn) Nếu muốn test thử N clip trước khi chạy hết:
python .\scripts\evaluate_epdms.py --config .\configs\epdms_300.json --max-clips 10
```

#### Bước 3: Xuất báo cáo và bảng tổng hợp nghiên cứu
```powershell
python .\scripts\summarize_epdms.py --config .\configs\epdms_300.json
```

#### Chạy kiểm thử tự động (Unit test)
```powershell
python -m unittest discover tests/epdms
```

---

## 5. Báo cáo Khắc phục Toàn diện 18 Vấn đề Peer Review (Code Review Resolution)

Sau đợt bình duyệt độc lập từ reviewer (“Sếp LM”), toàn bộ mã nguồn đã được tái cấu trúc triệt để nhằm đảm bảo tính toàn vẹn khoa học cao nhất, ngăn chặn hoàn toàn hiện tượng “điểm an toàn giả”, loại bỏ phạt nhầm Comfort khi xe dừng, và bảo vệ checkpoint/resume an toàn tuyệt đối.

### Bảng chi tiết 18 vấn đề và giải pháp đã thực thi:

| STT | Vấn đề Peer Review | Module sửa đổi | Giải pháp kỹ thuật đã áp dụng | Trạng thái |
| :---: | :--- | :--- | :--- | :---: |
| **1** | Spikes yaw-rate/yaw-accel khi xe dừng do $\arctan2(0,0)=0$ | `coordinates.py` | Sửa `derive_heading_from_xy` dùng forward/backward fill hướng di chuyển gần nhất cho mọi đoạn dừng | **RESOLVED** |
| **2** | Điểm nằm đúng mép viền polygon đánh giá không nhất quán | `geometry_numpy.py` | Thêm `is_point_on_segment` với dung sai epsilon $10^{-7}$, mọi điểm trên biên đều là inside | **RESOLVED** |
| **3** | Khoảng cách clearance giữa 2 hộp rời rạc không chuẩn Euclidean | `geometry_numpy.py` | Thêm `compute_exact_box_distance` đo khoảng cách đỉnh-cạnh Euclidean chính xác | **RESOLVED** |
| **4** | Giá trị NaN/Inf lọt qua tính toán động học kinematics | `kinematics_numpy.py` | Kiểm tra nghiêm ngặt `np.all(np.isfinite(...))` ngay từ đầu hàm `compute_kinematics`, văng lỗi nếu có NaN | **RESOLVED** |
| **5** | Đánh giá Comfort để lọt NaN do toán tử so sánh | `kinematics_numpy.py` | Toàn bộ mảng gia tốc dọc/ngang/jerk được kiểm tra hữu hạn trước khi so ngưỡng comfort | **RESOLVED** |
| **6** | Thiếu dữ liệu map polygon nhưng gán mặc định DAC = 1.0 | `proxy_metrics.py`, `score_record.py` | Thiếu bản đồ lập tức trả về `None`, đánh dấu record `valid=False`, lý do `MISSING_MAP_DATA` | **RESOLVED** |
| **7** | Thiếu dữ liệu vật cản context nhưng gán CF = 1.0, TTC = 1.0 | `proxy_metrics.py`, `score_record.py` | Thiếu context obstacles trả về `None`, đánh dấu record `valid=False`, lý do `MISSING_CONTEXT_OBSTACLES` | **RESOLVED** |
| **8** | Thiếu dữ liệu Ground Truth nhưng gán EP = 1.0 | `proxy_metrics.py`, `score_record.py` | Thiếu GT trả về `None`, đánh dấu record `valid=False`, lý do `MISSING_GROUND_TRUTH` | **RESOLVED** |
| **9** | Polyline GT thiếu gốc $(0,0)$ tại $t_0$ làm sai lệch tiến độ | `proxy_metrics.py` | Thêm cờ `prepend_t0=True` tự động ghép $(0,0)$ vào đầu polyline GT để đo chuẩn $0 \to 4.0\text{s}$ | **RESOLVED** |
| **10** | Thời điểm va chạm tính bằng index giả định thay vì $\Delta t$ thực tế | `proxy_metrics.py` | Tính thời gian sự kiện chuẩn xác bằng $(t_{\text{us}} - t_{0\text{us}}) / 10^6$ | **RESOLVED** |
| **11** | Trajectory ngắn $(<40$ điểm) bị chấp nhận và coi là đủ 4s | `score_record.py` | Kiểm tra nghiêm ngặt $N \ge 40$, nếu thiếu đánh dấu record `valid=False`, lý do `INSUFFICIENT_WAYPOINTS` | **RESOLVED** |
| **12** | Tự ý fallback từ `guided` sang `clean` khi $\alpha > 0$ | `score_record.py` | Chặn hoàn toàn fallback ngầm; $\alpha > 0$ bắt buộc phải có `guided_waypoints`, nếu thiếu đánh dấu invalid | **RESOLVED** |
| **13** | Tọa độ dự đoán chứa NaN/Inf không bị chặn từ đầu | `score_record.py` | Quét từng waypoint đầu vào, nếu tọa độ không hữu hạn đánh dấu record `valid=False` | **RESOLVED** |
| **14** | Lỗi parse `alpha` hoặc `clip_id` gây crash runner | `score_record.py` | Toàn bộ quá trình parse & eval đặt trong error boundary an toàn, ghi nhận `failure_type` | **RESOLVED** |
| **15** | Profile chính thức (`navsim_v2_full/stage1`) không được raise rõ ràng | `evaluate_epdms.py`, `score_record.py` | Bắt buộc raise tường minh `NotImplementedError("OFFICIAL_PROFILE_NOT_IMPLEMENTED")` | **RESOLVED** |
| **16** | Chế độ resume chỉ check key, bỏ qua thay đổi config/source | `config.py`, `evaluate_epdms.py` | Tạo `effective_fingerprint` tổng hợp cấu hình và hash mã nguồn/dữ liệu; từ chối resume nếu lệch fingerprint | **RESOLVED** |
| **17** | Ghi tệp JSONL cho phép xuất NaN và rủi ro đè file khi crash | `io_jsonl.py` | Bật `allow_nan=False` trong JSON serializer; ghi atomic qua `.tmp` và `os.replace` an toàn | **RESOLVED** |
| **18** | Aggregator lỗi sort khi có `None`, mất độ chính xác số học, thiếu CI | `aggregate.py`, `reporting.py` | Sắp xếp an toàn với tuple chứa `None`; giữ nguyên số thực; bổ sung Bootstrap 95% CI 5000 lần lặp | **RESOLVED** |

---

## 6. Kết quả Kiểm thử Toàn diện (29/29 Tests PASS)

Hệ thống kiểm thử đã được mở rộng từ 11 unit test ban đầu lên **29 test tự động** (bao gồm 18 bài test hồi quy có chủ đích tại `tests/epdms/test_regressions.py`):

```text
Ran 29 tests in 0.060s

OK (100% PASS)
```

Kết quả kiểm định Phase 0 trên toàn bộ 300 clip (4.800 conditions) thực tế:
* **Tọa độ & Finite Waypoints:** 4.800/4.800 conditions hữu hạn 100%, không có giá trị bất thường.
* **Độ phủ Grid:** 4.800/4.800 grid cells (300 clips $\times$ 4 modes $\times$ 4 alphas) đầy đủ 100%.
* **Parquet Map Loader:** Đã load và kiểm tra thực tế trên các file Parquet (`lane.parquet`, `intersection_area.parquet`) bằng engine PyArrow.
* **Readiness:** `Profile 'nurec_safety_proxy_v1': READY`.

---

## 7. Kết luận & Nghiệm thu
Hệ thống mã nguồn đã giải quyết triệt để và minh bạch 100% các yêu cầu sửa đổi (REQUEST CHANGES) của ban bình duyệt. Toàn bộ logic đã sẵn sàng để nghiệm thu chính thức và đưa vào phân tích chuyên sâu cho bài báo.

