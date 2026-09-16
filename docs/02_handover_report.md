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

## 5. Hướng dẫn Đẩy mã nguồn lên Git (GitHub Push)

Để đưa toàn bộ mã nguồn vừa tạo lên repo cá nhân của bạn trên GitHub, bạn chỉ cần thực hiện các lệnh sau trong PowerShell tại thư mục `epdms_evaluate`:

```powershell
cd "c:\Users\DELL\OneDrive\Tài liệu\ChatGPT\read paper\epdms_evaluate"

# 1. Kiểm tra trạng thái các file mới
git status

# 2. Thêm tất cả file vào staging
git add .

# 3. Tạo commit bàn giao
git commit -m "feat: complete NuRec 300 EPDMS and Safety Proxy v1 evaluation pipeline"

# 4. Đẩy code lên nhánh chính (main) trên GitHub
git push origin main
```

---

## 6. Kết luận
Hệ thống đánh giá đã được xây dựng hoàn chỉnh, tuân thủ nghiêm ngặt 100% các quyết định kỹ thuật và nguyên tắc nghiên cứu trong bản kế hoạch `01_epdms_evaluation_plan.md`. Hệ thống sẵn sàng cho việc chạy thực nghiệm quy mô toàn bộ 300 clip (4.800 conditions) để đưa ra câu trả lời dứt khoát cho câu hỏi nghiên cứu của bạn.
