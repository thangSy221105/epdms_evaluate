# Kế hoạch triển khai hệ thống đánh giá EPDMS cho 300 clip NuRec

## 1. Tên dự án
**NuRec 300 Clips – NAVSIM v2 EPDMS Evaluation and Safety Analysis**

Mục đích của dự án là xây dựng một pipeline đánh giá có thể:
* Tính **EPDMS** theo NAVSIM v2 khi dữ liệu và môi trường hiện tại đáp ứng đầy đủ yêu cầu.
* Tính một chỉ số thay thế có kiểm soát, tên là **NuRec Safety Proxy v1**, khi dữ liệu hiện tại chưa đủ để tái tạo EPDMS chính thức.
* So sánh ảnh hưởng của Rule Filter với:
  $$\alpha \in \{0, 0.5, 1, 2\}$$
  trên bốn mode: `cross_scene`, `no_reasoning`, `noisy`, `opposite_action`.
* Làm rõ một câu hỏi nghiên cứu quan trọng:
  > **Khi $\alpha$ tăng, chất lượng lái thực sự kém an toàn hơn, hay quỹ đạo chỉ khác Ground Truth nên bị ADE phạt?**

---

## 2. Các quyết định đã chốt

### 2.1. Không cài thêm thư viện
* Tuyệt đối không thực hiện: `pip install`, `conda install`.
* Phần proxy chỉ được dùng:
  * Python standard library.
  * NumPy.
  * Pandas.
* Phần EPDMS chính thức chỉ được phép dùng các thư viện NAVSIM, nuPlan, Shapely, SciPy… nếu chúng đã tồn tại sẵn trong môi trường `alpamayo-ar1`.
* Chương trình không được tự động cài thư viện.

### 2.2. Không được tự động đổi từ EPDMS sang proxy
* Người chạy phải chọn rõ profile:
  * `navsim_v2_full`
  * `navsim_v2_stage1`
  * `nurec_safety_proxy_v1`
* Nếu chọn `navsim_v2_full` nhưng thiếu dữ liệu hoặc dependency, chương trình phải dừng và báo rõ nguyên nhân.
* **Không được âm thầm chuyển sang proxy rồi vẫn đặt tên kết quả là EPDMS.**

### 2.3. Khung đánh giá chính là 4 giây
Cấu hình chính:
```json
{
  "horizon_s": 4.0,
  "frequency_hz": 10.0,
  "dt_s": 0.1,
  "future_poses": 40,
  "states_including_t0": 41
}
```
* Ở tần số 10 Hz, 4 giây tương ứng với 40 pose tương lai; khi mô phỏng cần thêm trạng thái hiện tại $t = 0$, tạo thành 41 trạng thái. Code NAVSIM cũng lấy mẫu từ $0$ đến hết horizon và mô phỏng `num_poses + 1` trạng thái.
* Dữ liệu 6.4 giây chỉ dùng cho một phân tích phụ: `nurec_safety_proxy_v1_h6p4`.
* Kết quả 6.4 giây không được gọi là EPDMS chuẩn NAVSIM.

### 2.4. Trọng số NAVSIM v2 không được thay đổi trong kết quả chính
EPDMS stage 1 sử dụng:
* **Nhóm nhân bắt buộc (Multiplicative Gates):**
  $$NC \times DAC \times DDC \times TLC$$
* **Nhóm cộng có trọng số:**
  $$\frac{5EP + 5TTC + 2LK + 2HC + 2EC}{16}$$

Do đó:
$$\text{EPDMS}_{\text{stage1}} = (NC \times DAC \times DDC \times TLC) \times \frac{5EP + 5TTC + 2LK + 2HC + 2EC}{16}$$

*(NAVSIM v2 bổ sung LK, EC, DDC, TLC và cơ chế loại bỏ false-positive penalty khi người lái thật cũng gặp cùng vi phạm).*

Mọi thử nghiệm đổi trọng số phải được lưu riêng dưới tên:
`weight_sensitivity_analysis`
Không được trộn với bảng kết quả chính.

---

## 3. Mục tiêu chi tiết

### 3.1. Mục tiêu nghiên cứu
* **Mục tiêu 1 — Đánh giá ảnh hưởng của $\alpha$**:
  * Đo xem khi tăng $\alpha$: Điểm an toàn có giảm hay không? Tỷ lệ va chạm có tăng hay không? Tỷ lệ ra khỏi vùng xe chạy có tăng hay không? TTC có xấu đi hay không? Độ êm ái có giảm hay không? Tiến độ di chuyển có giảm hay không? Ảnh hưởng có khác nhau giữa bốn mode hay không?
* **Mục tiêu 2 — Kiểm chứng hạn chế của ADE**:
  * Xác định các trường hợp:
    1. ADE xấu hơn nhưng an toàn tốt hơn.
    2. ADE xấu hơn nhưng mức an toàn không đổi.
    3. ADE tốt hơn nhưng an toàn xấu hơn.
    4. ADE và chỉ số an toàn cùng tốt lên hoặc cùng xấu đi.
* **Mục tiêu 3 — Phân tích nguyên nhân thay vì chỉ báo một điểm tổng**:
  * Mỗi record phải có đầy đủ điểm thành phần và thông tin chẩn đoán:
    * Thời điểm va chạm đầu tiên.
    * Đối tượng va chạm.
    * Thời điểm đầu tiên ra ngoài vùng xe chạy.
    * Khoảng cách an toàn nhỏ nhất.
    * Giá trị gia tốc, jerk và yaw lớn nhất.
    * Tiến độ quỹ đạo.
    * Lý do metric không thể tính.
* **Mục tiêu 4 — Bảo đảm kết quả có thể tái tạo**:
  * Hai lần chạy cùng dữ liệu đầu vào, config, mã nguồn, random seed phải tạo kết quả giống hệt nhau.

### 3.2. Mục tiêu kỹ thuật
* Xử lý đủ 300 clip.
* Hỗ trợ bốn mode và bốn mức alpha.
* Không bỏ qua record lỗi một cách âm thầm.
* Có checkpoint và resume.
* Có schema đầu vào rõ ràng.
* Có kiểm thử hình học và động học.
* Có manifest lưu phiên bản môi trường.
* Tách rõ official score và proxy score.
* Có bảng tổng hợp CSV, JSON và Markdown.
* Có thể đối chiếu từng clip khi phát hiện bất thường.

---

## 4. Ba profile đánh giá

### 4.1. `navsim_v2_full`
* Đây là profile cao nhất.
* Bao gồm:
  * Mô phỏng ego bằng LQR và kinematic bicycle model.
  * Tính đầy đủ: NC, DAC, DDC, TLC, TTC, EP, LK, HC, EC.
  * Human penalty filtering.
  * Stage 1.
  * Các synthetic follow-up scene của stage 2.
  * Gaussian weighting.
  * Nhân kết quả stage 1 với tổng hợp stage 2.
* **Được phép đặt tên kết quả:** `epdms_full`.
* **Điều kiện bắt buộc:** Phải có đầy đủ NAVSIM metric cache, ego state hiện tại, PDM reference trajectory, centerline, route lane IDs, drivable-area map, semantic map layers, traffic-light information, past human trajectory, human future trajectory, adjacent-frame mapping để tính EC, second-stage scenes, stage-1-to-stage-2 mapping, traffic agent policy phù hợp.

### 4.2. `navsim_v2_stage1`
* Bao gồm đầy đủ các metric của EPDMS nhưng chỉ đánh giá stage 1.
* **Được phép đặt tên:** `epdms_stage1`.
* **Không được đặt là:** `epdms_full` hoặc `official_final_epdms`.
* **Điều kiện bắt buộc:** Phải tính được đủ 9 thành phần: NC, DAC, DDC, TLC, TTC, EP, LK, HC, EC. (Nếu thiếu EC thì record không được gắn nhãn `epdms_stage1`).

### 4.3. `nurec_safety_proxy_v1`
* Profile dùng cho 3 file JSONL hiện tại trong trường hợp không đủ dữ liệu/dependency của NAVSIM.
* **Tên điểm bắt buộc:** `nurec_safety_proxy_v1`.
* **Công thức cố định:**
  $$S_{\text{proxy}} = CF \times DAC_p \times \frac{5TTC_p + 5EP_{GT} + 2FC}{12}$$
  Trong đó:
  * $CF$: Collision Free proxy.
  * $DAC_p$: Drivable Area Compliance proxy.
  * $TTC_p$: TTC proxy.
  * $EP_{GT}$: Tiến độ theo polyline Ground Truth.
  * $FC$: Future Comfort proxy.
* Không được thêm hoặc bớt metric khỏi mẫu số theo từng record. Nếu một metric bắt buộc không tính được, record phải được đánh dấu `valid = false`.

---

## 5. Quy mô dữ liệu dự kiến
* **Số điều kiện đánh giá logic:**
  $$300 \times 4 \times 4 = 4.800 \text{ conditions}$$
  Mỗi điều kiện xác định bởi: `clip_id + mode + alpha`.
* **Số quỹ đạo duy nhất (nếu $\alpha=0$ giống nhau giữa 4 mode):**
  $$300 \times (1 + 4 \times 3) = 3.900 \text{ trajectories}$$
* Pipeline vẫn xuất 4.800 dòng để tiện so sánh theo mode, nhưng tính quỹ đạo giống nhau một lần rồi tham chiếu lại bằng:
  `trajectory_hash`, `baseline_source_key`, `is_shared_baseline`.
* Khi phân tích thống kê tổng hợp, bốn bản baseline giống nhau không được coi là bốn quan sát độc lập.

---

## 6. Dữ liệu đầu vào

### 6.1. File dự đoán
`D:\300_clip_nurec\00_raw\ar1_output\reasoning_intervention_nurec_selected_300.jsonl`
* Schema: `clip_id`, `mode`, `alpha`, `waypoints`, `timestamps` hoặc `dt`, `coordinate_frame`.
* Waypoint: tối thiểu `x`, `y`; nên có `heading`, `time_s` hoặc `timestamp_us`, `velocity`. Nếu suy ra heading: ghi `heading_source = "derived_from_xy"`.

### 6.2. File context
`D:\300_clip_nurec\01_context\full\nurec_context_full_300.jsonl`
* Cần: `clip_id`, `t0_us`, `ego_current_state`, `obstacle.all_obstacles`, `road_boundary.all_geometry_records`.
* Obstacle: `track_id`, `object_type`, `timestamp`, `x`, `y`, `heading`, `length`, `width`, `valid`. Nếu thiếu kích thước/heading, metric collision phải báo không đủ dữ liệu.

### 6.3. File Ground Truth
`D:\300_clip_nurec\00_raw\ground_truth\ego_future_gt_nurec_300.jsonl`
* Cần: `clip_id`, `gt_waypoints`, `timestamps` hoặc `dt`.
* Dùng cho: Human-reference filtering, `progress_gt_proxy`, ADE/FDE, phát hiện sai lệch timestamp/coordinate frame.

### 6.4. Dữ liệu bổ sung cho NAVSIM chính thức
Cần tìm trong project hoặc tạo adapter tới: `MetricCache`, `centerline`, `route_lane_ids`, `drivable_area_map`, `map_parameters`, `traffic_light_status`, `past_human_trajectory`, `human_trajectory`, `previous_frame_token`, `second_stage_scene_mapping`. Nếu không tìm thấy, profile official tương ứng phải bị khóa.

---

## 7. Cấu trúc thư mục đề xuất

```text
alpamayo-ar1\
├── configs\
│   └── epdms_300.json
│
├── tools\
│   └── epdms\
│       ├── __init__.py
│       ├── schemas.py
│       ├── config.py
│       ├── io_jsonl.py
│       ├── audit.py
│       ├── normalize.py
│       ├── coordinates.py
│       ├── geometry_numpy.py
│       ├── kinematics_numpy.py
│       ├── proxy_metrics.py
│       ├── navsim_adapter.py
│       ├── score_record.py
│       ├── aggregate.py
│       ├── statistics.py
│       ├── reporting.py
│       └── diagnostics.py
│
├── scripts\
│   ├── audit_epdms_inputs.py
│   ├── evaluate_epdms.py
│   ├── summarize_epdms.py
│   └── run_epdms.ps1
│
└── tests\
    └── epdms\
        ├── test_geometry.py
        ├── test_coordinates.py
        ├── test_kinematics.py
        ├── test_proxy_metrics.py
        ├── test_navsim_adapter.py
        ├── test_aggregation.py
        └── fixtures\
```

---

## 8. File cấu hình: `configs\epdms_300.json`

```json
{
  "metric_profile": "nurec_safety_proxy_v1",
  "horizon_s": 4.0,
  "frequency_hz": 10.0,
  "strict_mode": true,
  "resume": true,
  "random_seed": 2026,
  "alphas": [0.0, 0.5, 1.0, 2.0],
  "modes": [
    "cross_scene",
    "no_reasoning",
    "noisy",
    "opposite_action"
  ],
  "vehicle": {
    "reference_point": "rear_axle",
    "front_length_m": 4.049,
    "rear_length_m": 1.127,
    "width_m": 2.297
  },
  "proxy": {
    "touch_is_collision": true,
    "ttc_horizon_s": 1.0,
    "progress_stationary_threshold_m": 5.0,
    "practical_score_delta": 0.01,
    "practical_ade_delta_m": 0.05
  },
  "paths": {
    "prediction_jsonl": "D:\\300_clip_nurec\\00_raw\\ar1_output\\reasoning_intervention_nurec_selected_300.jsonl",
    "context_jsonl": "D:\\300_clip_nurec\\01_context\\full\\nurec_context_full_300.jsonl",
    "ground_truth_jsonl": "D:\\300_clip_nurec\\00_raw\\ground_truth\\ego_future_gt_nurec_300.jsonl",
    "score_dir": "D:\\300_clip_nurec\\02_gtrs\\scores\\epdms",
    "analysis_dir": "D:\\300_clip_nurec\\04_analysis\\epdms"
  }
}
```

*Kích thước xe Pacifica trong nuPlan:*
* Width: $2.297\text{ m}$.
* Front length (tính từ rear axle): $4.049\text{ m}$, Rear length: $1.127\text{ m}$. Tổng dài: $5.176\text{ m}$.
* Tâm hình học cách rear axle: $\frac{4.049 - 1.127}{2} = 1.461\text{ m}$.

---

## 9. Giai đoạn 0 — Kiểm tra môi trường và dữ liệu

### 9.1. Dependency audit (`audit_epdms_inputs.py`)
Kiểm tra: Python version, NumPy, Pandas, NAVSIM import, nuPlan import, Shapely, SciPy, exact git commit của NAVSIM/nuPlan (nếu có).

### 9.2. Data contract audit
Kiểm tra số clip duy nhất, clip trùng, thiếu prediction/context/GT, tần số 10Hz, timestamp, frame tọa độ, heading unwrap, road polygon, obstacle dimensions.

### 9.3. Kết quả audit
Xuất các file báo cáo:
* `D:\300_clip_nurec\04_analysis\epdms\data_contract_report.json`
* `D:\300_clip_nurec\04_analysis\epdms\data_contract_report.md`
* `D:\300_clip_nurec\04_analysis\epdms\missing_records.csv`
* `D:\300_clip_nurec\04_analysis\epdms\duplicate_records.csv`
* `D:\300_clip_nurec\04_analysis\epdms\environment_manifest.json`

Kết luận dứt khoát trạng thái `READY` hoặc `NOT READY` cho từng profile.

---

## 10. Giai đoạn 1 — Chuẩn hóa dữ liệu
* **10.1. Join record:** Key chuẩn: `condition_key = clip_id|mode|alpha`.
* **10.2. Chuẩn hóa thời gian:** Timeline $t = 0.0, 0.1, \dots, 4.0$. Prepend $t=0$ nếu chưa có; lấy đúng 40 future waypoints.
* **10.3. Chuẩn hóa tọa độ:** Ghi nhận origin và biến đổi sang global nếu cần:
  $$x_g = x_0 + x_l \cos\theta_0 - y_l \sin\theta_0$$
  $$y_g = y_0 + x_l \sin\theta_0 + y_l \cos\theta_0$$
  $$\theta_g = \operatorname{wrap}(\theta_0 + \theta_l)$$
* **10.4. Heading:** Dùng `np.unwrap(...)` tránh góc nhảy giữa $\pm\pi$.
* **10.5. Hash và provenance:** Tính SHA-256 các file input, config và `trajectory_hash`.

---

## 11. Giai đoạn 2 — Pipeline NAVSIM v2 chính thức
* Gọi trực tiếp `PDMSimulator`, `PDMScorer`, `SceneAggregator` của NAVSIM.
* Cấu hình traffic agents policy: `log_replay`.
* Tính đầy đủ NC, DAC, DDC, TLC, TTC, EP, LK, HC, EC.
* Lưu song song `raw_metric`, `human_metric`, `filtered_metric`, `filter_applied`.

---

## 12. Giai đoạn 3 — NuRec Safety Proxy v1
* **Hình học xe ego:** Rectangle $5.176\text{m} \times 2.297\text{m}$, tâm cách rear axle $1.461\text{m}$.
* **Collision Free proxy ($CF$):** Đồng bộ timestamp, dùng SAT (Separating Axis Theorem) kiểm tra va chạm giữa ego và obstacles. Overlap = 0, Không overlap = 1.
* **DAC proxy ($DAC_p$):** Kiểm tra 4 góc xe có nằm trong ít nhất một drivable polygon không. 4 góc nằm trong toàn bộ thời gian = 1, có frame chệch = 0.
* **TTC proxy ($TTC_p$):** Chiếu ego theo vận tốc/heading hiện tại qua các mốc $t+0.0, 0.3, 0.6, 0.9\text{s}$. Không va chạm = 1, có nguy cơ va chạm trong 1s = 0.
* **Progress GT proxy ($EP_{GT}$):** Chiếu endpoint lên polyline GT:
  $$EP_{GT} = \begin{cases} 1, & s_{\text{GT}} \le 5\text{ m} \\ \operatorname{clip}\left(\frac{s_{\text{pred}}}{s_{\text{GT}}}, 0, 1\right), & s_{\text{GT}} > 5\text{ m} \end{cases}$$
* **Future Comfort proxy ($FC$):** Kiểm tra 6 ngưỡng tham chiếu NAVSIM:
  * Jerk magnitude $\le 8.37\text{ m/s}^3$
  * Lateral acceleration $\le 4.89\text{ m/s}^2$
  * Longitudinal accel $\in [-4.05, 2.40]\text{ m/s}^2$
  * Longitudinal jerk $\le 4.13\text{ m/s}^3$
  * Yaw acceleration $\le 1.93\text{ rad/s}^2$
  * Yaw rate $\le 0.95\text{ rad/s}$
* Các metric mở rộng (LK, DDC, TLC, HC, EC) được xuất riêng chẩn đoán, không đưa vào mẫu số của proxy v1.

---

## 13. Schema kết quả từng condition
Mỗi dòng JSONL có cấu trúc chuẩn chi tiết: `record_key`, `clip_id`, `mode`, `alpha`, `rule_group`, `metric_profile`, `valid`, `collision_free_proxy`, `dac_proxy`, `ttc_proxy`, `progress_gt_proxy`, `future_comfort_proxy`, `nurec_safety_proxy_v1`, thời điểm vi phạm đầu tiên, các giá trị động học cực đại, runtime, SHA256 hashes...

---

## 14. Cơ chế lỗi và resume
* Không crash toàn bộ batch vì một clip (bọc trong error boundary).
* Ghi log lỗi vào `epdms_errors_300.jsonl`.
* Checkpoint flush mỗi 25–50 conditions; hỗ trợ resume chính xác dựa trên hash.
* Atomic write qua file `*.tmp` tránh hỏng file khi dừng đột ngột.

---

## 15. File kết quả
* **Dữ liệu chi tiết:**
  * `D:\300_clip_nurec\02_gtrs\scores\epdms\epdms_scores_300.jsonl`
  * `D:\300_clip_nurec\02_gtrs\scores\epdms\epdms_scores_300.csv`
  * `D:\300_clip_nurec\02_gtrs\scores\epdms\epdms_errors_300.jsonl`
  * `D:\300_clip_nurec\02_gtrs\scores\epdms\run_manifest.json`
* **Báo cáo tổng hợp:**
  * `D:\300_clip_nurec\04_analysis\epdms\epdms_by_mode_alpha.csv` / `.md`
  * `D:\300_clip_nurec\04_analysis\epdms\epdms_by_rule_group_alpha.csv` / `.md`
  * `D:\300_clip_nurec\04_analysis\epdms\component_failure_rates.csv`
  * `D:\300_clip_nurec\04_analysis\epdms\paired_delta_vs_alpha0.csv`
  * `D:\300_clip_nurec\04_analysis\epdms\ade_safety_disagreement.csv`
  * `D:\300_clip_nurec\04_analysis\epdms\final_research_summary.md` / `.json`

---

## 16. Phân tích thống kê
* So sánh theo cặp: $\Delta \text{Score}_\alpha = \text{Score}_\alpha - \text{Score}_{\alpha=0}$.
* Bảng thống kê theo mode/alpha với Mean, Median, Std, CI 95%, Tỷ lệ cải thiện/suy giảm.
* Phân định 2 mức không đổi:
  * Numerical tie: $|\Delta| \le 10^{-9}$
  * Practical tie: $|\Delta| < 0.01$
* Bootstrap confidence interval (5000 iterations, seed 2026), lấy mẫu theo `clip_id`.

---

## 17. Đối chiếu ADE với an toàn
Bảng ma trận phân loại:

| Thay đổi ADE | Thay đổi Safety | Diễn giải |
| :--- | :--- | :--- |
| **Xấu hơn** | **Tốt hơn** | Quỹ đạo khác GT nhưng có dấu hiệu an toàn hơn |
| **Xấu hơn** | **Không đổi** | ADE nhạy với khác biệt hình học, safety không đổi đáng kể |
| **Tốt hơn** | **Xấu hơn** | Gần GT hơn nhưng metric an toàn kém hơn |
| **Tốt hơn** | **Tốt hơn** | Cả hai metric cùng xác nhận cải thiện |

*Ngưỡng phân loại:* ADE practical delta $= 0.05\text{ m}$, Safety practical delta $= 0.01$.

---

## 18. Verification Plan
* **Unit test hình học:** SAT 2 rectangles (xa nhau, overlap, chạm cạnh, chạm góc, xoay $45^\circ$, lồng nhau), Point-in-polygon.
* **Unit test thời gian & tọa độ:** Frame transforms, heading unwrap, timeline 10Hz.
* **Unit test động học:** Xe đứng yên, chạy thẳng đều, phanh gấp, quay tròn, zigzag.
* **Scenario test tổng hợp:** Case A (Safe straight), Case B (Direct collision), Case C (Near miss), Case D (One corner off-road), Case E (Aggressive braking), Case F (Heading wrap), Case G (Stationary GT), Case H (Wavy path).
* **Invariance test & Regression test:** Không đổi kết quả khi tịnh tiến/xoay tọa độ.

---

## 19. Chạy thử theo từng mức
* **Mức 1 — Synthetic tests:** Chạy toàn bộ unit test và scenario test.
* **Mức 2 — Curated real clips:** Chọn 10–20 clip đại diện (đường thẳng, rẽ, giao lộ, đông xe, can thiệp mạnh).
* **Mức 3 — Một mode, một alpha:** 300 clip với `mode = cross_scene, alpha = 0`.
* **Mức 4 — Một mode, đủ alpha:** $300 \times 4 = 1.200\text{ conditions}$.
* **Mức 5 — Toàn bộ:** $4.800\text{ conditions}$.

---

## 20. Lệnh chạy dự kiến
```powershell
# 20.1. Audit
python .\scripts\audit_epdms_inputs.py --config .\configs\epdms_300.json

# 20.2. Chạy proxy
python .\scripts\evaluate_epdms.py --config .\configs\epdms_300.json --profile nurec_safety_proxy_v1 --horizon 4.0 --resume

# 20.3. Chạy NAVSIM stage 1 (khi đủ điều kiện)
python .\scripts\evaluate_epdms.py --config .\configs\epdms_300.json --profile navsim_v2_stage1 --traffic-agents-policy log_replay --resume

# 20.4. Chạy NAVSIM full (khi đủ điều kiện)
python .\scripts\evaluate_epdms.py --config .\configs\epdms_300.json --profile navsim_v2_full --resume

# 20.5. Tổng hợp
python .\scripts\summarize_epdms.py --config .\configs\epdms_300.json --score-file "D:\300_clip_nurec\02_gtrs\scores\epdms\epdms_scores_300.jsonl"
```

---

## 21. Tiêu chí nghiệm thu
* **Data integrity:** Đúng số clip, không duplicate, không bỏ sót record âm thầm.
* **Metric integrity:** Proxy không gắn nhãn EPDMS, official score chỉ xuất khi đủ component, không thay NaN bằng 0.
* **Testing:** 100% unit test bắt buộc pass, invariance test pass.
* **Batch completion:** Đủ 4.800 conditions logic, error file đầy đủ, resume chuẩn xác.
* **Research outputs:** Trả lời dứt khoát các câu hỏi: $\alpha$ nào làm giảm điểm nhiều nhất? Mode nào nhạy nhất? Suy giảm do va chạm, lệch đường, comfort hay progress? Tỷ lệ ADE phạt nhầm là bao nhiêu?

---

## 22. Trình tự triển khai chính thức
1. **Phase 0 — Audit:** Xuất `environment_manifest.json`, `data_contract_report.json`, `data_contract_report.md`.
2. **Phase 1 — Normalization:** Xuất `normalized_sample_20.jsonl`, `normalization_diagnostics.csv`.
3. **Phase 2 — Proxy engine:** Xây dựng core SAT, point-in-polygon, kinematics, CF, DAC, TTC, Progress, Comfort.
4. **Phase 3 — Official NAVSIM adapter:** Kích hoạt nếu audit xác nhận dependency.
5. **Phase 4 — Tests & synthetic verification:** Chạy test suite trước khi chạy dữ liệu lớn.
6. **Phase 5 — Full batch:** Chạy 4.800 conditions và lưu manifest.
7. **Phase 6 — Comparative analysis:** Join với ADE và tính paired delta.
8. **Phase 7 — Research summary:** Xuất `final_research_summary.md` và `.json`.

---

## 23. Kết quả cuối cùng cần đạt
Hệ thống cho ra 2 lớp kết luận độc lập:
* **Lớp 1 — Kết luận metric:** Phân tích điểm số chi tiết từng thành phần (Comfort, Progress, Collision, Off-road).
* **Lớp 2 — Kết luận ADE đối chiếu safety:** Phân định rõ ràng giữa "quỹ đạo thực sự kém an toàn" và "quỹ đạo chỉ bị ADE phạt do lệch khỏi hành vi con người".
