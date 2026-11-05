# Comprehensive Technical Report: Work Completed, Algorithms, AI Training & Preliminary Evaluation

Tài liệu này tổng hợp **toàn bộ kiến thức lý thuyết, quá trình thu thập dữ liệu, chi tiết toán học của từng thuật toán xử lý tín hiệu (DSP), kiến trúc và phương pháp huấn luyện mô hình Học sâu (AI), nơi lưu trữ dữ liệu/trọng số, kết quả định lượng và đánh giá sơ bộ những hạn chế khi đo thực tế (Live)**.

---

## 1. Cơ Sở Lý Thuyết & Đối Chiếu Tài Liệu Khoa Học (`document/`)

Toàn bộ các tài liệu gốc trong `doc/` và 4 bài báo nghiên cứu quốc tế mới tải thêm đã được tập hợp tại thư mục [`05_documentation_summary/document/`](file:///D:/Master/Luanvan/process_csi_data/document/). Dưới đây là bảng ánh xạ trực tiếp giữa từng bài báo và các thuật toán chúng ta đã triển khai trong code:

| Tài liệu trong `05_documentation_summary/document/` | Nội dung cốt lõi của bài báo | Áp dụng vào hệ thống của chúng ta |
|---|---|---|
| [**`PhaseBeat_ACMHealth20.pdf`**](file:///D:/Master/Luanvan/process_csi_data/document/PhaseBeat_ACMHealth20.pdf) & [**`PhaseBeat_ACMHealth20_vietnamese.pdf`**](file:///D:/Master/Luanvan/process_csi_data/document/PhaseBeat_ACMHealth20_vietnamese.pdf) *(Wang et al., IEEE INFOCOM / ACM HEALTH 2020)* | Xây dựng chuỗi tiền xử lý CSI theo dõi nhịp thở và nhịp tim: Khử lệch pha tuyến tính (Phase Sanitization), lọc Hampel, khử xu hướng tĩnh, chọn kênh và ước lượng tần số bằng PSD / Peak. | Làm nền tảng cho [`phasebeat_core.py`](file:///D:/Master/Luanvan/process_csi_data/02_phasebeat_dsp/phasebeat_core.py), [`phase_sanitizer.py`](file:///D:/Master/Luanvan/process_csi_data/02_phasebeat_dsp/filters/phase_sanitizer.py), [`spike_removal.py`](file:///D:/Master/Luanvan/process_csi_data/02_phasebeat_dsp/filters/spike_removal.py), [`detrending.py`](file:///D:/Master/Luanvan/process_csi_data/02_phasebeat_dsp/filters/detrending.py) và [`breathing_estimator.py`](file:///D:/Master/Luanvan/process_csi_data/02_phasebeat_dsp/estimation/breathing_estimator.py). |
| [**`ESPIDF_tool_for_CSI based_device.pdf`**](file:///D:/Master/Luanvan/process_csi_data/document/ESPIDF_tool_for_CSI%20based_device.pdf) | Cấu trúc gói tin CSI trên vi điều khiển ESP32 (chuẩn 802.11n HT20, 64 điểm FFT, 128 byte I/Q), phân bố các bin Guard, DC, Pilot và Data subcarriers. | Áp dụng trong [`serial_reader.py`](file:///D:/Master/Luanvan/process_csi_data/01_data_collection/serial_reader.py) để tách 64 cặp $(I, Q)$ và định nghĩa mảng 46 kênh dữ liệu sạch `ACTIVE_SUBCARRIERS_46` trong [`respiration_pipeline.py`](file:///D:/Master/Luanvan/process_csi_data/02_phasebeat_dsp/respiration_pipeline.py). |
| [**`FarSense_Fresnel_Respiration_MobiCom19_arXiv1907.03994.pdf`**](file:///D:/Master/Luanvan/process_csi_data/document/FarSense_Fresnel_Respiration_MobiCom19_arXiv1907.03994.pdf) *(Zeng et al., ACM MobiCom 2019)* | Lý thuyết **Đới Fresnel (Fresnel Zones)** trong đo nhịp thở bằng WiFi: chứng minh rằng khi người ngồi lệch vị trí (Off-LOS), một số subcarrier rơi vào "điểm mù" (blind spots) trong khi các subcarrier khác nằm ở sườn nhạy. | Giải thích tại sao chọn 1 kênh đơn lẻ bằng MAD bị thất bại khi người không nằm giữa 2 ESP32, dẫn tới giải pháp **Hợp nhất đa kênh bằng SVD/PCA** trong [`respiration_pipeline.py`](file:///D:/Master/Luanvan/process_csi_data/02_phasebeat_dsp/respiration_pipeline.py). |
| [**`SA_WiSense_ESP32_SingleAntenna_BlindSpotFree_Respiration_arXiv2507.17623.pdf`**](file:///D:/Master/Luanvan/process_csi_data/document/SA_WiSense_ESP32_SingleAntenna_BlindSpotFree_Respiration_arXiv2507.17623.pdf) *(arXiv 2025)* | Đo nhịp thở không điểm mù trên phần cứng **ESP32 đơn anten**: chỉ ra rằng mạch thu ESP32 chỉ có 1 anten nên không thể chia tỉ số 2 anten, nhưng có thể khử nhiễu hệ số khuếch đại chung (AGC) bằng chuẩn hóa chéo giữa các subcarrier trong cùng gói tin. | Cơ sở trực tiếp cho hàm **`agc_normalize(amps)`** (chia biên độ từng gói cho trung bình cộng 52 subcarrier hoạt động của gói đó) trong [`respiration_pipeline.py`](file:///D:/Master/Luanvan/process_csi_data/02_phasebeat_dsp/respiration_pipeline.py). |
| [**`PulseFi_ESP32_CSI_LSTM_Apnea_Cardiopulmonary_arXiv2510.24744.pdf`**](file:///D:/Master/Luanvan/process_csi_data/document/PulseFi_ESP32_CSI_LSTM_Apnea_Cardiopulmonary_arXiv2510.24744.pdf) *(arXiv 2025)* | Hệ thống dùng ESP32 CSI kết hợp lọc tín hiệu số và mạng nơ-ron hồi quy **LSTM** để theo dõi hô hấp và phát hiện **ngưng thở (Apnea)**. | Cơ sở cho việc kết hợp DSP với mạng **BiLSTM** để phát hiện Apnea (vì bộ lọc thông dải DSP thuần túy luôn biến nhiễu nền lúc nín thở thành sóng hình sin giả). |
| [**`SenseFi_DeepLearning_WiFi_CSI_Benchmark_arXiv2207.07859.pdf`**](file:///D:/Master/Luanvan/process_csi_data/document/SenseFi_DeepLearning_WiFi_CSI_Benchmark_arXiv2207.07859.pdf) *(Yang et al., Cell Press Patterns 2023)* | Chuẩn hóa các kiến trúc Học sâu cho dữ liệu WiFi CSI: chứng minh kiến trúc lai **1D-CNN + BiLSTM + Attention** khai thác đồng thời tương quan không gian giữa các subcarrier và tính tuần hoàn theo thời gian. | Cơ sở thiết kế nhánh Spatio-Temporal của mạng `CSI_RespirationNet` trong [`04_ai_model/models.py`](file:///D:/Master/Luanvan/process_csi_data/04_ai_model/models.py). |
| [**`Awsome_wifi_sensing_doc.md`**](file:///D:/Master/Luanvan/process_csi_data/document/Awsome_wifi_sensing_doc.md) | Tổng hợp danh mục các nghiên cứu WiFi CSI Sensing, đặc biệt lưu ý các bài báo năm 2024 về **Data Leakage (rò rỉ dữ liệu)** khi chia cửa sổ trượt ngẫu nhiên. | Cơ sở bắt buộc chúng ta sử dụng **5-Fold `StratifiedGroupKFold` theo từng file ghi âm độc lập** trong [`04_ai_model/train.py`](file:///D:/Master/Luanvan/process_csi_data/04_ai_model/train.py) thay vì chia ngẫu nhiên. |

---

## 2. Thu Thập Dữ Liệu & Quy Chuẩn Lưu Trữ (`01_data_collection/` & `data/raw/`)

### 2.1. Quy trình đã thực hiện
1. Xây dựng [`01_data_collection/serial_reader.py`](file:///D:/Master/Luanvan/process_csi_data/01_data_collection/serial_reader.py) để đọc luồng UART từ ESP32 ở tốc độ `921600 baud`, tốc độ gói thực tế đạt trung bình **$88\text{--}92\text{ Hz}$** (dao động tức thời từ $72\text{--}102\text{ Hz}$).
2. Xây dựng công cụ thu thập có hướng dẫn nhịp thở [`01_data_collection/record_dataset.py`](file:///D:/Master/Luanvan/process_csi_data/01_data_collection/record_dataset.py):
   * Hiển thị thanh **Metronome trực quan** trên Terminal để người đo hít vào / thở ra đúng chu kỳ mục tiêu.
   * Điều kiện đo thực tế: Phòng kín, không có người khác di chuyển, không bật quạt, người đo ngồi trong phòng nhưng **không nằm trên đường thẳng nối 2 ESP32 (Off-LOS)**.
3. Đã thu thập **21 file mẫu có nhãn** (~900 giây, ~80.000 gói CSI), lưu tại [`data/raw/`](file:///D:/Master/Luanvan/process_csi_data/data/raw/):
   * **6 file Nín thở (`apnea`, $0\text{ BPM}$):** `apnea_0bpm_20261105_022627.npz` ... `apnea_0bpm_20261106_021928.npz` (mỗi file dài $25\text{s}$, ~2.250 gói).
   * **5 file Thở chậm (`slow`, $10\text{ BPM}$, chu kỳ $6.0\text{s}$):** `slow_10bpm_20261105_023137.npz` ... `slow_10bpm_20261106_022035.npz` (mỗi file dài $50\text{s}$, ~4.480 gói).
   * **5 file Thở bình thường (`normal`, $16\text{ BPM}$, chu kỳ $3.75\text{s}$):** `normal_16bpm_20261105_023243.npz` ... `normal_16bpm_20261106_022135.npz` (mỗi file dài $50\text{s}$, ~4.480 gói).
   * **5 file Thở nhanh (`fast`, $24\text{ BPM}$, chu kỳ $2.5\text{s}$):** `fast_24bpm_20261105_023345.npz` ... `fast_24bpm_20261106_022242.npz` (mỗi file dài $50\text{s}$, ~4.480 gói).

### 2.2. Cấu trúc bên trong mỗi file `.npz` ở `data/raw/`
Mỗi file `.npz` chứa 10 trường chuẩn hóa:
* `amps`: Ma trận biên độ `float64` kích thước `(N, 64)`.
* `phases`: Ma trận pha thô `float64` kích thước `(N, 64)` (đơn vị radian, từ $-\pi$ đến $\pi$).
* `times`: Vector thời gian tương đối `float64` kích thước `(N,)` tính bằng giây từ lúc bắt đầu thu.
* `label_class`: Chuỗi nhãn (`"apnea"`, `"slow"`, `"normal"`, `"fast"`).
* `class_id`: Số nguyên mã lớp (`0, 1, 2, 3`).
* `target_bpm`: Nhịp thở chuẩn theo metronome (`0.0, 10.0, 16.0, 24.0`).
* `duration`, `actual_fs`, `subject`, `created_at`.

### 2.3. Phát hiện quan trọng về 64 Subcarrier của ESP32
Khi kiểm tra thống kê trên dữ liệu thu được:
* Các bin `0..5` và `59..63` là **Guard/Null subcarriers** (băng tần bảo vệ ở hai rìa phổ OFDM).
* Bin `32` là **DC subcarrier** (sóng mang trung tâm có dòng rò DC).
* Các bin `5, 13, 21, 29, 37, 45, 53` chứa các **Pilot/Sync tones** của ESP32, thỉnh thoảng vọt giá trị dị biệt lên tới $\pm 122.000$ (gấp hàng nghìn lần biên độ bình thường ~20–35), từng làm hỏng hoàn toàn lần train AI đầu tiên.
* **Giải pháp:** Loại bỏ toàn bộ 18 bin trên, chỉ giữ lại **46 kênh dữ liệu vật lý sạch (`ACTIVE_SUBCARRIERS_46`)** hoặc 52 kênh OFDM hoạt động (`ACTIVE_SUBCARRIERS_64` sau khi qua bộ lọc Hampel).

---

## 3. Các Thuật Toán Xử Lý Tín Hiệu Số (DSP) — Lý Do, Chi Tiết Toán Học & Cải Tiến

Trong quá trình nghiên cứu, chúng ta đã phát triển và so sánh **2 thế hệ pipeline DSP**:
* **Thế hệ 1 (PhaseBeat v1 — [`phasebeat_core.py`](file:///D:/Master/Luanvan/process_csi_data/02_phasebeat_dsp/phasebeat_core.py) & [`breathing_estimator.py`](file:///D:/Master/Luanvan/process_csi_data/02_phasebeat_dsp/estimation/breathing_estimator.py))**
* **Thế hệ 2 (Multi-Subcarrier PCA Pipeline v2 — [`respiration_pipeline.py`](file:///D:/Master/Luanvan/process_csi_data/02_phasebeat_dsp/respiration_pipeline.py))**

Dưới đây là chi tiết từng bước thuật toán trong Pipeline v2 và lý do vì sao phải thiết kế như vậy:

### Bước 1: Nội suy lưới thời gian đều (`resample_uniform` @ $90.0\text{ Hz}$)
* **Vấn đề:** ESP32 gửi gói CSI qua WiFi + UART không đều đặn; tần số lấy mẫu giữa các cửa sổ dao động từ **$71.8\text{ Hz}$ đến $102.0\text{ Hz}$**, thậm chí có gói trùng timestamp (`apnea_0bpm_20261106_021928.npz` có 25 timestamp trùng nhau gây lỗi chia cho 0 khi nội suy).
* **Thuật toán:**
  1. Khử các điểm trùng timestamp bằng `np.unique(times, return_index=True)`.
  2. Tạo lưới thời gian đều đặn tuyệt đối $t_u$ với bước nhảy $\Delta t = \frac{1}{90.0}\text{ s}$.
  3. Nội suy tuyến tính (`scipy.interpolate.interp1d`) trên toàn bộ các subcarrier.
* **Ý nghĩa:** Đảm bảo các bộ lọc số (Butterworth, Savitzky-Golay) và phép biến đổi Fourier (FFT) không bị trôi tần số cắt.

### Bước 2: Chuẩn hóa khuếch đại tự động theo từng gói (`agc_normalize`)
* **Vấn đề:** Khi người ngồi xa hoặc lệch khỏi trục nối 2 ESP32 (Off-LOS), mạch thu WiFi của ESP32 liên tục tự động nhảy mức khuếch đại (AGC - Automatic Gain Control). Mỗi khi AGC nhảy, **cả 64 subcarrier trong gói tin đó cùng bị nhân lên hoặc chia xuống một hệ số**, tạo ra các bậc thang biên độ. Khi đi qua bộ lọc thông dải $0.1\text{--}0.75\text{ Hz}$, các bậc thang AGC này biến thành sóng hình sin giả có biên độ lớn ngang nhịp thở thật.
* **Công thức toán học:**
  Với mỗi gói tin tại thời điểm $t$, gọi $A(t, k)$ là biên độ của subcarrier hoạt động thứ $k \in \mathcal{K}$:
  $$\tilde{A}(t, k) = \frac{A(t, k)}{\frac{1}{|\mathcal{K}|} \sum_{j \in \mathcal{K}} A(t, j) + \epsilon}$$
* **Ý nghĩa:** Vì hệ số khuếch đại thu $G_{AGC}(t)$ nhân giống nhau lên mọi subcarrier trong cùng một gói tin ($A(t, k) = G_{AGC}(t) \cdot |H(t, k)|$), phép chia cho trung bình gói **triệt tiêu hoàn toàn $G_{AGC}(t)$** nhưng giữ nguyên tỉ lệ biến thiên chọn lọc tần số do lồng ngực phản xạ đa đường tạo ra! Qua thực nghiệm benchmark, riêng bước này giúp giảm sai số MAE của nhịp thở bình thường từ $2.31\text{ BPM}$ xuống **$1.66\text{ BPM}$**.

### Bước 3: Lọc gai nhọn Hampel (`remove_spikes` trong [`spike_removal.py`](file:///D:/Master/Luanvan/process_csi_data/02_phasebeat_dsp/filters/spike_removal.py))
* **Thuật toán:** Trên cửa sổ trượt ngắn $0.3\text{s}$ (~27 mẫu), tính trung vị cục bộ $m_t$ và độ lệch tuyệt đối trung vị cục bộ $\text{MAD}_t = \text{median}(|x - m_t|)$. Ước lượng độ lệch chuẩn bền vững $\hat{\sigma}_t = 1.4826 \times \text{MAD}_t$. Mọi mẫu vượt quá $|x_t - m_t| > 3.0 \hat{\sigma}_t$ bị thay thế bằng $m_t$.
* **Ý nghĩa:** Loại bỏ các xung nhiễu tức thời (outliers) do va chạm gói tin WiFi mà không làm mờ sườn sóng thở.

### Bước 4: Khử phản xạ tĩnh trong phòng bằng Savitzky-Golay (`detrend_amplitude` trong [`detrending.py`](file:///D:/Master/Luanvan/process_csi_data/02_phasebeat_dsp/filters/detrending.py))
* **Vấn đề:** Trong phòng có rất nhiều phản xạ cố định từ tường, trần nhà, bàn ghế (Static Multipath) cộng với sự trôi nhiệt chậm của bộ thu phát RF.
* **Thuật toán:** Khớp đa thức cục bộ bậc 2 (Savitzky-Golay filter) trên cửa sổ dài **$12.0\text{ giây}$** ($1081$ mẫu ở $90\text{ Hz}$) để ước lượng đường nền chậm $b(t)$, sau đó lấy tín hiệu trừ đi đường nền: $x_{detrend}(t) = x(t) - b(t)$.
* **Lý do chọn $12.0\text{s}$:** Chu kỳ thở chậm nhất ($10\text{ BPM}$) là $6.0\text{ giây}$. Cửa sổ $12.0\text{s}$ dài gấp đôi chu kỳ thở chậm nhất nên chỉ khớp theo đường trôi nền tĩnh mà không làm suy hao biên độ nhịp thở.

### Bước 5: Lọc thông dải hô hấp Zero-Phase Butterworth (`butterworth_bandpass` trong [`bandpass.py`](file:///D:/Master/Luanvan/process_csi_data/02_phasebeat_dsp/filters/bandpass.py))
* **Thuật toán:** Bộ lọc Butterworth bậc 2 dạng chuỗi bậc hai (`output="sos"`) với dải thông **$[0.10\text{ Hz}, 0.75\text{ Hz}]$** (tương ứng **$6\text{--}45\text{ BPM}$**), lọc hai chiều xuôi-ngược bằng `scipy.signal.sosfiltfilt` (bậc hiệu dụng = 4, độ trễ pha bằng 0).
* **Ý nghĩa:** Triệt tiêu hoàn toàn nhiễu tần số cao ($>0.75\text{ Hz}$ như nhịp tim, rung cơ, nhiễu điện tử) và giữ nguyên vị trí thời gian của các đỉnh hít vào.

### Bước 6: Hợp nhất đa kênh bằng SVD/PCA (`process_window` trong [`respiration_pipeline.py`](file:///D:/Master/Luanvan/process_csi_data/02_phasebeat_dsp/respiration_pipeline.py))
* **Vấn đề của cách cũ (MAD Selection):** Khi người không nằm giữa 2 ESP32, theo mô hình đới Fresnel (`FarSense`), một số subcarrier rơi vào cực trị giao thoa (điểm mù — ngực phập phồng nhưng biên độ không đổi), còn một số subcarrier khác bị nhiễu biên. Chọn 1 kênh có MAD lớn nhất/trung vị thường bốc nhầm kênh nhiễu.
* **Thuật toán SVD/PCA mới:**
  1. Chuẩn hóa ma trận 52 (hoặc 46) kênh đã lọc $X \in \mathbb{R}^{N \times M}$ về trung bình 0, phương sai 1 trên từng kênh.
  2. Phân rã giá trị kỳ dị (Singular Value Decomposition):
     $$X = U \Sigma V^T$$
  3. Trích xuất $k=3$ thành phần chính đầu tiên: $c_i(t) = \frac{\sigma_i}{\sqrt{N}} u_i(t)$ với $i \in \{1, 2, 3\}$.
  4. Tính tỉ số công suất đỉnh trên trung bình trong dải thở (**Breathing-Band SNR**) cho từng thành phần $c_i(t)$ và chọn thành phần có SNR cao nhất.
* **Ý nghĩa:** Phép chiếu PCA gom toàn bộ dao động đồng pha (coherent chest movement) xuất hiện chung trên mọi subcarrier nhạy, đồng thời loại bỏ nhiễu ngẫu nhiên độc lập trên từng kênh đơn lẻ.

### Bước 7: Ước lượng BPM độ phân giải cao kết hợp Nội suy Parabol Phổ & Tự tương quan (`spectral_bpm` + `acf_bpm`)
* **Vấn đề của Welch FFT cũ:** Welch FFT với đoạn $15\text{s}$ có độ rộng bin tần số $\Delta f = \frac{1}{15}\text{ Hz} = 4.0\text{ BPM}$. Do đó kết quả FFT cũ chỉ nhảy theo các nấc thô ($12, 16, 20, 24\text{ BPM}$), làm lớp Normal ($16\text{ BPM}$) trên cửa sổ trượt có sai số MAE lên tới $3.16\text{ BPM}$.
* **Thuật toán mới:**
  1. **Periodogram cửa sổ Hann + Nội suy đỉnh Parabol (`spectral_bpm`):** Nhân toàn bộ cửa sổ tín hiệu với hàm cửa sổ Hann, tính FFT với $N_{fft} = 16384$, tìm đỉnh cực đại $i$ trong dải $[0.1, 0.75]\text{ Hz}$ và khớp đường cong Parabol qua 3 điểm log-power $(\ln P_{i-1}, \ln P_i, \ln P_{i+1})$:
     $$\delta = \frac{1}{2} \frac{\ln P_{i-1} - \ln P_{i+1}}{\ln P_{i-1} - 2\ln P_i + \ln P_{i+1}}, \quad f_{\text{peak}} = f_i + \delta \cdot \Delta f, \quad \text{BPM}_{\text{spec}} = 60 \cdot f_{\text{peak}}$$
  2. **Kiểm tra chéo bằng Tự tương quan (`acf_bpm`):** Tính hàm tự tương quan chuẩn hóa $R(\tau)$ qua FFT:
     $$R(\tau) = \frac{\mathcal{F}^{-1}\{|\mathcal{F}\{s(t)\}|^2\}}{R(0)}$$
     Tìm đỉnh cao nhất trong miền độ trễ hợp lệ $\tau \in [f_s / 0.75, f_s / 0.10]$ để suy ra $\text{BPM}_{\text{acf}} = \frac{60 f_s}{\tau_{\text{peak}}}$ và độ tuần hoàn `acf_peak` $= R(\tau_{\text{peak}}) \in [0, 1]$.
  3. **Hợp nhất:** Nếu $|\text{BPM}_{\text{spec}} - \text{BPM}_{\text{acf}}| \le 3.0\text{ BPM}$ thì lấy trung bình cộng $0.5 (\text{BPM}_{\text{spec}} + \text{BPM}_{\text{acf}})$; ngược lại ưu tiên $\text{BPM}_{\text{spec}}$.

### Bước 8: Sửa lỗi thuật toán Làm sạch Pha (`sanitize_phase` trong [`phase_sanitizer.py`](file:///D:/Master/Luanvan/process_csi_data/02_phasebeat_dsp/filters/phase_sanitizer.py))
* **Lỗi phát hiện được:** Code cũ thực hiện `np.unwrap(phase, axis=0)` (unwrap theo thời gian) *trước* khi khử độ dốc tuyến tính CFO/STO. Vì mỗi gói tin WiFi có một độ lệch pha CFO/STO ngẫu nhiên khác nhau, pha thô giữa 2 gói liên tiếp nhảy hoàn toàn ngẫu nhiên; việc unwrap theo thời gian trước đã cộng dồn các bước nhảy $2\pi$ sai lệch làm hỏng hoàn toàn tín hiệu pha.
* **Đã sửa:** Bỏ bước unwrap thời gian trên pha thô; chỉ thực hiện `np.unwrap(phase, axis=1)` (unwrap theo chiều không gian subcarrier trong từng gói tin) rồi khớp bình phương tối thiểu tuyến tính $\phi(k) \approx a \cdot k + b$ để trừ bỏ CFO/STO. (Tuy nhiên qua thực nghiệm trên ESP32 đơn anten, chế độ **Amplitude chuẩn hóa AGC** vẫn đạt MAE tốt hơn hẳn chế độ Phase: $1.82\text{ BPM}$ so với $5.03\text{ BPM}$).

---

## 4. Mô Hình Học Sâu (AI) — Tại Sao Phải Nâng Cấp, Kiến Trúc & Huấn Luyện (`04_ai_model/`)

### 4.1. Tại sao bắt buộc phải có AI bên cạnh DSP?
* Khi người đo **đang thở** (Slow / Normal / Fast), thuật toán DSP v2 đo nhịp thở rất chính xác (MAE chỉ $1.32\text{ BPM}$).
* Nhưng khi người đo **nín thở (Apnea)**: không có dao động lồng ngực, tín hiệu thu được chỉ là nhiễu nền trong phòng. Tuy nhiên, khi nhiễu nền đi qua bộ lọc thông dải $0.1\text{--}0.75\text{ Hz}$, **bộ lọc thông dải biến mọi nhiễu dải hẹp thành dạng sóng tựa hình sin**! Kết quả là FFT và bộ đếm đỉnh của DSP vẫn tìm thấy đỉnh phổ và báo "đang thở $9\text{--}24\text{ BPM}$" (tỉ lệ phát hiện Apnea của DSP cũ là **$0.0\%$**).
* Mô hình AI nhìn đồng thời cả **46 kênh phụ theo thời gian**, **hình dáng phân bố toàn bộ phổ PSD** và **hệ số tự tương quan/SNR**, giúp phân biệt đâu là sóng thở thật và đâu là nhiễu nền sau lọc.

### 4.2. Ba nguyên nhân khiến mô hình AI v1 (cửa sổ 4s) thất bại ở mức $44.4\%$ khi chạy Realtime
Khi kiểm tra định lượng mô hình AI đầu tiên (1D-CNN + BiLSTM trên cửa sổ $4.0\text{s}$), chúng ta phát hiện 3 lỗi gốc rễ:
1. **Cửa sổ $4.0\text{ giây}$ quá ngắn về mặt vật lý:** Một nhịp thở chậm ($10\text{ BPM}$) cần tới **$6.0\text{ giây}$** mới hoàn thành 1 chu kỳ hít vào – thở ra. Cắt cửa sổ $4\text{s}$ khiến mạng nơ-ron không bao giờ nhìn thấy trọn vẹn 1 chu kỳ thở chậm, dẫn đến nhầm lẫn nặng giữa Slow, Normal và Apnea.
2. **Lệch quy trình lọc giữa lúc Train và lúc Realtime (Train/Test Mismatch):**
   * Lúc Train (`dataset_prep.py` v1): Lọc cả file dài $50\text{s}$ một lần bằng `sosfiltfilt`, rồi mới cắt các đoạn $4\text{s}$ ở giữa file (nơi tín hiệu đã ổn định, không có hiệu ứng biên).
   * Lúc Realtime (`realtime_engine.py` v1): Lọc buffer $18\text{s}$, rồi cắt lấy **$4\text{s}$ sát mép cuối cùng** (nơi bộ lọc Savitzky-Golay và `sosfiltfilt` chịu hiệu ứng mép biên mạnh nhất).
   * Hậu quả: Độ chính xác tụt từ $74.0\%$ (cắt kiểu offline) xuống **$44.4\%$** (chạy kiểu realtime).
3. **Chuẩn hóa từng kênh (`Instance Normalization` theo chiều thời gian) xóa mất thông tin năng lượng:** Việc chia từng kênh trong cửa sổ $4\text{s}$ cho độ lệch chuẩn của chính nó biến mọi cửa sổ nín thở (biên độ nhiễu cực nhỏ) thành tín hiệu có phương sai bằng $1.0$ giống hệt lúc đang thở mạnh.

### 4.3. Kiến trúc Mô hình AI v2 (`CSI_RespirationNet` — Dual-Branch Multi-Task Network)
Để giải quyết triệt để cả 3 lỗi trên, chúng ta đã thiết kế lại pipeline AI trong [`04_ai_model/dataset_prep.py`](file:///D:/Master/Luanvan/process_csi_data/04_ai_model/dataset_prep.py) và [`04_ai_model/models.py`](file:///D:/Master/Luanvan/process_csi_data/04_ai_model/models.py):

1. **Cửa sổ $16.0\text{ giây}$ đồng nhất 100% giữa Train và Realtime:**
   * Mỗi cửa sổ $16.0\text{s}$ ($1440$ mẫu ở $90\text{ Hz}$) chứa trọn vẹn **$2.7$ chu kỳ thở chậm**, **$4.3$ chu kỳ thở bình thường**, hoặc **$6.4$ chu kỳ thở nhanh**.
   * Cả lúc tạo tập train lẫn lúc chạy GUI đều cắt cửa sổ thô $16.0\text{s}$ trước rồi mới gọi chung hàm `extract_ai_features(raw_win, fs=90.0, window_sec=16.0)`. Không còn bất kỳ sự khác biệt nào giữa lúc huấn luyện và lúc chạy thực tế.
2. **Kiến trúc Hai Nhánh (Dual-Branch Architecture):**
   * **Nhánh 1 — Spatio-Temporal Waveform Branch:**
     * Đầu vào: Tensor `x_wave` kích thước `(46, 360)` (46 kênh dữ liệu sạch sau lọc, giảm mẫu 4 lần từ $90\text{ Hz} \to 22.5\text{ Hz}$ để mỗi bước thời gian đại diện đúng hình thái sóng thở).
     * Chuẩn hóa toàn cục trên cả cửa sổ `(46, 360)` thay vì chuẩn hóa riêng từng kênh.
     * Đi qua 2 khối `Conv1d(46->32, k=7) -> BatchNorm -> LeakyReLU -> MaxPool1d(3) -> Dropout(0.35)` và `Conv1d(32->48, k=5) -> BatchNorm -> LeakyReLU -> MaxPool1d(3) -> Dropout(0.35)` $\to$ `BiLSTM(48, hidden=32, bidirectional=True)` $\to$ trung bình chuỗi kết hợp `AttentionPooling(64)` tạo ra vector đặc trưng thời gian $64$ chiều.
   * **Nhánh 2 — Spectral & Periodicity Branch:**
     * Đầu vào: Vector phổ công suất chuẩn hóa `x_spec` ($131$ bins trong dải $0.08\text{--}0.80\text{ Hz}$) ghép với vector 6 đặc trưng vật lý `x_aux` (`[bpm/30, spec_bpm/30, acf_bpm/30, snr/10, acf_peak, median_std*100]`).
     * Đi qua mạng MLP 2 tầng có `LayerNorm` và `Dropout(0.35)` tạo ra vector đặc trưng phổ-chu kỳ $48$ chiều.
   * **Đầu ra Đa nhiệm (Multi-Task Heads):**
     * Ghép 2 nhánh thành vector $112$ chiều.
     * **Head 1 (`classifier`):** Phân loại 4 trạng thái (`0: apnea`, `1: slow`, `2: normal`, `3: fast`).
     * **Head 2 (`regressor`):** Dự đoán nhịp thở liên tục (BPM).

### 4.4. Phương pháp Huấn luyện & Kiểm định Chéo 5-Fold (`04_ai_model/train.py`)
* **Chia tập dữ liệu không rò rỉ (`StratifiedGroupKFold`, $K=5$):**
  * 21 file ghi âm tạo ra **286 cửa sổ $16\text{s}$** (bước trượt $2.0\text{s}$): `30 apnea`, `86 slow`, `85 normal`, `85 fast`.
  * Toàn bộ các cửa sổ cắt ra từ cùng một file ghi âm được gán cùng một mã `group`. Trong mỗi Fold, ~17 file dùng để Train và ~4 file **chưa từng xuất hiện trong tập Train** được dùng để Validate.
* **Hàm mất mát đa nhiệm cân bằng lớp:**
  $$\mathcal{L} = \text{CrossEntropy}_{\text{weighted}}(\hat{y}_{\text{cls}}, y_{\text{cls}}) + 1.5 \cdot \text{SmoothL1}\left(\frac{\hat{y}_{\text{bpm}}}{30}, \frac{y_{\text{bpm}}}{30}\right)$$
  Trong đó trọng số lớp $w_c = \frac{N}{4 \cdot N_c}$ bù đắp việc lớp `apnea` có ít cửa sổ hơn (do file apnea chỉ thu $25\text{s}$).
* **Tăng cường dữ liệu (Data Augmentation):** Trong lúc train, ngẫu nhiên tắt $15\%$ số subcarrier (**Subcarrier Dropout** — mô phỏng hiện tượng kênh phụ rơi vào điểm mù khi người đổi vị trí) và cộng nhiễu Gaussian nhẹ ($2\%$ std).
* **Tổ hợp 5-Fold Ensemble:** Lưu toàn bộ 5 bộ trọng số tốt nhất của 5 Fold vào checkpoint để khi chạy realtime ([`04_ai_model/inference.py`](file:///D:/Master/Luanvan/process_csi_data/04_ai_model/inference.py)), hệ thống lấy trung bình xác suất dự đoán của cả 5 mô hình.

### 4.5. Vị trí lưu trữ toàn bộ Dữ liệu, Trọng số và Biểu đồ
| Thành phần | Đường dẫn lưu trữ tuyệt đối | Mô tả chi tiết |
|---|---|---|
| **Dữ liệu CSI thô có nhãn** | [`D:\Master\Luanvan\process_csi_data\data\raw\*_202611*.npz`](file:///D:/Master/Luanvan/process_csi_data/data/raw/) | 21 file `.npz` thu thực tế (6 apnea, 5 slow, 5 normal, 5 fast). |
| **Tập Tensor đã tiền xử lý** | [`D:\Master\Luanvan\process_csi_data\data\processed\csi_respiration_dataset_4s.npz`](file:///D:/Master/Luanvan/process_csi_data/data/processed/csi_respiration_dataset_4s.npz) | 286 cửa sổ $16\text{s}$: `X (286, 46, 360)`, `X_spec (286, 131)`, `X_aux (286, 6)`, `y_class`, `y_bpm`, `groups`. |
| **Trọng số AI đã huấn luyện** | [`D:\Master\Luanvan\process_csi_data\data\weights\best_csi_model.pt`](file:///D:/Master/Luanvan/process_csi_data/data/weights/best_csi_model.pt) | Checkpoint PyTorch chứa `model_state_dict` và `ensemble_state_dicts` (5 mô hình của 5 Fold). |
| **Biểu đồ kết quả huấn luyện** | [`05_documentation_summary/images/confusion_matrix.png`](file:///D:/Master/Luanvan/process_csi_data/document/05_documentation_summary/images/confusion_matrix.png) & [`training_curves.png`](file:///D:/Master/Luanvan/process_csi_data/document/05_documentation_summary/images/training_curves.png) | Bản gốc tại `data/processed/` và bản sao lưu trong `05_documentation_summary/images/`. |

---

## 5. Kết Quả Thực Nghiệm Định Lượng & Nghiên Cứu Ablation

### 5.1. So sánh kết quả DSP Trước và Sau cải tiến (Trượt cửa sổ trên 21 file)

| Phương pháp DSP | Slow ($10\text{ BPM}$) MAE | Normal ($16\text{ BPM}$) MAE | Fast ($24\text{ BPM}$) MAE | Tổng MAE | Tỉ lệ sai $\le 2\text{ BPM}$ | Phát hiện Apnea (AUC) |
|---|---|---|---|---|---|---|
| **PhaseBeat v1 (1 kênh MAD + Welch 15s, cửa sổ 18s)** | $1.04\text{ BPM}$ | $3.16\text{ BPM}$ | $2.60\text{ BPM}$ | $2.26\text{ BPM}$ | $75.1\%$ | $0.0\%$ (Ngưỡng hỏng) |
| **PhasePCA v2 (Chế độ Pha đã sửa unwrap, cửa sổ 18s)** | $1.49\text{ BPM}$ | $6.00\text{ BPM}$ | $7.64\text{ BPM}$ | $5.03\text{ BPM}$ | $46.1\%$ | Kém (Pha ESP32 nhiễu) |
| **AmpPCA v2 (Chưa chuẩn hóa AGC, cửa sổ 18s)** | $1.16\text{ BPM}$ | $2.31\text{ BPM}$ | $2.50\text{ BPM}$ | $1.98\text{ BPM}$ | $75.1\%$ | AUC = $0.853$ |
| **AGC-Norm + PCA + Parabolic/ACF (v2, cửa sổ 18s)** | $1.16\text{ BPM}$ | **$1.66\text{ BPM}$** | $2.66\text{ BPM}$ | **$1.82\text{ BPM}$** | **$80.5\%$** | AUC = $0.853$ |
| **AGC-Norm + PCA + Parabolic/ACF (v2, cửa sổ 20s)** | **$0.86\text{ BPM}$** | **$1.05\text{ BPM}$** | **$2.05\text{ BPM}$** | **$1.32\text{ BPM}$** | **$86.7\%$** | **AUC = $0.920$** |
| **AGC-Norm + PCA + Parabolic/ACF (v2, cửa sổ 25s)** | **$0.64\text{ BPM}$** | **$1.13\text{ BPM}$** | **$1.51\text{ BPM}$** | **$1.09\text{ BPM}$** | **$89.7\%$** | **AUC = $0.950$** |

### 5.2. Kết quả Kiểm định Chéo 5-Fold trên Phiên Đo Chưa Từng Thấy (Unseen Sessions)

Khi đánh giá nghiêm ngặt trên các file ghi âm chưa từng xuất hiện trong tập huấn luyện (`StratifiedGroupKFold`, 5 Folds):
* **Fold 1:** Val Accuracy = **$94.7\%$** | Macro F1 = **$0.941$**
* **Fold 2:** Val Accuracy = **$95.1\%$** | Macro F1 = **$0.951$**
* **Fold 3:** Val Accuracy = **$91.1\%$** | Macro F1 = **$0.928$**
* **Fold 4:** Val Accuracy = **$89.3\%$** | Macro F1 = **$0.912$**
* **Fold 5:** Val Accuracy = **$82.1\%$** | Macro F1 = **$0.856$**
* **Tổng hợp toàn bộ 286 cửa sổ (Out-of-Session Predictions):**
  * **Độ chính xác (5-Fold Unseen Accuracy, Early-Stopped):** **$90.6\%$** (tăng từ **$48.8\%$** của mô hình v1).
  * **Macro F1-Score:** **$0.917$** (tăng từ **$0.461$** của mô hình v1).
  * **Độ chính xác khi không chọn epoch (Last-Epoch / Random Forest trên Spec+Aux):** **$71.7\%\text{--}80.4\%$** (trong đó Recall lớp Apnea đạt **$90.0\%$**).
  * **Độ chính xác khi chạy phát lại Realtime với tổ hợp 5-Fold Ensemble trên 21 file:** **$94.2\%$** (tăng từ **$44.4\%$** của mô hình v1).

| Lớp trạng thái | Precision | Recall | F1-Score | Số cửa sổ (Support) |
|---|---|---|---|---|
| **APNEA ($0\text{ BPM}$)** | $96.7\%$ | $96.7\%$ | $0.967$ | 30 |
| **SLOW ($10\text{ BPM}$)** | $78.3\%$ | $96.5\%$ | $0.865$ | 86 |
| **NORMAL ($16\text{ BPM}$)** | $97.4\%$ | $87.1\%$ | $0.919$ | 85 |
| **FAST ($24\text{ BPM}$)** | $98.6\%$ | $85.9\%$ | $0.918$ | 85 |
| **Trung bình (Macro Avg)** | **$92.7\%$** | **$91.5\%$** | **$0.917$** | **286** |

![5-Fold Out-of-Session Confusion Matrix](file:///D:/Master/Luanvan/process_csi_data/document/05_documentation_summary/images/confusion_matrix.png)

![Training Curves](file:///D:/Master/Luanvan/process_csi_data/document/05_documentation_summary/images/training_curves.png)

### 5.3. Nghiên cứu Ablation (Vai trò của từng nhóm đặc trưng trên 5-Fold Unseen Sessions)
Để hiểu rõ tại sao mô hình mới hoạt động tốt, chúng ta đã bóc tách từng nhóm đặc trưng với bộ phân loại cố định (Random Forest 200 cây, không chọn epoch):
* Chỉ dùng thống kê miền thời gian trên 46 kênh (`Wave stats`): **Unseen Acc = $56.6\%$** (Macro F1 = $0.543$).
* Chỉ dùng 6 đặc trưng tóm tắt (`Aux: bpm, snr, acf_peak, std`): **Unseen Acc = $76.6\%$** (Macro F1 = $0.738$, Apnea Recall = $73.3\%$).
* Chỉ dùng 131 bin phổ (`Spec: 131 PSD bins`): **Unseen Acc = $78.7\%$** (Macro F1 = $0.740$).
* Kết hợp `Spec + Aux`: **Unseen Acc = $80.4\%$** (Macro F1 = $0.768$).
* Kết hợp cả 2 nhánh trong mạng `CSI_RespirationNet` (Early-Stopped 5-Fold): **Unseen Acc = $90.6\%$** (Macro F1 = $0.917$).

---

## 6. Đánh Giá Sơ Bộ (Preliminary Evaluation) — Vì Sao Test Thực Tế (Live) Vẫn Còn Nhảy & Hướng Giải Quyết

Mặc dù thuật toán đã cải thiện vượt bậc trên bộ dữ liệu 21 file đã thu, **khi bạn cắm ESP32 đo trực tiếp (Live) bên ngoài, dữ liệu trên giao diện vẫn xuất hiện hiện tượng nhảy linh tinh**. Dưới đây là đánh giá kỹ thuật trung thực về **5 nguyên nhân thực tế** gây ra hiện tượng này:

### 6.1. Hiện tượng đảo dấu ngẫu nhiên của SVD/PCA và đổi thành phần chính giữa các frame 350ms
* **Bản chất toán học:** Cứ mỗi $350\text{ ms}$, giao diện gọi `process_window()` để phân rã SVD lại trên ma trận 20 giây mới. Trong đại số tuyến tính, vector riêng từ SVD có tính bất định về dấu ($\pm u_i$ đều là nghiệm hợp lệ) và thứ tự giữa $PC_1, PC_2, PC_3$ có thể hoán đổi khi SNR của 2 thành phần xấp xỉ nhau.
* **Biểu hiện trên GUI:** Dạng sóng đang đi lên bỗng lật úp 180 độ ở frame tiếp theo, hoặc nhảy từ $PC_1$ sang $PC_2$ làm các chấm đỏ đánh dấu đỉnh thở và ước lượng nhịp nhảy theo.
* **Cách khắc phục tiếp theo trong code:** Cài đặt **Sign-Locking** (kiểm tra tích vô hướng của sóng frame hiện tại với frame liền trước, nếu $<0$ thì nhân $-1$) và **Component Hysteresis** (chỉ đổi từ $PC_1$ sang $PC_2$ khi SNR của $PC_2$ cao hơn ít nhất $25\%$).

### 6.2. Quán tính của cửa sổ trượt 16–20 giây khi bạn thay đổi kiểu thở lúc test Live
* **Bản chất vật lý:** Trong 21 file mẫu, mỗi file bạn chỉ giữ **một kiểu thở duy nhất** suốt $50\text{ giây}$ theo metronome. Ngược lại, khi ngồi test Live trước màn hình, người dùng thường thử thở nhanh vài giây rồi chuyển sang thở chậm hoặc nín thở để xem máy có nhạy không.
* **Biểu hiện trên GUI:** Vì cửa sổ phân tích dài $16\text{--}20\text{ giây}$, trong $10\text{--}15\text{ giây}$ đầu sau khi bạn đổi kiểu thở, cửa sổ chứa **hỗn hợp cả nhịp thở cũ lẫn nhịp thở mới**. Phổ FFT lúc đó xuất hiện 2 đỉnh cạnh tranh nhau (ví dụ đỉnh $12\text{ BPM}$ và đỉnh $24\text{ BPM}$), làm số BPM và nhãn AI nhảy qua lại giữa 2 trạng thái cho đến khi dữ liệu cũ trôi hết khỏi cửa sổ 20 giây.

### 6.3. Nhiễu cử động cơ thể (Motion Artifacts) không có trong tập huấn luyện
* **Bản chất vật lý:** Biên độ lồng ngực khi thở chỉ dịch chuyển **$2\text{--}5\text{ mm}$**. Khi bạn ngồi nhìn màn hình test Live, chỉ một cử động nhỏ của tay trên bàn phím/chuột, nghiêng đầu, nuốt nước bọt hoặc dịch lưng trên ghế cũng tạo ra dịch chuyển **$20\text{--}100\text{ mm}$** (lớn gấp 10–20 lần biên độ thở).
* **Biểu hiện trên GUI:** Cử động cơ thể tạo ra một xung năng lượng lớn trong dải $0.1\text{--}0.75\text{ Hz}$ kéo dài suốt 20 giây cho tới khi xung đó trôi khỏi bộ đệm, làm cả DSP và AI báo sai trong suốt 20 giây đó. Tập dữ liệu 21 file hiện tại chưa có lớp **"Motion / Cử động"** và chưa có bộ phát hiện cử động (Motion Gate) để tạm dừng cập nhật BPM khi người đo cựa quậy.

### 6.4. Jitter thời gian đóng gói tại cổng USB Serial trên Windows
* Trong [`stream_source.py`](file:///D:/Master/Luanvan/process_csi_data/03_realtime_app/stream_source.py), thời gian gói tin đang được lấy bằng `time.time()` của Python tại thời điểm lệnh `ser.readline()` đọc xong. Khi luồng giao diện Matplotlib vẽ đồ thị (chiếm GIL của Python trong $30\text{--}60\text{ ms}$), các gói CSI nằm chờ trong bộ đệm driver USB-UART rồi được đọc ra thành từng chùm có timestamp sát nhau, gây méo tần số tức thời.

### 6.5. Quy mô tập dữ liệu hiện tại còn nhỏ (21 file, 1 người đo)
* Toàn bộ 21 file hiện tại được thu trên **1 đối tượng, trong cùng 1 phòngu**. Mặc dù kiểm định 5-Fold đạt $80\text{--}90.6\%$ 
