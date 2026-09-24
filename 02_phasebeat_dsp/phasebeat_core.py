import numpy as np
from scipy.signal import butter, filtfilt, medfilt, detrend
from sklearn.decomposition import PCA

class PhaseBeatCore:
    def __init__(self, fs=75.5, lowcut=0.1, highcut=0.6, order=3, top_k=5):
        self.fs = fs
        self.top_k = top_k
        nyq = 0.5 * self.fs
        low = max(lowcut / nyq, 1e-4)
        high = min(highcut / nyq, 0.99)
        self.b, self.a = butter(order, [low, high], btype='band')

    def pbCore_removeOutliers(self, data):
        cleaned = np.zeros_like(data)
        for i in range(data.shape[1]):
            cleaned[:, i] = medfilt(data[:, i], kernel_size=15)
        return cleaned

    def pbCore_process(self, amps):
        # 1. Gọt gai nhiễu bằng Median Filter
        cleaned = self.pbCore_removeOutliers(amps)
        
        # 2. ÉP XUẤT PHÁT ĐIỂM VỀ 0 (CHỐNG SỐC BỘ LỌC)
        # Tính baseline từ 1 giây đầu tiên để ghim điểm t=0 về đúng trục Y=0
        warmup_pts = int(self.fs) if int(self.fs) > 0 else 75
        initial_baseline = np.mean(cleaned[:warmup_pts, :], axis=0, keepdims=True)
        anchored = cleaned - initial_baseline
        
        # 3. KHỬ DỐC TRÔI (LINEAR DETREND)
        # Bẻ thẳng toàn bộ các đường bị trôi chéo trước khi nạp vào Bandpass
        detrended = detrend(anchored, axis=0, type='linear')
        
        # 4. Lọc Bandpass (0.1 - 0.6 Hz)
        # Lúc này t=0 đã bằng 0 và không còn dốc, bộ lọc sẽ KHÔNG tạo ra "ngọn núi"
        filtered = filtfilt(self.b, self.a, detrended, axis=0)
        
        # 5. Tính phương sai trên data SẠCH và bốc Top 5
        variances = np.var(filtered, axis=0)
        top_idx = np.argsort(variances)[-self.top_k:]
        top_subs = filtered[:, top_idx]
        
        # 6. Ép 5 kênh thành 1 bằng PCA
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(top_subs).flatten()
        
        # Đảo pha cho đúng cực hô hấp (hít vào hướng lên)
        if np.corrcoef(pc1, np.mean(top_subs, axis=1))[0, 1] < 0:
            pc1 = -pc1
            
        return pc1 - np.mean(pc1)