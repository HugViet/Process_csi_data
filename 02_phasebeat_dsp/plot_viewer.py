import matplotlib.pyplot as plt

def plotViewer_showResults(raw_amplitudes, clean_waveform):
    plt.figure(figsize=(10, 6), facecolor='w')
    
    plt.subplot(2, 1, 1)
    plt.plot(raw_amplitudes[:, :10])
    plt.title("Raw CSI Amplitudes (First 10 Subcarriers)")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(clean_waveform, color='red', linewidth=1.5)
    plt.title("Selected Respiration-Band CSI Amplitude")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()   1404 14 