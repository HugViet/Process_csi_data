import matplotlib.pyplot as plt

def plotViewer_showResults(raw_phases, clean_waveform):
    plt.figure(figsize=(10, 6), facecolor='w')
    
    plt.subplot(2, 1, 1)
    plt.plot(raw_phases[:, :10])
    plt.title("Raw Phases (First 10 Subcarriers)")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(clean_waveform, color='red', linewidth=1.5)
    plt.title("PhaseBeat Clean 1D Waveform (DC Removed)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()