import numpy as np
from scipy.signal import butter, filtfilt

class DwtSeparator:
    def __init__(self, fs=30):
        self.fs = fs
        nyq = 0.5 * fs
        self.b_resp, self.a_resp = butter(3, [0.1 / nyq, 0.5 / nyq], btype='band')
        self.b_heart, self.a_heart = butter(3, [0.8 / nyq, 2.0 / nyq], btype='band')

    def dwtSeparator_extractBreathing(self, clean_signal):
        return filtfilt(self.b_resp, self.a_resp, clean_signal, axis=0)

    def dwtSeparator_extractHeartbeat(self, clean_signal):
        return filtfilt(self.b_heart, self.a_heart, clean_signal, axis=0)