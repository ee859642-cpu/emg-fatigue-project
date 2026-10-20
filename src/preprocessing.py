import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def filtrar_senal(emg_signal, fs=1000):
    """
    Aplica un filtro Butterworth pasabanda (20-450 Hz) y un Notch (60 Hz)
    para limpiar la señal de EMG.
    """
    nyquist = 0.5 * fs
    
    # 1. Filtro Butterworth (20 - 450 Hz)
    low_cutoff = 20 / nyquist
    high_cutoff = 450 / nyquist
    b, a = butter(4, [low_cutoff, high_cutoff], btype='band')
    emg_bandpassed = filtfilt(b, a, emg_signal)
    
    # 2. Filtro Notch (60 Hz)
    b_notch, a_notch = iirnotch(60.0 / nyquist, 30.0)
    emg_clean = filtfilt(b_notch, a_notch, emg_bandpassed)
    
    return emg_clean

def segmentar_ventanas(signal, window_size=500, step=250):
    """
    Segmenta la señal en ventanas de 500 ms con 50% de overlap.
    """
    ventanas = []
    for i in range(0, len(signal) - window_size + 1, step):
        ventanas.append(signal[i:i+window_size])
    return np.array(ventanas)
