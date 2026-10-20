import numpy as np

def extraer_todas_las_features(V):
    """
    Recibe una matriz de ventanas (V) y calcula las 13 features.
    Devuelve un array de numpy con las columnas: 
    [RMS, MAV, WL, ZC, SSC, MDF, MNF, PotBaja, PotMedia, PotAlta, Hjorth_Mob, Hjorth_Comp, Kurtosis]
    """
    # 1. Amplitud / Tiempo
    rms = np.sqrt((V**2).mean(axis=1))
    mav = np.abs(V).mean(axis=1)
    wl = np.abs(np.diff(V, axis=1)).sum(axis=1)
    zc = ((V[:, :-1] * V[:, 1:]) < 0).sum(axis=1)
    ssc = ((np.diff(V, axis=1)[:, :-1] * np.diff(V, axis=1)[:, 1:]) < 0).sum(axis=1)
    
    # 2. Frecuencia (MDF, MNF, Bandas)
    w = V * np.hanning(V.shape[1])
    P = np.abs(np.fft.rfft(w, n=2048, axis=1))**2
    f = np.fft.rfftfreq(2048, 1/1000)
    banda = (f >= 20) & (f <= 450)
    P, f = P[:, banda], f[banda]
    tot = P.sum(axis=1)
    
    acum = np.cumsum(P, axis=1)
    mdf = f[(acum >= acum[:, -1:] / 2).argmax(axis=1)]
    mnf = (P * f).sum(axis=1) / tot
    
    pot_baja = P[:, (f >= 20) & (f < 60)].sum(axis=1) / tot
    pot_media = P[:, (f >= 60) & (f < 150)].sum(axis=1) / tot
    pot_alta = P[:, (f >= 150)].sum(axis=1) / tot
    
    # 3. Hjorth / Forma
    d1 = np.diff(V, axis=1)
    d2 = np.diff(d1, axis=1)
    var0 = np.var(V, axis=1) + 1e-9
    var1 = np.var(d1, axis=1) + 1e-9
    var2 = np.var(d2, axis=1) + 1e-9
    
    mob = np.sqrt(var1 / var0)
    comp = np.sqrt(var2 / var1) / mob
    kurt = np.mean((V - np.mean(V, axis=1, keepdims=True))**4, axis=1) / (var0**2)
    
    # Unir todas las columnas en una sola matriz
    return np.column_stack((rms, mav, wl, zc, ssc, mdf, mnf, pot_baja, pot_media, pot_alta, mob, comp, kurt))

# Lista de nombres de las columnas en el orden exacto que devuelve la función
FEATURE_NAMES = ['RMS', 'MAV', 'WL', 'ZC', 'SSC', 'MDF', 'MNF', 
                 'PotBaja', 'PotMedia', 'PotAlta', 'Hjorth_Mob', 'Hjorth_Comp', 'Kurtosis']
