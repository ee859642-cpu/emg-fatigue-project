import pandas as pd
import numpy as np
import joblib
import argparse
import matplotlib.pyplot as plt  # <-- Agregado para graficar
from preprocessing import filtrar_senal, segmentar_ventanas
from features import extraer_todas_las_features

def predecir_nuevo_archivo(filepath, modelo_path='models/modelo_emg_fatiga.pkl'):
    """
    Carga el modelo entrenado y predice si un archivo nuevo de EMG presenta fatiga.
    """
    print(f"Cargando modelo desde: {modelo_path}")
    try:
        modelo = joblib.load(modelo_path)
    except FileNotFoundError:
        print("Error: No se encontró el archivo del modelo. Asegúrate de haber corrido el entrenamiento y guardado el .pkl.")
        return

    print(f"Procesando archivo de prueba: {filepath}")
    
    # 1. Cargar y filtrar datos
    data = pd.read_csv(filepath)
    col_name = 'Voltage_mV' if 'Voltage_mV' in data.columns else data.columns[1]
    raw_signal = data[col_name].values.astype(float)
    
    clean_signal = filtrar_senal(raw_signal)
    
    # 2. Segmentar en ventanas de 500 ms
    V = segmentar_ventanas(clean_signal, window_size=500, step=250)
    if len(V) == 0:
        print("El archivo es demasiado corto para ser evaluado (menos de 500 ms).")
        return
        
    # 3. Extraer las 13 features
    X_new = extraer_todas_las_features(V)
    
    # 4. Predecir
    predicciones = modelo.predict(X_new)
    probabilidades = modelo.predict_proba(X_new)[:, 1]
    
    # 5. Mostrar resultados
    porcentaje_fatiga = np.mean(predicciones) * 100
    prob_media = np.mean(probabilidades) * 100
    
    print("\n=== RESULTADOS DE LA PREDICCIÓN ===")
    print(f"Ventanas evaluadas: {len(predicciones)}")
    print(f"Porcentaje de tiempo en Fatiga: {porcentaje_fatiga:.2f}%")
    print(f"Probabilidad promedio de Fatiga: {prob_media:.2f}%")
    
    clasificacion = "FATIGADO" if porcentaje_fatiga > 50 else "NO FATIGADO"
    print(f"Clasificación final: {clasificacion}")

    # --- 6. GENERAR LA GRÁFICA ---
    # Calcular RMS y MDF para graficar (calculados directamente de las ventanas V)
    rms_vals = np.sqrt((V**2).mean(axis=1))
    
    w = V * np.hanning(V.shape[1])
    P = np.abs(np.fft.rfft(w, n=2048, axis=1))**2
    f = np.fft.rfftfreq(2048, 1/1000)
    banda = (f >= 20) & (f <= 450)
    P, f = P[:, banda], f[banda]
    acum = np.cumsum(P, axis=1)
    mdf_vals = f[(acum >= acum[:, -1:] / 2).argmax(axis=1)]

    tiempo = np.arange(len(V)) * 0.25
    tercio = len(V) // 3

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(tiempo, rms_vals, color='blue')
    ax[0].set_ylabel('RMS (Amplitud)')
    ax[0].set_title('Evolución de Fatiga en la Señal Evaluada')
    ax[0].axvline(tiempo[tercio], color='gray', linestyle='--')
    ax[0].axvline(tiempo[-tercio], color='gray', linestyle='--')

    ax[1].plot(tiempo, mdf_vals, color='red')
    ax[1].set_ylabel('MDF (Hz)')
    ax[1].set_xlabel('Tiempo (segundos)')
    ax[1].axvline(tiempo[tercio], color='gray', linestyle='--')
    ax[1].axvline(tiempo[-tercio], color='gray', linestyle='--')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predecir fatiga en una señal EMG nueva.")
    parser.add_argument("--file", type=str, required=True, help="Ruta al archivo CSV que se va a evaluar.")
    parser.add_argument("--model", type=str, default="models/modelo_emg_fatiga.pkl", help="Ruta al modelo guardado (.pkl).")
    
    args = parser.parse_args()
    predecir_nuevo_archivo(args.file, args.model)
