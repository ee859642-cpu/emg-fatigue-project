import pandas as pd
import numpy as np
import re

def extraer_metadatos(filename):
    """Extrae la serie y la pierna del nombre del archivo (ej. R1_dr_Sujeto.csv)"""
    match = re.search(r'R(\d)_(dr|iz)_', filename)
    if match:
        serie = f"R{match.group(1)}"
        pierna = match.group(2)
        return pd.Series([serie, pierna])
    return pd.Series([np.nan, np.nan])

def etiquetar_tercios(group):
    """Aplica la regla de los tercios: primer tercio 0, último tercio 1, medio NaN"""
    n = len(group)
    tercio = n // 3
    labels = np.full(n, np.nan)
    labels[:tercio] = 0     # No fatigado
    labels[-tercio:] = 1    # Fatigado
    return pd.Series(labels, index=group.index)

def procesar_etiquetado(df):
    """Toma el DataFrame de features y le añade las etiquetas"""
    df[['Set', 'Leg']] = df['File_Name'].apply(extraer_metadatos)
    df['Subject'] = df['File_Name'].apply(lambda x: x.split('_')[-1].replace('.csv', ''))
    
    # Ordenar y agrupar para etiquetar cada serie individualmente
    df = df.sort_values(by=['File_Name']).reset_index(drop=True)
    df['Label'] = df.groupby(['Subject', 'Leg', 'Set'], group_keys=False).apply(etiquetar_tercios)
    
    # Quitar el tercio medio
    df_final = df.dropna(subset=['Label']).copy()
    df_final['Label'] = df_final['Label'].astype(int)
    return df_final
