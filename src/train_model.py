import pandas as pd
import numpy as np
import glob
import os
import re
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# Importamos nuestras funciones locales
from preprocessing import filtrar_senal, segmentar_ventanas
from features import extraer_todas_las_features, FEATURE_NAMES

def cargar_y_procesar_datos(base_path):
    """Lee los archivos crudos, filtra, ventanea y etiqueta."""
    all_files = sorted(glob.glob(base_path, recursive=True))
    datos = []
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        try:
            m = re.search(r'R(\d)_(dr|iz)_(.+)\.csv', filename)
            if not m: continue
            set_num, leg, subject = f"R{m.group(1)}", m.group(2), m.group(3)
            
            data = pd.read_csv(filepath)
            col_name = 'Voltage_mV' if 'Voltage_mV' in data.columns else data.columns[1]
            raw = data[col_name].values.astype(float)
            
            clean_signal = filtrar_senal(raw)
            V = segmentar_ventanas(clean_signal, window_size=500, step=250)
            if len(V) == 0: continue
            
            X_feats = extraer_todas_las_features(V)
            
            # Etiquetado por tercios
            n = len(V)
            tercio = n // 3
            labels = np.full(n, np.nan)
            labels[:tercio] = 0
            labels[-tercio:] = 1
            
            for i in range(n):
                if not np.isnan(labels[i]):
                    row = {'Subject': subject, 'Leg': leg, 'Set': set_num, 'Label': int(labels[i])}
                    for j, feat_name in enumerate(FEATURE_NAMES):
                        row[feat_name] = X_feats[i, j]
                    datos.append(row)
        except:
            pass
            
    return pd.DataFrame(datos)

def evaluar_loso(df, nombre_modelo, pipeline, param_grid=None):
    """Entrena y evalúa usando Leave-One-Subject-Out."""
    X = df[FEATURE_NAMES]
    y = df['Label']
    groups = df['Subject']
    
    logo = LeaveOneGroupOut()
    resultados = []
    
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        if param_grid:
            inner_cv = GroupKFold(n_splits=3)
            gs = GridSearchCV(pipeline, param_grid, cv=inner_cv.split(X_train, y_train, groups.iloc[train_idx]), scoring='f1_macro', n_jobs=-1)
            gs.fit(X_train, y_train)
            mejor_modelo = gs.best_estimator_
        else:
            mejor_modelo = pipeline
            mejor_modelo.fit(X_train, y_train)
            
        y_pred = mejor_modelo.predict(X_test)
        try:
            y_prob = mejor_modelo.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = np.nan
            
        p, r, f, s = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
        resultados.append({'Modelo': nombre_modelo, 'Accuracy': accuracy_score(y_test, y_pred), 'F1_Macro': f, 'ROC_AUC': auc})
        
    return pd.DataFrame(resultados)

if __name__ == "__main__":
    # NOTA: Cambia esta ruta por donde tengas tus archivos crudos
    BASE_PATH = '/content/dataset/EMG - RAW data/**/*.csv'
    
    print("1. Procesando archivos y calculando las 13 features...")
    df = cargar_y_procesar_datos(BASE_PATH)
    print(f"Dataset listo: {len(df)} ventanas totales.")
    
    print("\n2. Entrenando modelos (LOSO)...")
    pipe_maj = make_pipeline(DummyClassifier(strategy='most_frequent'))
    res_maj = evaluar_loso(df, "Baseline 1 (Majority)", pipe_maj)
    
    pipe_lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    res_lda = evaluar_loso(df, "Baseline 2 (LDA)", pipe_lda)
    
    pipe_rf = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
    grid_rf = {'randomforestclassifier__n_estimators': [100, 200], 'randomforestclassifier__max_depth': [None, 10]}
    res_rf = evaluar_loso(df, "Random Forest", pipe_rf, param_grid=grid_rf)
    
    # Consolidar y guardar
    df_res = pd.concat([res_maj, res_lda, res_rf])
    resumen = df_res.groupby('Modelo')[['Accuracy', 'F1_Macro', 'ROC_AUC']].agg(['mean', 'std']).round(3)
    
    print("\n=== RESULTADOS FINALES ===")
    print(resumen.to_string())
    
    # Guardar results.csv automáticamente
    df_res.to_csv('results.csv', index=False)
    print("\nArchivo 'results.csv' guardado exitosamente.")
