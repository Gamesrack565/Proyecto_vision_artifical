import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class CBIRClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.db_metadata = None
        self.X_scaled = None
        self.df = None
        
        # Ahora usamos 4 características para mayor precisión
        self.feature_columns = [
            'caract_1_solidez',
            'caract_2_circularidad',
            'caract_3_matiz_H',
            'caract_4_saturacion_S'
        ]

    def entrenar_modelo(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"No se encontró {csv_path}")
            return False

        self.df = df
        for col in self.feature_columns:
            if col not in df.columns:
                print(f"Falta columna: {col}")
                return False

        self.db_metadata = df[['clase', 'imagen', 'objeto_id']]
        X = df[self.feature_columns]
        
        # Normalizamos los datos para que HSV y Circularidad pesen igual
        self.X_scaled = self.scaler.fit_transform(X)
        return len(df)

    def consultar_nueva_imagen(self, vector_caracteristicas, nombre_imagen=None, umbral_confianza=0.45):
        vector_df = pd.DataFrame([vector_caracteristicas], columns=self.feature_columns)
        vector_scaled = self.scaler.transform(vector_df)[0]

        # ========================================================
        # CÁLCULO CBIR PURO (Distancia Euclidiana Vectorial)
        # ========================================================
        distancias = np.linalg.norm(self.X_scaled - vector_scaled, axis=1)
        
        resultados_temp = self.db_metadata.copy()
        resultados_temp['distancia'] = distancias
        resultados_ordenados = resultados_temp.sort_values(by='distancia')

        # ========================================================
        # LÓGICA DE "ZONAS" (Estricta a 1 sola clase)
        # ========================================================
        # Tomamos SOLO la clase del vector matemáticamente más cercano (Top 1)
        clase_ganadora = resultados_ordenados.iloc[0]['clase']

        resultados_finales = []
        imagenes_vistas = set()

        for _, fila in resultados_ordenados.iterrows():
            # SOLO permitimos que entren al Top 5 si son de la clase ganadora absoluta
            if fila['clase'] == clase_ganadora:
                if fila['imagen'] not in imagenes_vistas:
                    resultados_finales.append({
                        'clase_recuperada': fila['clase'],
                        'nombre_imagen': fila['imagen'],
                        'indice_error': round(fila['distancia'], 4)
                    })
                    imagenes_vistas.add(fila['imagen'])
                
            if len(resultados_finales) == 5:
                break

        clase_final = resultados_finales[0]['clase_recuperada']
        error_distancia = resultados_finales[0]['indice_error']

        print(f"--> [DEBUG] Motor CBIR - Distancia a la fruta más cercana: {error_distancia:.3f}")

        # UMBRAL DE RECHAZO MODERADO: Si la distancia matemática supera 2.5
        if error_distancia > 1.5:
            print("--> [DEBUG] RECHAZADO POR FILTRO MATEMÁTICO (Muy lejos del dataset)")
            return ("No entendí tu imagen", [], error_distancia)

        return (clase_final, resultados_finales, error_distancia)