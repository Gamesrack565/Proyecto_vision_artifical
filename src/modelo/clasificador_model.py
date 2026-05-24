import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

class CBIRClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.knn    = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        self.db_metadata    = None
        
        # Actualizado con las nuevas columnas definitivas
        self.feature_columns = [
            'caract_1_circularidad',
            'caract_2_matiz_H',
            'caract_3_saturacion_S'
        ]

    def entrenar_modelo(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"Error: No se encontró {csv_path}")
            return False

        for col in self.feature_columns:
            if col not in df.columns:
                print(f"Error: Falta columna '{col}'. Regenera el CSV.")
                return False

        self.db_metadata = df[['clase', 'imagen', 'objeto_id']]
        X = df[self.feature_columns]
        y = df['clase']
        self.knn.fit(self.scaler.fit_transform(X), y)
        return len(df)

    def consultar_nueva_imagen(self, vector_caracteristicas, umbral_desconocido=3.0):
        vector_df     = pd.DataFrame([vector_caracteristicas], columns=self.feature_columns)
        vector_scaled = self.scaler.transform(vector_df)

        clase_predicha   = self.knn.predict(vector_scaled)[0]
        n_busqueda       = min(30, len(self.db_metadata))
        distancias, indices = self.knn.kneighbors(vector_scaled, n_neighbors=n_busqueda)
        distancia_minima    = distancias[0][0]

        if distancia_minima > umbral_desconocido:
            return "Desconocido (No lo conocen)", [], distancia_minima

        resultados_similares = []
        for i, idx in enumerate(indices[0]):
            meta = self.db_metadata.iloc[idx]
            if meta['clase'] == clase_predicha:
                resultados_similares.append({
                    'clase_recuperada': meta['clase'],
                    'nombre_imagen':    meta['imagen'],
                    'indice_error':     round(distancias[0][i], 4)
                })
                
            if len(resultados_similares) == 5:
                break
                
        return clase_predicha, resultados_similares, distancia_minima