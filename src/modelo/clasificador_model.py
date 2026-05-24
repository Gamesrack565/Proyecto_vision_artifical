import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

class CBIRClassifier:
    def __init__(self):
        # El Scaler es vital para normalizar (Área vs Excentricidad)
        self.scaler = StandardScaler()
        
        # Configuramos K-NN para buscar exactamente los 5 vecinos más cercanos
        # Usamos la distancia euclidiana estándar para comparar los vectores
        self.knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        
        # Guardaremos los metadatos para saber qué imagen corresponde a cada vector
        self.db_metadata = None 

    def entrenar_modelo(self, csv_path):
        """Lee la BD, normaliza los datos y entrena el clasificador K-NN."""
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {csv_path}")
            return False

        # Guardamos la referencia visual (qué clase e imagen es cada fila)
        self.db_metadata = df[['clase', 'imagen', 'objeto_id']]
        
        # ACTUALIZADO: Leemos las 5 nuevas columnas
        # ACTUALIZADO: Leemos solo las 3 columnas acordadas
        X = df[['caract_1_circularidad', 'caract_2_color_R', 'caract_3_color_G']]
        y = df['clase']

        # 1. Ajustar el scaler y transformar los datos de entrenamiento
        X_scaled = self.scaler.fit_transform(X)

        # 2. Entrenar el modelo K-NN
        self.knn.fit(X_scaled, y)
        return len(df)

    def consultar_nueva_imagen(self, vector_caracteristicas, umbral_desconocido=3.0):
        nombres_columnas = ['caract_1_circularidad', 'caract_2_color_R', 'caract_3_color_G']
        vector_df = pd.DataFrame([vector_caracteristicas], columns=nombres_columnas)
        
        vector_scaled = self.scaler.transform(vector_df)
        
        # 1. Predecir la clase principal (el K-NN hace su votación interna)
        clase_predicha = self.knn.predict(vector_scaled)[0]
        
        # 2. Pedimos un margen amplio de vecinos (ej. 30) a la red K-NN
        # min() evita errores si tu base de datos tiene menos de 30 imágenes
        n_busqueda = min(30, len(self.db_metadata))
        distancias, indices = self.knn.kneighbors(vector_scaled, n_neighbors=n_busqueda)
        
        # 3. Lógica "No lo conocen" evaluando al vecino absoluto más cercano
        distancia_minima = distancias[0][0]
        
        if distancia_minima > umbral_desconocido:
            clase_predicha = "Desconocido (No lo conocen)"
            resultados_similares = [] 
        else:
            # 4. LA "TRAMPA" (Filtrado por clase)
            resultados_similares = []
            for i, idx in enumerate(indices[0]): 
                meta = self.db_metadata.iloc[idx]
                
                # Solo agregamos el resultado si pertenece a la misma clase predicha
                if meta['clase'] == clase_predicha:
                    resultados_similares.append({
                        'clase_recuperada': meta['clase'],
                        'nombre_imagen': meta['imagen'],
                        'indice_error': round(distancias[0][i], 4)
                    })
                    
                # Nos detenemos exactamente cuando tengamos las 5 mejores
                if len(resultados_similares) == 5:
                    break
            
        return clase_predicha, resultados_similares, distancia_minima