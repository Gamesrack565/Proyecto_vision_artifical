import os
from src.modelo.image_model import ImageModel
from src.modelo.clasificador_model import CBIRClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class MainController:
    def __init__(self, dataset_path):
        self.image_model = ImageModel(dataset_path)
        self.classifier = CBIRClassifier()
        self.dataset_path = dataset_path

    def ejecutar_procesamiento_inicial(self, callback=None):
        def log(mensaje):
            print(mensaje)
            if callback:
                callback(mensaje)

        log("--- FASE 1: EXTRACCIÓN Y BASE DE DATOS ---")
        
        # MEJORA: Evitamos los 30 minutos de espera si el CSV ya está creado
        if os.path.exists('base_datos_caracteristicas.csv'):
            log("Base de datos CSV detectada. Saltando extracción de imágenes...")
        else:
            log("Paso 1, 2 y 3: Cargando imágenes y detectando objetos (Tomará tiempo)...")
            self.image_model.load_and_process_dataset()
            
            log("Paso 4: Extrayendo características (Solidez, Circ, Matiz, Saturación, Textura)...")
            db = self.image_model.create_indexed_database()
            
            if db is not None:
                log("Base de datos indexada creada y exportada a CSV.")
            else:
                log("ADVERTENCIA: No se generó la base de datos.")

        log("\n--- FASE 2: MOTOR CBIR ---")
        log("Paso 5: Indexando Espacio Vectorial...")
        total = self.classifier.entrenar_modelo('base_datos_caracteristicas.csv')

        if total:
            log(f"Modelo escalado con {total} vectores.")
            log("=> Motor de Visión Artificial CBIR cargado y listo.")
        else:
            log("ERROR: No se pudo indexar.")

    def procesar_consulta(self, ruta_imagen, callback=None):
        if callback:
            callback("Segmentando imagen y extrayendo vector...")

        vector_ia = self.image_model.extraer_caracteristicas_imagen_usuario(ruta_imagen)

        if vector_ia is None:
            return ("Error al segmentar imagen", [], 0)

        # Recibimos las 9 características
        sol, circ, prop, rugosidad, p_roj, p_nar, p_ver, sat, bri = vector_ia
        
        # DEBUG: Observa cómo el programa lee la textura y la saturación
        print(f"--> [DEBUG] Análisis:")
        print(f"    Rugosidad (Bordes): {rugosidad:.4f} | Proporción: {prop:.2f}")
        print(f"    Saturación: {sat:.1f}/255 | Brillo: {bri:.1f}/255")

        if callback:
            callback("Consultando similitud en el Motor CBIR...")

        prediccion, top_5, error = self.classifier.consultar_nueva_imagen(
            vector_caracteristicas=vector_ia,
            nombre_imagen=ruta_imagen
        )

        return (prediccion, top_5, error)
    
    def generar_matriz_confusion(self, ruta_carpeta_prueba, archivo_salida='matriz_confusion.csv'):
        print("\n--- INICIANDO EXAMEN: GENERANDO MATRIZ DE CONFUSIÓN ---")
        y_verdadero = []
        y_predicho = []

        # Las clases que tu modelo conoce
        clases = self.image_model.target_classes

        for clase_real in clases:
            ruta_clase = os.path.join(ruta_carpeta_prueba, clase_real)
            if not os.path.exists(ruta_clase): 
                continue

            print(f"Evaluando imágenes de la carpeta: {clase_real}...")
            # Analizamos cada imagen de esta carpeta
            for img_name in os.listdir(ruta_clase):
                if img_name.startswith("._"): continue
                
                ruta_img = os.path.join(ruta_clase, img_name)
                
                # 1. La IA extrae el vector
                vector_ia = self.image_model.extraer_caracteristicas_imagen_usuario(ruta_img)
                
                if vector_ia is not None:
                    # 2. La IA hace su predicción
                    prediccion, _, _ = self.classifier.consultar_nueva_imagen(vector_ia, ruta_img)
                    
                    # 3. Guardamos la realidad vs la predicción
                    y_verdadero.append(clase_real)
                    y_predicho.append(prediccion)

        # Usamos scikit-learn para cruzar los datos matemáticamente
        matriz = confusion_matrix(y_verdadero, y_predicho, labels=clases)
        
        # Convertimos la matriz en una tabla bonita de Pandas
        df_matriz = pd.DataFrame(matriz, 
                                 index=[f"Real_{c}" for c in clases], 
                                 columns=[f"Pred_{c}" for c in clases])
        
        # Exportamos al archivo
        with open('matriz_confusion.txt', 'w') as f:
            f.write(df_matriz.to_string())
        print(f"\n¡Examen terminado! Matriz guardada en: matriz_confusion.txt")

        plt.figure(figsize=(8, 6))
        # Usamos 'Purples' para que combine con el diseño de tu Dark UI
        sns.heatmap(df_matriz, annot=True, fmt='d', cmap='Purples') 
        plt.title('Matriz de Confusión CBIR')
        plt.ylabel('Fruta Real (Verdadero)')
        plt.xlabel('Predicción de la IA')

        # Guardar como imagen
        plt.tight_layout()
        plt.savefig('matriz_confusion.png')
        print("Matriz guardada como imagen: matriz_confusion.png")