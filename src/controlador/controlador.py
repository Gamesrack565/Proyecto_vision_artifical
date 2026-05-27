import os
import cv2
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from src.modelo.image_model import ImageModel
from src.modelo.clasificador_model import CBIRClassifier

class MainController:
    def __init__(self, dataset_path):
        self.image_model = ImageModel(dataset_path)
        self.classifier = CBIRClassifier()
        self.dataset_path = dataset_path

    def ejecutar_procesamiento_inicial(self, callback=None, forzar_recalculo=False):
        def log(mensaje):
            print(mensaje)
            if callback: 
                callback(mensaje)

        log("--- FASE 1: EXTRACCIÓN Y BASE DE DATOS ---")
        if not forzar_recalculo and os.path.exists('base_datos_caracteristicas.csv'):
            log("Base de datos CSV detectada. Saltando extracción de imágenes...")
        else:
            log("Paso 1, 2 y 3: Cargando imágenes y detectando objetos (Tomará tiempo)...")
            self.image_model.load_and_process_dataset()
            
            log("Paso 4: Extrayendo características avanzadas (Forma, Textura, Color)...")
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
            callback("Analizando imagen...")

        # ==============================================================
        # DEFENSA 1: BLOQUEO DE PERSONAS (Calibrado a 80x80 y minNeighbors=12)
        # ==============================================================
        image_cv = cv2.imread(ruta_imagen)
        if image_cv is not None:
            gray_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Subimos vecinos a 15 y tamaño a 100x100 para evitar los "ojos" del marañón
            rostros = face_cascade.detectMultiScale(gray_cv, scaleFactor=1.1, minNeighbors=15, minSize=(100, 100))
            
            if len(rostros) > 0:
                print("--> [DEBUG] SEGURIDAD: Rostro humano detectado. Bloqueando consulta.")
                return ("Rostro humano detectado", [], 0)

        # ==============================================================
        # EXTRACCIÓN Y DEFENSA 2 (Imagen fuera de contexto / Vacía)
        # ==============================================================
        if callback: 
            callback("Segmentando imagen y extrayendo vector...")

        vector_ia = self.image_model.extraer_caracteristicas_imagen_usuario(ruta_imagen)

        if vector_ia is None or sum(vector_ia) == 0:
            print("--> [DEBUG] SEGURIDAD: No se encontró ningún objeto clasificable.")
            return ("No reconocí ninguna fruta", [], 0)

        # Desempaquetamos las 9 características extraídas
        sol, circ, prop, rugosidad, p_roj, p_nar, p_ver, sat, bri = vector_ia
        print(f"--> [DEBUG] Análisis: Rugosidad: {rugosidad:.4f} | Sat: {sat:.1f} | Bri: {bri:.1f}")

        # ==============================================================
        # DEFENSA 3: CANDADO GEOMÉTRICO (Filtro Anti-Objetos Cotidianos)
        # ==============================================================
        if sol < 0.65 or prop > 3.0 or prop < 0.33:
            print(f"--> [DEBUG] SEGURIDAD: Geometría inválida. Solidez: {sol:.2f}, Proporción: {prop:.2f}")
            return ("Forma no válida (No parece una fruta)", [], 0)

        if callback: 
            callback("Consultando similitud en el Motor CBIR...")

        # ==============================================================
        # PASO CLAVE: CONSULTA AL CLASIFICADOR (Aquí se define la variable 'error')
        # ==============================================================
        prediccion, top_5, error = self.classifier.consultar_nueva_imagen(
            vector_caracteristicas=vector_ia,
            nombre_imagen=ruta_imagen
        )

        # ==============================================================
        # DEFENSA 5: UMBRAL DE DISTANCIA (Filtro de Objetos Desconocidos)
        # ==============================================================
        UMBRAL_MAXIMO = 8.5 
        
        if error > UMBRAL_MAXIMO:
            print(f"--> [DEBUG] SEGURIDAD: Objeto desconocido. Error ({error:.4f}) superó el umbral de {UMBRAL_MAXIMO}.")
            return (f"No lo reconoce (Distancia altísima: {error:.2f})", [], error)

        return (prediccion, top_5, error)

    # ==============================================================
    # EVALUACIÓN BAJO DEMANDA (Matriz de Confusión)
    # ==============================================================
    def generar_matriz_confusion(self, ruta_carpeta_prueba, archivo_salida='mi_matriz_confusion.csv', callback=None):
        if callback: 
            callback("Preparando entorno estadístico...")
        y_verdadero = []
        y_predicho = []
        clases = self.image_model.target_classes

        total_imagenes = 0
        for clase_real in clases:
            ruta_clase = os.path.join(ruta_carpeta_prueba, clase_real)
            if os.path.exists(ruta_clase):
                total_imagenes += len([f for f in os.listdir(ruta_clase) if not f.startswith("._")])

        contador = 0
        for clase_real in clases:
            ruta_clase = os.path.join(ruta_carpeta_prueba, clase_real)
            if not os.path.exists(ruta_clase): 
                continue

            for img_name in os.listdir(ruta_clase):
                if img_name.startswith("._"): 
                    continue
                ruta_img = os.path.join(ruta_clase, img_name)
                
                contador += 1
                if callback: 
                    callback(f"Evaluando {clase_real} ({contador}/{total_imagenes})...")
                
                vector_ia = self.image_model.extraer_caracteristicas_imagen_usuario(ruta_img)
                if vector_ia is not None:
                    prediccion, _, _ = self.classifier.consultar_nueva_imagen(vector_ia, ruta_img)
                    y_verdadero.append(clase_real)
                    y_predicho.append(prediccion)
                else:
                    y_verdadero.append(clase_real)
                    y_predicho.append("No lo reconoce")

        if callback: 
            callback("Calculando Matriz y Mapas de Calor...")
        
        labels_completos = clases + ["No lo reconoce"] if "No lo reconoce" in y_predicho else clases
        matriz = confusion_matrix(y_verdadero, y_predicho, labels=labels_completos)
        df_matriz = pd.DataFrame(matriz, index=[f"Real_{c}" for c in clases], columns=[f"Pred_{c}" for c in labels_completos])
        
        df_matriz.to_csv(archivo_salida)
        
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            sns.heatmap(df_matriz, annot=True, fmt='d', cmap='Purples')
            plt.title('Matriz de Confusión - Motor CBIR')
            plt.ylabel('Clase Real (Verdadero)')
            plt.xlabel('Predicción del Modelo')
            plt.tight_layout()
            plt.savefig('matriz_confusion.png')
            plt.close()
        except Exception as e:
            print(f"No se pudo generar la gráfica PNG: {e}")

        if callback: 
            callback("Métricas guardadas: 'mi_matriz_confusion.csv' y 'matriz_confusion.png'")