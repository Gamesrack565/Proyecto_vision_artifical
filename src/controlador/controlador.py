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

    def ejecutar_procesamiento_inicial(self, callback=None, forzar_recalculo=False):
        def log(mensaje):
            print(mensaje)
            if callback:
                callback(mensaje)

        log("--- FASE 1: EXTRACCIÓN Y BASE DE DATOS ---")
        
        # LÓGICA DE DECISIÓN: 
        # Si NO forzamos y el CSV existe, usamos la carga ultra rápida.
        # Si forzamos, o si el archivo no existe, hace el escaneo de 30 minutos.
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

        import cv2 # Nos aseguramos de tener OpenCV disponible aquí

        # ==============================================================
        # DEFENSA 1: BLOQUEO DE PERSONAS Y ROSTROS (Haar Cascade)
        # ==============================================================
        image_cv = cv2.imread(ruta_imagen)
        if image_cv is not None:
            gray_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            # Cargamos el modelo pre-entrenado de OpenCV para detectar rostros frontales
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # minNeighbors=10 es el secreto. Exige mucha seguridad antes de decir "es una cara"
            # minSize=(150, 150) asegura que no confunda manchas pequeñas con rostros
            rostros = face_cascade.detectMultiScale(gray_cv, scaleFactor=1.1, minNeighbors=7, minSize=(80, 80))
            
            if len(rostros) > 0:
                print("--> [DEBUG] SEGURIDAD: Rostro humano detectado. Bloqueando consulta.")
                # Al devolver los corchetes vacíos [], tu vista automáticamente mostrará "BLOQUEADO"
                return ("Rostro humano detectado", [], 0)

        # ==============================================================
        # EXTRACCIÓN NORMAL DE LA FRUTA
        # ==============================================================
        if callback:
            callback("Segmentando imagen y extrayendo vector...")

        vector_ia = self.image_model.extraer_caracteristicas_imagen_usuario(ruta_imagen)

        # DEFENSA 2: Si no encontró nada con forma en absoluto (ej. un paisaje vacío)
        if vector_ia is None or sum(vector_ia) == 0:
            print("--> [DEBUG] SEGURIDAD: No se encontró ningún objeto clasificable.")
            return ("No reconocí ninguna fruta", [], 0)

        # Recibimos las 9 características
        sol, circ, prop, rugosidad, p_roj, p_nar, p_ver, sat, bri = vector_ia
        
        print(f"--> [DEBUG] Análisis de Fruta:")
        print(f"    Rugosidad (Bordes): {rugosidad:.4f} | Proporción: {prop:.2f}")
        print(f"    Saturación: {sat:.1f}/255 | Brillo: {bri:.1f}/255")

        if callback:
            callback("Consultando similitud en el Motor CBIR...")

        prediccion, top_5, error = self.classifier.consultar_nueva_imagen(
            vector_caracteristicas=vector_ia,
            nombre_imagen=ruta_imagen
        )

        return (prediccion, top_5, error)
    
    def generar_matriz_confusion(self, ruta_carpeta_prueba, archivo_salida='mi_matriz_confusion.csv', callback=None):
        if callback: callback("Preparando entorno estadístico...")
        y_verdadero = []
        y_predicho = []
        clases = self.image_model.target_classes

        # Contamos cuántas imágenes hay en total para calcular el porcentaje de avance
        total_imagenes = 0
        for clase_real in clases:
            ruta_clase = os.path.join(ruta_carpeta_prueba, clase_real)
            if os.path.exists(ruta_clase):
                total_imagenes += len([f for f in os.listdir(ruta_clase) if not f.startswith("._")])

        contador = 0
        for clase_real in clases:
            ruta_clase = os.path.join(ruta_carpeta_prueba, clase_real)
            if not os.path.exists(ruta_clase): continue

            for img_name in os.listdir(ruta_clase):
                if img_name.startswith("._"): continue
                ruta_img = os.path.join(ruta_clase, img_name)
                
                contador += 1
                if callback: 
                    callback(f"Evaluando {clase_real} ({contador}/{total_imagenes})...")
                
                # Extracción y predicción silenciosa
                vector_ia = self.image_model.extraer_caracteristicas_imagen_usuario(ruta_img)
                if vector_ia is not None:
                    prediccion, _, _ = self.classifier.consultar_nueva_imagen(vector_ia, ruta_img)
                    y_verdadero.append(clase_real)
                    y_predicho.append(prediccion)

        if callback: callback("Calculando Matriz y Mapas de Calor...")
        matriz = confusion_matrix(y_verdadero, y_predicho, labels=clases)
        df_matriz = pd.DataFrame(matriz, 
                                 index=[f"Real_{c}" for c in clases], 
                                 columns=[f"Pred_{c}" for c in clases])
        
        # Exportación del archivo de datos
        df_matriz.to_csv(archivo_salida)
        
        # Exportación de la gráfica con Matplotlib y Seaborn
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

        if callback: callback("Métricas guardadas: 'mi_matriz_confusion.csv' y 'matriz_confusion.png'")