import os
from src.modelo.image_model import ImageModel
from src.modelo.clasificador_model import CBIRClassifier

class MainController:
    def __init__(self, dataset_path):
        self.image_model = ImageModel(dataset_path)
        self.classifier  = CBIRClassifier()
        self.dataset_path = dataset_path

    def ejecutar_procesamiento_inicial(self, callback=None):
        """Ejecuta el procesamiento y envía mensajes a la vista usando el callback."""
        def log(mensaje):
            print(mensaje)
            if callback: 
                callback(mensaje)

        log("--- FASE 1: EXTRACCIÓN Y BASE DE DATOS ---")
        log("Paso 1, 2 y 3: Cargando imágenes y detectando objetos...")
        self.image_model.load_and_process_dataset()
        
        log("Paso 4: Extrayendo características (Circ, Matiz, Saturación)...")
        db = self.image_model.create_indexed_database()
        
        if db is not None:
            log("Base de datos indexada creada y exportada a CSV.")
        else:
            log("ADVERTENCIA: No se generó la base de datos (revisa tu carpeta dataset).")

        log("\n--- FASE 2: ENTRENAMIENTO Y CLASIFICACIÓN ---")
        log("Paso 5: Entrenando clasificador K-NN...")
        total = self.classifier.entrenar_modelo('base_datos_caracteristicas.csv')
        
        if total:
            log(f"Modelo entrenado con {total} vectores.")
            log("=> Motor de Visión Artificial cargado y listo.")
        else:
            log("ERROR: No se pudo entrenar el modelo.")

    def procesar_consulta(self, ruta_imagen, callback=None):
        """Procesa una imagen desde la interfaz gráfica."""
        if callback: callback("Segmentando imagen y removiendo fondo...")
        vector_real = self.image_model.extraer_caracteristicas_imagen_usuario(ruta_imagen)
        
        if vector_real is None:
            return "Error al leer imagen", [], 0

        solidez_trampa = vector_real[0]
        vector_ia      = vector_real[1:4] 

        if callback: callback("Evaluando solidez y geometría biológica...")
        
        # --- ESCUDO ANTI-HUMANOS (Nivel 1) ---
        if solidez_trampa == -1:
            return "Persona Detectada (Rostro encontrado)", [], -1

        # --- TRAMPA DE SOLIDEZ (Nivel 2) ---
        if solidez_trampa < 0.85:
            return "No es una fruta (Forma Irregular / Extremidades)", [], solidez_trampa

        if callback: callback("Consultando características en el modelo K-NN...")
        # --- INTELIGENCIA ARTIFICIAL K-NN (Nivel 3) ---
        prediccion, top_5, error_minimo = self.classifier.consultar_nueva_imagen(
            vector_ia, umbral_desconocido=3.0
        )
        
        return prediccion, top_5, error_minimo