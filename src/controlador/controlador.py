import os
from src.modelo.image_model import ImageModel
from src.modelo.clasificador_model import CBIRClassifier

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
        log("Paso 1, 2 y 3: Cargando imágenes y detectando objetos...")
        self.image_model.load_and_process_dataset()

        log("Paso 4: Extrayendo características (Solidez, Circ, Matiz, Saturación)...")
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

        # Ahora el vector_ia contiene EXACTAMENTE [solidez, circ, h, sat]
        # Ya no hay bloqueos manuales, dejamos que la IA y la geometría decidan.

        if callback:
            callback("Consultando similitud en el Motor CBIR...")

        prediccion, top_5, error = self.classifier.consultar_nueva_imagen(
            vector_caracteristicas=vector_ia,
            nombre_imagen=ruta_imagen
        )

        return (prediccion, top_5, error)