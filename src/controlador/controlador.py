from src.modelo.image_model import ImageModel
from src.modelo.clasificador_model import CBIRClassifier

class MainController:
    def __init__(self, dataset_path):
        self.image_model = ImageModel(dataset_path)
        self.classifier = CBIRClassifier()

    def ejecutar_procesamiento_inicial(self):
        print("\n--- FASE 1: EXTRACCIÓN Y BASE DE DATOS ---")
        print("Paso 1 - Carga de imágenes...")
        print("Paso 2 - Creación de arreglo...")
        print("Paso 3 - Detección de objetos...")
        self.image_model.load_and_process_dataset()
        
        print("Paso 4 - Extracción de características...")
        # Prints actualizados para reflejar las 5 características finales
        print("         Características utilizadas: 1) Circularidad, 2) Rojo, 3) Verde")
        db = self.image_model.create_indexed_database()
        
        if db is not None:
            print("\nBase de datos indexada creada y exportada a CSV exitosamente.")

        print("\n--- FASE 2: ENTRENAMIENTO Y CLASIFICACIÓN CBIR ---")
        print("Paso 5 - Entrenando clasificador K-NN con datos normalizados...")
        total_entrenados = self.classifier.entrenar_modelo('base_datos_caracteristicas.csv')
        print(f"         Modelo entrenado con {total_entrenados} vectores de características.")

        print("\nPaso 6 - Prueba de Consulta con Imagen del Usuario...")
        
        # Aquí define la ruta de la imagen con la que quieres probar
        ruta_imagen_usuario = "/home/gamesrack/Descargas/ss.jpg"
        
        print(f"         Procesando imagen: {ruta_imagen_usuario}")
        vector_real = self.image_model.extraer_caracteristicas_imagen_usuario(ruta_imagen_usuario)

        if vector_real:
            # Formateamos el print del vector para que se lea más limpio en consola
            vector_redondeado = [round(val, 4) for val in vector_real]
            print(f"         Vector extraído: {vector_redondeado}")
            
            # --- FILTRO BIOLÓGICO DE FORMA ANTES DEL K-NN ---
            # --- FILTRO BIOLÓGICO DE FORMA ANTES DEL K-NN ---
            circularidad = vector_real[0] # ¡Ahora el índice 0 es la circularidad!
            
            if circularidad < 0.12:
                # Si la circularidad es menor a 0.3, es físicamente imposible que sea una fruta entera
                print("\n=> RECHAZO AUTOMÁTICO: La silueta detectada es demasiado irregular (Circularidad < 0.3).")
                print("   -> El objeto no tiene forma de fruta. Búsqueda abortada.")
            else:
                # Si pasa el filtro biológico, le preguntamos a la IA con un umbral de seguridad estricto (1.5)
                prediccion, top_5, error_minimo = self.classifier.consultar_nueva_imagen(vector_real, umbral_desconocido=3.0)
                
                print(f"\n=> RESULTADO: '{prediccion}' (Índice de error mínimo: {error_minimo:.4f})")
                
                if top_5:
                    print("\n=> LAS 5 IMÁGENES MÁS PARECIDAS SON:")
                    for i, res in enumerate(top_5, 1):
                        print(f"   {i}. Imagen: {res['nombre_imagen']} | Clase: {res['clase_recuperada']} | Índice de error: {res['indice_error']}")
                else:
                    print("   -> El sistema descartó la imagen (superó el límite de error permitido por la base de datos).")