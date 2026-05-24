import os
import cv2
import pandas as pd
import numpy as np # Nueva importación
from skimage import measure
from rembg import remove

class ImageModel:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        # Limitado a 5 clases como solicitaste
        self.target_classes = ['watermelon', 'orange', 'fuji_apple', 'kiwi', 'plum']
        
        # Almacenamiento en memoria
        self.images_array = []   # Arreglo para guardar las imágenes completas
        self.features_data = []  # Lista temporal para la base de datos
        self.indexed_db = None   # Base de datos final indexada (Pandas DataFrame)

    def load_and_process_dataset(self):
        """Lee el dataset, guarda en arreglo, cuenta objetos y extrae características."""
        for fruit_class in self.target_classes:
            class_path = os.path.join(self.dataset_path, fruit_class)
            
            if not os.path.exists(class_path):
                print(f"Advertencia: No se encontró la ruta {class_path}")
                continue

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                
                # 1. Leer imagen
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # 2. Guardar la imagen en el arreglo (todas las imágenes)
                self.images_array.append({
                    'clase': fruit_class,
                    'nombre_archivo': img_name,
                    'imagen_cv2': image
                })

                # 3. Procesamiento para encontrar objetos (Binarización Otsu)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Aplicamos desenfoque para reducir el ruido del fondo
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                # Invertimos porque skimage espera que el objeto sea blanco (255) y el fondo negro (0)
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # 4. Etiquetado y Extracción de características (Equivalente a regionprops)
                labels = measure.label(thresh)
                props = measure.regionprops(labels)

                # 5. Contar objetos presentes en la imagen
                # Filtramos áreas muy pequeñas para no contar ruido como objetos
                objetos_validos = [p for p in props if p.area > 500]
                num_objetos = len(objetos_validos)

                for obj_idx, prop in enumerate(objetos_validos):
                    # --- NUEVO: Extraer color promedio ---
                    # Creamos una máscara exclusiva para este objeto
                    obj_mask = (labels == prop.label).astype('uint8')
                    
                    # cv2.mean devuelve (Blue, Green, Red, Alpha) usando la máscara
                    mean_color = cv2.mean(image, mask=obj_mask)
                    promedio_b, promedio_g, promedio_r = mean_color[:3]

                    # --- NUEVO: Calcular Circularidad ---
                    perimetro = prop.perimeter
                    # Fórmula de circularidad. Si es 0, evitamos división por cero.
                    circularidad = (4 * np.pi * prop.area) / (perimetro ** 2) if perimetro > 0 else 0

                    self.features_data.append({
                        'clase': fruit_class,
                        'imagen': img_name,
                        'objeto_id': obj_idx + 1,
                        'caract_1_circularidad': circularidad,
                        'caract_2_color_R': promedio_r,
                        'caract_3_color_G': promedio_g
                    })

    def create_indexed_database(self):
        """Genera una base de datos indexada a partir de los datos extraídos."""
        if not self.features_data:
            print("No hay datos para crear la base de datos.")
            return

        df = pd.DataFrame(self.features_data)
        
        # Creamos una base de datos indexada usando la Clase y la Imagen como índices principales
        self.indexed_db = df.set_index(['clase', 'imagen', 'objeto_id'])
        
        # Guardar en CSV para visualizar los resultados fácilmente
        self.indexed_db.to_csv('base_datos_caracteristicas.csv')
        return self.indexed_db

    def extraer_caracteristicas_imagen_usuario(self, ruta_imagen):
        """
        Lee una imagen, remueve el fondo usando Inteligencia Artificial (U-Net)
        y devuelve su vector [Excentricidad, Circularidad, R, G, B].
        """
        if not os.path.exists(ruta_imagen):
            print(f"Error: No se encontró la imagen en {ruta_imagen}")
            return None

        # 1. Leer imagen original
        image = cv2.imread(ruta_imagen)
        if image is None:
            print("Error: OpenCV no pudo leer la imagen.")
            return None

        print("         -> Aplicando segmentación con IA (Removiendo fondo)...")
        # 2. IA en acción: remove() recorta el objeto y devuelve una imagen RGBA
        # (Red, Green, Blue, Alpha). El canal Alpha determina qué es transparente.
        imagen_sin_fondo = remove(image)

        # 3. El canal Alpha (índice 3) es nuestra máscara perfecta (255 = objeto, 0 = fondo)
        mascara_ia = imagen_sin_fondo[:, :, 3]

        # 4. Obtener características de forma usando la máscara de la IA
        labels = measure.label(mascara_ia > 0)
        props = measure.regionprops(labels)

        if not props:
            print("         -> La IA no detectó ningún objeto principal.")
            return None

        # Asumimos que la fruta es el objeto más grande detectado por la IA
        objeto_principal = max(props, key=lambda item: item.area)

        # 5. Extraer el color promedio usando la imagen original, pero solo 
        # dentro de la máscara que generó la IA
        mean_color = cv2.mean(image, mask=mascara_ia)
        promedio_b, promedio_g, promedio_r = mean_color[:3]

        # 6. Calcular Circularidad
        perimetro = objeto_principal.perimeter
        circularidad = (4 * np.pi * objeto_principal.area) / (perimetro ** 2) if perimetro > 0 else 0

        # Vector con las 5 características
        # Vector con 3 características para facilitar la matriz
        vector_extraido = [
            circularidad,
            promedio_r,
            promedio_g
        ]
        
        return vector_extraido