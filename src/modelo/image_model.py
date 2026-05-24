import os
import io
import cv2
import numpy as np
import pandas as pd
from skimage import measure
from rembg import remove
from PIL import Image as PILImage

class ImageModel:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.target_classes = ['watermelon', 'orange', 'fuji_apple', 'kiwi', 'plum']
        self.images_array  = []
        self.features_data = []
        self.indexed_db    = None

    def _calcular_caracteristicas_desde_mascara(self, image, mascara_binaria):
        contours, _ = cv2.findContours(mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, 0, 0, 0
            
        contorno = max(contours, key=cv2.contourArea)
        area     = cv2.contourArea(contorno)
        perim    = cv2.arcLength(contorno, True)
        
        # --- SOLIDEZ (Trampa anti-humanos) ---
        hull = cv2.convexHull(contorno)
        hull_area = cv2.contourArea(hull)
        solidez = float(area) / hull_area if hull_area > 0 else 0

        # --- CARACTERÍSTICA 1: Circularidad ---
        circularidad = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0

        # --- CARACTERÍSTICAS 2 y 3: Matiz (H) y Saturación (S) ---
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_hsv = cv2.mean(hsv, mask=mascara_binaria)
        promedio_h = mean_hsv[0] 
        promedio_s = mean_hsv[1] 

        return solidez, circularidad, promedio_h, promedio_s

    def load_and_process_dataset(self):
        for fruit_class in self.target_classes:
            class_path = os.path.join(self.dataset_path, fruit_class)
            if not os.path.exists(class_path):
                continue

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                image    = cv2.imread(img_path)
                if image is None:
                    continue

                self.images_array.append({'clase': fruit_class, 'nombre_archivo': img_name, 'imagen_cv2': image})

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                
                # --- NUEVO: Detección inteligente de fondo ---
                # Analizamos los píxeles de las 4 esquinas de la foto
                esquinas = [gray[0,0], gray[0,-1], gray[-1,0], gray[-1,-1]]
                fondo_claro = np.mean(esquinas) > 127
                
                if fondo_claro:
                    # Si el fondo es blanco (mayoría del dataset), invertimos
                    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                else:
                    # Si el fondo es oscuro (como tu orange_090.jpg), binarizamos normal
                    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                kernel = np.ones((7, 7), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

                labels = measure.label(thresh)
                props  = measure.regionprops(labels)

                for obj_idx, prop in enumerate([p for p in props if p.area > 500]):
                    obj_mask = ((labels == prop.label) * 255).astype('uint8')
                    
                    solidez, circ, h, sat = self._calcular_caracteristicas_desde_mascara(image, obj_mask)

                    self.features_data.append({
                        'clase':                  fruit_class,
                        'imagen':                 img_name,
                        'objeto_id':              obj_idx + 1,
                        'caract_1_circularidad':  circ,
                        'caract_2_matiz_H':       h,
                        'caract_3_saturacion_S':  sat
                    })

    def create_indexed_database(self):
        if not self.features_data:
            return
        df = pd.DataFrame(self.features_data)
        self.indexed_db = df.set_index(['clase', 'imagen', 'objeto_id'])
        self.indexed_db.to_csv('base_datos_caracteristicas.csv')
        return self.indexed_db

    def extraer_caracteristicas_imagen_usuario(self, ruta_imagen):
        if not os.path.exists(ruta_imagen):
            return None

        image = cv2.imread(ruta_imagen)
        if image is None:
            try:
                pil_img = PILImage.open(ruta_imagen).convert("RGB")
                image   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. ESCUDO ANTI-HUMANOS (Nivel 1)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        rostros = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(rostros) > 0:
            return [-1, 0, 0, 0]

        # 2. ENRUTAMIENTO INTELIGENTE (Dataset vs Internet)
        # Analizamos la desviación estándar de color en las 4 esquinas de la foto
        esquinas = [gray[0,0], gray[0,-1], gray[-1,0], gray[-1,-1]]
        
        # Si las esquinas son casi del mismo color (desviación menor a 15), es una foto de estudio/dataset
        if np.std(esquinas) < 15:
            # ---> RUTA A: OPENCV (Precisión 100% para Dataset)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            fondo_claro = np.mean(esquinas) > 127
            
            if fondo_claro:
                _, mascara_binaria = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:
                _, mascara_binaria = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
            # Usamos el MISMO kernel (7x7) que en el entrenamiento
            kernel = np.ones((7, 7), np.uint8)
            mascara_binaria = cv2.morphologyEx(mascara_binaria, cv2.MORPH_OPEN, kernel)
            
        else:
            # ---> RUTA B: INTELIGENCIA ARTIFICIAL (Para fotos de Internet/complejas)
            _, buf = cv2.imencode('.png', image)
            imagen_sin_fondo_bytes = remove(buf.tobytes())

            pil_result = PILImage.open(io.BytesIO(imagen_sin_fondo_bytes)).convert("RGBA")
            imagen_sin_fondo = np.array(pil_result)
            
            mascara_binaria = (imagen_sin_fondo[:, :, 3] > 127).astype('uint8') * 255

            if mascara_binaria.max() == 0:
                return None

            # Ajustamos el kernel de la IA a 7x7 para minimizar diferencias de área
            kernel_usuario = np.ones((7, 7), np.uint8)
            mascara_binaria = cv2.morphologyEx(mascara_binaria, cv2.MORPH_OPEN, kernel_usuario)

        # 3. Extracción Final
        solidez, circ, h, sat = self._calcular_caracteristicas_desde_mascara(image, mascara_binaria)
        return [solidez, circ, h, sat]