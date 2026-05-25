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
        self.images_array = []
        self.features_data = []
        self.indexed_db = None

    def _calcular_caracteristicas_desde_mascara(self, image, mascara_binaria):
        contours, _ = cv2.findContours(mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, 0, 0, 0

        contorno = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contorno)
        perim = cv2.arcLength(contorno, True)

        # Solidez
        hull = cv2.convexHull(contorno)
        hull_area = cv2.contourArea(hull)
        solidez = float(area) / hull_area if hull_area > 0 else 0

        # Circularidad
        circularidad = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0

        # HSV
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
                if img_name.startswith("._"):
                    continue

                img_path = os.path.join(class_path, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                self.images_array.append({'clase': fruit_class, 'nombre_archivo': img_name, 'imagen_cv2': image})

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                esquinas = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
                fondo_claro = np.mean(esquinas) > 127

                if fondo_claro:
                    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                else:
                    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                kernel = np.ones((7, 7), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                labels = measure.label(thresh)
                props = measure.regionprops(labels)

                for obj_idx, prop in enumerate([p for p in props if p.area > 500]):
                    obj_mask = ((labels == prop.label) * 255).astype('uint8')
                    solidez, circ, h, sat = self._calcular_caracteristicas_desde_mascara(image, obj_mask)
                    
                    self.features_data.append({
                        'clase': fruit_class,
                        'imagen': img_name,
                        'objeto_id': obj_idx + 1,
                        'caract_1_solidez': solidez,          # NUEVO: Solidez incluida en DB
                        'caract_2_circularidad': circ,
                        'caract_3_matiz_H': min(max(h, 0), 179),
                        'caract_4_saturacion_S': min(max(sat, 0), 255)
                    })

    def create_indexed_database(self):
        if not self.features_data:
            return None
        df = pd.DataFrame(self.features_data)
        self.indexed_db = df.set_index(['clase', 'imagen', 'objeto_id'])
        self.indexed_db.to_csv('base_datos_caracteristicas.csv')
        return self.indexed_db

    # ============================================================
    # USUARIO (CORREGIDO PARA SIMETRÍA EXACTA CON DATASET)
    # ============================================================

    def extraer_caracteristicas_imagen_usuario(self, ruta_imagen):
        if not os.path.exists(ruta_imagen):
            return None

        image = cv2.imread(ruta_imagen)
        
        if image is None:
            try:
                pil_img = PILImage.open(ruta_imagen).convert("RGB")
                image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except:
                return None

        # Usamos EXACTAMENTE el mismo preprocesamiento que en load_and_process_dataset
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        esquinas = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
        fondo_claro = np.mean(esquinas) > 127

        if fondo_claro:
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((7, 7), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        labels = measure.label(thresh)
        props = measure.regionprops(labels)

        # Filtramos objetos muy pequeños (basura del fondo)
        objetos_validos = [p for p in props if p.area > 500]

        if not objetos_validos:
            return None

        # Si hay varias frutas (ej. 3 naranjas), tomamos la más grande para la consulta principal
        prop_principal = max(objetos_validos, key=lambda p: p.area)

        # Generamos la máscara SOLO para esa fruta
        obj_mask = ((labels == prop_principal.label) * 255).astype('uint8')

        # Calculamos características
        solidez, circ, h, sat = self._calcular_caracteristicas_desde_mascara(image, obj_mask)

        # Aseguramos los mismos límites que en el dataset
        h = min(max(h, 0), 179)
        sat = min(max(sat, 0), 255)

        return [solidez, circ, h, sat]