import os
import cv2
import numpy as np
import pandas as pd
from rembg import remove
from PIL import Image as PILImage

class ImageModel:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.target_classes = ['onion', 'cashew', 'watermelon', 'plum', 'orange']
        self.features_data = []
    
    def _obtener_mascara_perfecta(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(image_rgb)
        output_pil = remove(pil_img)
        output_bgra = np.array(output_pil)
        alpha = output_bgra[:, :, 3]
        _, mascara = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        return mascara

    def _calcular_caracteristicas_desde_mascara(self, image, mascara_binaria):
        contours, _ = cv2.findContours(mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return [0]*9

        contorno = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contorno)
        perim = cv2.arcLength(contorno, True)
        
        # 1. FORMA (Circularidad y Trapecios)
        x, y, w, h = cv2.boundingRect(contorno)
        hull = cv2.convexHull(contorno)
        hull_area = cv2.contourArea(hull)
        
        solidez = float(area) / hull_area if hull_area > 0 else 0
        circularidad = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0
        proporcion_aspecto = float(w) / h if h > 0 else 0 # Detecta el alargamiento del Cashew

        # 2. TEXTURA (Rugosidad de la Naranja vs Piel Lisa de Ciruela/Sandía)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_masked = cv2.bitwise_and(gray, gray, mask=mascara_binaria)
        # El filtro Canny detecta los poros y líneas
        bordes = cv2.Canny(gray_masked, 50, 150)
        densidad_bordes = np.sum(bordes > 0) / area if area > 0 else 0

        # 3. COLOR AVANZADO (Tonos, Claridad y Vistosidad)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Ignoramos fondos y sombras profundas
        mask_color_real = (mascara_binaria == 255) & (s > 20) & (v > 20)
        h_fruta = h[mask_color_real]
        s_fruta = s[mask_color_real]
        v_fruta = v[mask_color_real]
        
        total_pixeles = len(h_fruta) if len(h_fruta) > 0 else 1

        # Porcentajes de Tono base
        porc_rojo = np.sum(((h_fruta >= 0) & (h_fruta <= 10)) | (h_fruta >= 165)) / total_pixeles
        porc_naranja = np.sum((h_fruta >= 11) & (h_fruta <= 35)) / total_pixeles
        porc_verde = np.sum((h_fruta >= 36) & (h_fruta <= 85)) / total_pixeles

        # Intensidad y Oscuridad (El secreto para separar Cebolla de Naranja y Ciruela de Rojo)
        saturacion_media = np.mean(s_fruta) if total_pixeles > 1 else 0
        brillo_medio = np.mean(v_fruta) if total_pixeles > 1 else 0

        return solidez, circularidad, proporcion_aspecto, densidad_bordes, porc_rojo, porc_naranja, porc_verde, saturacion_media, brillo_medio
    
    def load_and_process_dataset(self):
        for fruit_class in self.target_classes:
            class_path = os.path.join(self.dataset_path, fruit_class)
            if not os.path.exists(class_path): continue

            for img_name in os.listdir(class_path):
                if img_name.startswith("._"): continue
                img_path = os.path.join(class_path, img_name)
                image = cv2.imread(img_path)
                if image is None: continue

                mascara = self._obtener_mascara_perfecta(image)
                
                if cv2.countNonZero(mascara) > 500:
                    features = self._calcular_caracteristicas_desde_mascara(image, mascara)
                    
                    self.features_data.append({
                        'clase': fruit_class, 'imagen': img_name, 'objeto_id': 1,
                        'caract_1_solidez': features[0], 
                        'caract_2_circularidad': features[1],
                        'caract_3_proporcion': features[2],
                        'caract_4_rugosidad': features[3],
                        'caract_5_porc_rojo': features[4], 
                        'caract_6_porc_naranja': features[5],
                        'caract_7_porc_verde': features[6], 
                        'caract_8_saturacion': features[7], 
                        'caract_9_brillo': features[8]
                    })

    def create_indexed_database(self):
        if not self.features_data: return None
        df = pd.DataFrame(self.features_data)
        self.indexed_db = df.set_index(['clase', 'imagen', 'objeto_id'])
        self.indexed_db.to_csv('base_datos_caracteristicas.csv')
        return self.indexed_db

    def extraer_caracteristicas_imagen_usuario(self, ruta_imagen):
        if not os.path.exists(ruta_imagen): return None
        image = cv2.imread(ruta_imagen)
        if image is None: return None

        mascara = self._obtener_mascara_perfecta(image)
        if cv2.countNonZero(mascara) < 500: return None

        vector = self._calcular_caracteristicas_desde_mascara(image, mascara)
        return list(vector)