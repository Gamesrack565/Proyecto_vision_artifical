import os
# --- IMPORTACIONES DE LA VISTA (Asegúrate de tenerlas arriba) ---
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QStackedWidget, QFileDialog, 
                             QFrame, QProgressBar, QApplication, QSizePolicy, QScrollArea)
from PyQt6.QtGui import QPixmap, QColor, QPalette
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal
import os

# --- HILO PARA CARGA EN SEGUNDO PLANO (Se queda igual) ---
class LoadWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(str)
    
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        
    def run(self):
        self.controller.ejecutar_procesamiento_inicial(callback=self.progress.emit)
        self.finished.emit()

# --- VENTANA PRINCIPAL REDISEÑADA ---
class MainWindow(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Visión Artificial - CBIR Dark UI")
        self.setMinimumSize(1000, 800)
        
        # --- TEMA OSCURO GLOBAL ---
        self.setStyleSheet("""
            QMainWindow { background-color: #1A1A24; }
            QLabel { color: #FFFFFF; font-family: 'Segoe UI', Helvetica, sans-serif; }
            QPushButton {
                background-color: #2D2D44; 
                color: #FFFFFF;
                border: 1px solid #4A4A6A; 
                border-radius: 8px; 
                padding: 10px 20px;
                font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #3D3D5C; border: 1px solid #00E5FF; }
            QProgressBar {
                border: none; background-color: #2D2D44; height: 6px; border-radius: 3px;
            }
            QProgressBar::chunk { 
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #9D4EDD, stop:1 #00E5FF); 
                border-radius: 3px; 
            }
            QScrollArea { border: none; background-color: transparent; }
            QScrollBar:horizontal {
                border: none;
                background: #2D2D44;
                height: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal {
                background: #9D4EDD;
                border-radius: 5px;
            }
        """)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.stack.addWidget(self.crear_pantalla_bienvenida())
        self.stack.addWidget(self.crear_pantalla_carga())
        self.stack.addWidget(self.crear_pantalla_principal())

    def crear_pantalla_bienvenida(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("MOTOR CBIR")
        title.setStyleSheet("font-size: 48px; font-weight: 800; color: #FFFFFF; letter-spacing: 2px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        btn_enter = QPushButton("INICIAR SISTEMA")
        btn_enter.setFixedSize(250, 50)
        btn_enter.setStyleSheet("background-color: #9D4EDD; border: none; font-size: 16px;")
        btn_enter.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_enter.clicked.connect(self.iniciar_carga)

        layout.addStretch()
        layout.addWidget(title)
        layout.addSpacing(30)
        layout.addWidget(btn_enter, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        return page

    def crear_pantalla_carga(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.lbl_carga_titulo = QLabel("Inicializando tensores...")
        self.lbl_carga_titulo.setStyleSheet("font-size: 20px; color: #00E5FF; font-weight: bold;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) 
        self.progress_bar.setFixedSize(400, 6)

        self.lbl_carga_pasos = QLabel("Cargando base de datos...")
        self.lbl_carga_pasos.setStyleSheet("font-size: 14px; color: #8A8A9D; margin-top: 15px;")
        self.lbl_carga_pasos.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addStretch()
        layout.addWidget(self.lbl_carga_titulo, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(20)
        layout.addWidget(self.progress_bar, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_carga_pasos, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        return page

    def crear_pantalla_principal(self):
        page = QWidget()
        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(40, 20, 40, 40)
        main_layout.setSpacing(20)

        # --- BARRA SUPERIOR ---
        top_bar = QHBoxLayout()
        btn_load = QPushButton("Seleccionar Imagen")
        btn_load.clicked.connect(self.seleccionar_imagen)
        
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #00E5FF; font-size: 14px;")
        
        top_bar.addWidget(btn_load)
        top_bar.addSpacing(15)
        top_bar.addWidget(self.lbl_status)
        top_bar.addStretch()

        # --- SECCIÓN SUPERIOR: Imagen de Referencia (Centrada y Grande) ---
        self.display_query = QLabel("IMAGEN DE REFERENCIA")
        self.display_query.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Asignamos un tamaño fijo bastante grande para que luzca bien
        self.display_query.setFixedSize(600, 400) 
        self.display_query.setStyleSheet("""
            QLabel {
                background-color: #2D2D44;
                border-radius: 15px;
                font-size: 24px;
                font-weight: bold;
                color: rgba(255, 255, 255, 0.7);
            }
        """)
        
        # --- SECCIÓN INFERIOR: Resultados con Scroll ---
        bottom_layout = QVBoxLayout()
        
        self.lbl_prediccion = QLabel("PREDICCIÓN: ____________________")
        self.lbl_prediccion.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 5px;")
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        self.cards_container = QWidget()
        self.cards_container.setStyleSheet("background-color: transparent;")
        self.cards_layout = QHBoxLayout(self.cards_container)
        self.cards_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.cards_layout.setSpacing(25)
        
        self.scroll_area.setWidget(self.cards_container)

        bottom_layout.addWidget(self.lbl_prediccion)
        bottom_layout.addWidget(self.scroll_area)

        # Ensamblar layout principal
        main_layout.addLayout(top_bar)
        # ALIGN CENTER: Esto evita que se cree la barra negra gigante y la pone al centro
        main_layout.addWidget(self.display_query, alignment=Qt.AlignmentFlag.AlignCenter) 
        main_layout.addLayout(bottom_layout)

        return page

    # --- LÓGICA DE LA INTERFAZ ---

    def iniciar_carga(self):
        self.stack.setCurrentIndex(1)
        
        self.thread = QThread()
        self.worker = LoadWorker(self.controller)
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(lambda txt: self.lbl_carga_pasos.setText(txt))
        
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.finished.connect(lambda: self.stack.setCurrentIndex(2))
        
        self.thread.start()

    def seleccionar_imagen(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Cargar Imagen", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if ruta:
            pix = QPixmap(ruta)
            # Escalamos la imagen basándonos en el nuevo tamaño fijo (600x400)
            scaled_pix = pix.scaled(600, 400, 
                                    Qt.AspectRatioMode.KeepAspectRatio, 
                                    Qt.TransformationMode.SmoothTransformation)
            self.display_query.setPixmap(scaled_pix)
            
            # Dejamos el fondo transparente para que la imagen se vea limpia y sin bordes negros
            self.display_query.setStyleSheet("background-color: transparent;")
            
            # Limpiar layout de resultados anteriores dinámicamente
            self.lbl_prediccion.setText("PREDICCIÓN: Procesando...")
            for i in reversed(range(self.cards_layout.count())): 
                widget = self.cards_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            
            self.lbl_status.setText("Segmentando imagen...")
            QApplication.processEvents()
            
            def status_callback(msg):
                self.lbl_status.setText(msg)
                QApplication.processEvents()
                
            pred, top5, error = self.controller.procesar_consulta(ruta, callback=status_callback)
            self.lbl_status.setText("Consulta finalizada.")
            self.mostrar_resultados(pred, top5)

    def mostrar_resultados(self, pred, top5):
        if not top5:
            self.lbl_prediccion.setText(f"PREDICCIÓN: BLOQUEADO - {pred}")
            self.lbl_prediccion.setStyleSheet("font-size: 20px; font-weight: bold; color: #FF3366;")
            return

        self.lbl_prediccion.setText(f"PREDICCIÓN: {pred.upper()}")
        self.lbl_prediccion.setStyleSheet("font-size: 20px; font-weight: bold; color: #FFFFFF;")

        # Rellenar tarjetas dinámicamente para TODAS las imágenes en top5
        for r in top5:
            card_widget = QWidget()
            card_col = QVBoxLayout(card_widget)
            card_col.setAlignment(Qt.AlignmentFlag.AlignTop)
            card_col.setContentsMargins(0, 0, 0, 0)
            
            img_lbl = QLabel()
            img_lbl.setFixedSize(200, 200) # Tamaño fijo para las miniaturas
            img_lbl.setStyleSheet("background-color: #2D2D44; border-radius: 15px;")
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            img_path = os.path.join(self.controller.dataset_path, r['clase_recuperada'], r['nombre_imagen'])
            if os.path.exists(img_path):
                # Usamos KeepAspectRatioByExpanding para que las miniaturas sí sean cuadradas perfectas
                pix = QPixmap(img_path).scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
                img_lbl.setPixmap(pix)
            else:
                img_lbl.setText("Sin Imagen")

            txt_lbl = QLabel(f"{r['nombre_imagen']}\nError: {r['indice_error']:.4f}")
            txt_lbl.setStyleSheet("font-size: 12px; font-weight: bold; color: #A0A0B5; margin-top: 5px;")
            txt_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

            card_col.addWidget(img_lbl)
            card_col.addWidget(txt_lbl)
            
            self.cards_layout.addWidget(card_widget)