import os
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QScrollArea, 
                             QStackedWidget, QProgressBar)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal

class LoadWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(str)
    
    def __init__(self, controller, forzar_recalculo):
        super().__init__()
        self.controller = controller
        self.forzar_recalculo = forzar_recalculo
        
    def run(self):
        self.controller.ejecutar_procesamiento_inicial(callback=self.progress.emit, forzar_recalculo=self.forzar_recalculo)
        self.finished.emit()

class EvalWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(str)

    def __init__(self, controller, dataset_path):
        super().__init__()
        self.controller = controller
        self.dataset_path = dataset_path

    def run(self):
        self.controller.generar_matriz_confusion(
            ruta_carpeta_prueba=self.dataset_path,
            archivo_salida='mi_matriz_confusion.csv',
            callback=self.progress.emit
        )
        self.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Visión Artificial - CBIR Dark UI")
        self.setMinimumSize(1000, 800)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #1A1A24; }
            QLabel { color: #FFFFFF; font-family: 'Segoe UI', sans-serif; }
            QProgressBar { border: none; background-color: #2D2D44; height: 6px; border-radius: 3px; }
            QProgressBar::chunk { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #9D4EDD, stop:1 #00E5FF); border-radius: 3px; }
            QScrollArea { border: none; background-color: transparent; }
            QScrollBar:horizontal { border: none; background: #2D2D44; height: 10px; border-radius: 5px; }
            QScrollBar::handle:horizontal { background: #9D4EDD; border-radius: 5px; }
        """)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.stack.addWidget(self.crear_pantalla_bienvenida()) 
        self.stack.addWidget(self.crear_pantalla_opciones())   
        self.stack.addWidget(self.crear_pantalla_carga())      
        self.stack.addWidget(self.crear_pantalla_principal())  

    def crear_pantalla_bienvenida(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("MOTOR CBIR")
        title.setStyleSheet("font-size: 48px; font-weight: 800; color: #FFFFFF; letter-spacing: 2px;")
        btn_enter = QPushButton("INICIAR SISTEMA")
        btn_enter.setFixedSize(250, 50)
        btn_enter.setStyleSheet("QPushButton { background-color: #9D4EDD; color: #FFFFFF; border: none; border-radius: 8px; font-size: 16px; font-weight: bold; } QPushButton:hover { background-color: #B388FF; }")
        btn_enter.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        layout.addStretch()
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(30)
        layout.addWidget(btn_enter, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        return page

    def crear_pantalla_opciones(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("MÉTODO DE ARRANQUE")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #FFFFFF; margin-bottom: 20px;")
        
        btn_rapido = QPushButton("Usar Base de Datos Indexada (.csv)")
        btn_rapido.setFixedSize(400, 50)
        btn_rapido.setStyleSheet("QPushButton { background-color: #00E5FF; color: #1A1A24; border: none; border-radius: 8px; font-size: 16px; font-weight: bold; } QPushButton:hover { background-color: #66FFFF; }")
        btn_rapido.clicked.connect(lambda: self.iniciar_carga(forzar_recalculo=False))

        btn_lento = QPushButton("Cargar Datos desde 0 (Extraer Nuevamente)")
        btn_lento.setFixedSize(400, 50)
        btn_lento.setStyleSheet("QPushButton { background-color: #FF3366; color: #FFFFFF; border: none; border-radius: 8px; font-size: 16px; font-weight: bold; margin-top: 15px; } QPushButton:hover { background-color: #FF6688; }")
        btn_lento.clicked.connect(lambda: self.iniciar_carga(forzar_recalculo=True))

        layout.addStretch()
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(btn_rapido, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(btn_lento, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        return page

    def crear_pantalla_carga(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.lbl_carga_titulo = QLabel("Inicializando tensores...")
        self.lbl_carga_titulo.setStyleSheet("font-size: 20px; color: #00E5FF; font-weight: bold;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) 
        self.progress_bar.setFixedSize(400, 6)
        self.lbl_carga_pasos = QLabel("Esperando orden...")
        self.lbl_carga_pasos.setStyleSheet("font-size: 14px; color: #8A8A9D; margin-top: 15px;")
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

        top_bar = QHBoxLayout()
        
        self.btn_load = QPushButton("Seleccionar Imagen")
        self.btn_load.setStyleSheet("QPushButton { background-color: #FF3366; color: #FFFFFF; border: none; border-radius: 8px; padding: 10px 20px; font-size: 14px; font-weight: bold; } QPushButton:hover { background-color: #FF6688; } QPushButton:disabled { background-color: #444454; color: #8A8A9D; }")
        self.btn_load.clicked.connect(self.seleccionar_imagen)
        
        self.btn_eval = QPushButton("Evaluar Modelo")
        self.btn_eval.setStyleSheet("QPushButton { background-color: #9D4EDD; color: #FFFFFF; border: none; border-radius: 8px; padding: 10px 20px; font-size: 14px; font-weight: bold; } QPushButton:hover { background-color: #B388FF; } QPushButton:disabled { background-color: #444454; color: #8A8A9D; }")
        self.btn_eval.clicked.connect(self.iniciar_evaluacion)
        
        self.lbl_status = QLabel("Listo para consultas.")
        self.lbl_status.setStyleSheet("color: #00E5FF; font-size: 14px;")
        
        top_bar.addWidget(self.btn_load)
        top_bar.addSpacing(10)
        top_bar.addWidget(self.btn_eval)
        top_bar.addSpacing(15)
        top_bar.addWidget(self.lbl_status)
        top_bar.addStretch()

        self.display_query = QLabel("IMAGEN DE REFERENCIA")
        self.display_query.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display_query.setFixedSize(600, 400) 
        self.display_query.setStyleSheet("QLabel { background-color: #2D2D44; border-radius: 15px; font-size: 24px; font-weight: bold; color: rgba(255, 255, 255, 0.7); }")
        
        bottom_layout = QVBoxLayout()
        self.lbl_prediccion = QLabel("PREDICCIÓN: ____________________")
        self.lbl_prediccion.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 5px;")
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.cards_container = QWidget()
        self.cards_layout = QHBoxLayout(self.cards_container)
        self.cards_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.cards_layout.setSpacing(25)
        self.scroll_area.setWidget(self.cards_container)

        bottom_layout.addWidget(self.lbl_prediccion)
        bottom_layout.addWidget(self.scroll_area)

        main_layout.addLayout(top_bar)
        main_layout.addWidget(self.display_query, alignment=Qt.AlignmentFlag.AlignCenter) 
        main_layout.addLayout(bottom_layout)
        return page

    def iniciar_carga(self, forzar_recalculo):
        self.stack.setCurrentIndex(2)
        self.thread = QThread()
        self.worker = LoadWorker(self.controller, forzar_recalculo)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(lambda txt: self.lbl_carga_pasos.setText(txt))
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.finished.connect(lambda: self.stack.setCurrentIndex(3))
        self.thread.start()

    def iniciar_evaluacion(self):
        self.btn_load.setEnabled(False)
        self.btn_eval.setEnabled(False)
        self.lbl_status.setText("Inicializando examen masivo sobre las imágenes...")

        self.eval_thread = QThread()
        self.eval_worker = EvalWorker(self.controller, self.controller.dataset_path)
        self.eval_worker.moveToThread(self.eval_thread)
        self.eval_thread.started.connect(self.eval_worker.run)
        self.eval_worker.progress.connect(lambda txt: self.lbl_status.setText(txt))
        self.eval_worker.finished.connect(self.eval_thread.quit)
        self.eval_worker.finished.connect(self.eval_worker.deleteLater)
        self.eval_thread.finished.connect(self.eval_thread.deleteLater)

        def restaurar_interfaz():
            self.btn_load.setEnabled(True)
            self.btn_eval.setEnabled(True)

        self.eval_worker.finished.connect(restaurar_interfaz)
        self.eval_thread.start()

    def seleccionar_imagen(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Abrir Imagen", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)")
        if not ruta: 
            return

        pix = QPixmap(ruta)
        self.display_query.setPixmap(pix.scaled(self.display_query.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        def status_callback(msg):
            self.lbl_status.setText(msg)

        pred, top5, error = self.controller.procesar_consulta(ruta, callback=status_callback)
        self.mostrar_resultados(pred, top5)

    def mostrar_resultados(self, prediccion, resultados_top5):
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget(): 
                item.widget().deleteLater()

        if "detectado" in prediccion or "No" in prediccion:
            self.lbl_prediccion.setText(f"PREDICCIÓN: BLOQUEADO - {prediccion}")
            self.lbl_prediccion.setStyleSheet("font-size: 20px; font-weight: bold; color: #FF3366;")
            self.lbl_status.setText("Acceso Restringido.")
            return

        self.lbl_prediccion.setText(f"PREDICCIÓN: {prediccion.upper()}")
        self.lbl_prediccion.setStyleSheet("font-size: 20px; font-weight: bold; color: #FFFFFF;")
        self.lbl_status.setText("Consulta finalizada con éxito.")

        for res in resultados_top5:
            card = QWidget()
            card.setFixedSize(160, 220)
            card.setStyleSheet("QWidget { background-color: #2D2D44; border-radius: 10px; }")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 10, 10, 10)

            lbl_img = QLabel()
            lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_img.setFixedSize(140, 140)
            lbl_img.setStyleSheet("background-color: #1A1A24; border-radius: 5px;")

            ruta_recuperada = os.path.join(self.controller.dataset_path, res['clase_recuperada'], res['nombre_imagen'])
            if os.path.exists(ruta_recuperada):
                p_res = QPixmap(ruta_recuperada)
                lbl_img.setPixmap(p_res.scaled(lbl_img.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            else:
                lbl_img.setText("N/A")

            lbl_name = QLabel(res['nombre_imagen'])
            lbl_name.setStyleSheet("font-size: 11px; color: #8A8A9D;")
            lbl_name.setAlignment(Qt.AlignmentFlag.AlignCenter)

            lbl_err = QLabel(f"Error: {res['indice_error']:.4f}")
            lbl_err.setStyleSheet("font-size: 11px; color: #00E5FF; font-weight: bold;")
            lbl_err.setAlignment(Qt.AlignmentFlag.AlignCenter)

            card_layout.addWidget(lbl_img)
            card_layout.addWidget(lbl_name)
            card_layout.addWidget(lbl_err)
            self.cards_layout.addWidget(card)