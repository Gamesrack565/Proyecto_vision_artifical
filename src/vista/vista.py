import os
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QStackedWidget, QFileDialog, 
                             QScrollArea, QFrame, QProgressBar, QApplication)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal


# --- HILO PARA CARGA EN SEGUNDO PLANO ---
class LoadWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(str) # Señal para enviar el texto de los pasos
    
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        
    def run(self):
        # Pasamos la señal 'progress.emit' como callback al controlador
        self.controller.ejecutar_procesamiento_inicial(callback=self.progress.emit)
        self.finished.emit()

# --- VENTANA PRINCIPAL ---
class MainWindow(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Visión Artificial - CBIR Minimalista")
        self.setMinimumSize(1200, 700)
        
        # Diseño Minimalista Global (Fondo lila pastel muy suave)
        self.setStyleSheet("""
            QMainWindow { background-color: #F8F6FA; }
            QLabel { color: #333333; font-family: 'Segoe UI', Arial, sans-serif; }
            QPushButton {
                background-color: #F4D03F; color: #2C3E50;
                border: none; border-radius: 6px; padding: 12px 20px;
                font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #F7DC6F; }
            QProgressBar {
                border: none; background-color: #E8E2EE; height: 4px; border-radius: 2px;
            }
            QProgressBar::chunk { background-color: #9B59B6; border-radius: 2px; }
            QScrollArea { border: none; background: transparent; }
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

        title = QLabel("Clasificador de Frutas CBIR")
        title.setStyleSheet("font-size: 40px; font-weight: 300; color: #5B2C6F; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle = QLabel("Proyecto de Visión Artificial\nDesarrollado por:")
        subtitle.setStyleSheet("font-size: 16px; color: #7F8C8D; line-height: 1.5;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        btn_enter = QPushButton("Entrar al Sistema")
        btn_enter.setFixedSize(220, 50)
        btn_enter.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_enter.clicked.connect(self.iniciar_carga)

        layout.addStretch()
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(40)
        layout.addWidget(btn_enter, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        return page

    def crear_pantalla_carga(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.lbl_carga_titulo = QLabel("Preparando Motor de Visión...")
        self.lbl_carga_titulo.setStyleSheet("font-size: 24px; color: #5B2C6F; font-weight: 300;")
        
        # Barra de progreso indeterminada (minimalista)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) # Rango 0,0 la hace moverse infinitamente
        self.progress_bar.setFixedSize(400, 4)

        # Label dinámico para mostrar los pasos del controlador
        self.lbl_carga_pasos = QLabel("Iniciando...")
        self.lbl_carga_pasos.setStyleSheet("font-size: 14px; color: #7F8C8D; margin-top: 20px;")
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
        main_layout = QHBoxLayout(page)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(40)

        # --- PANEL IZQUIERDO (Carga de imagen) ---
        left_panel = QVBoxLayout()
        left_panel.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        btn_load = QPushButton("Seleccionar Imagen")
        btn_load.clicked.connect(self.seleccionar_imagen)
        
        # Contenedor blanco minimalista para la imagen
        self.display_query = QLabel("Ninguna imagen seleccionada")
        self.display_query.setFixedSize(400, 400)
        self.display_query.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display_query.setStyleSheet("""
            background-color: #FFFFFF;
            border: 1px solid #D1C4E9;
            border-radius: 8px;
            color: #BDC3C7;
        """)
        
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #9B59B6; font-size: 14px; margin-top: 10px;")
        
        left_panel.addWidget(btn_load)
        left_panel.addSpacing(15)
        left_panel.addWidget(self.display_query)
        left_panel.addWidget(self.lbl_status)
        left_panel.addStretch()

        # --- PANEL DERECHO (Resultados) ---
        right_panel = QVBoxLayout()
        
        header_layout = QHBoxLayout()
        title_res = QLabel("Resultados de Similitud")
        title_res.setStyleSheet("font-size: 20px; color: #5B2C6F; font-weight: bold;")
        
        btn_home = QPushButton("Inicio")
        btn_home.setFixedSize(100, 40)
        btn_home.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        btn_home.setStyleSheet("background-color: #E8E2EE; color: #5B2C6F;") # Botón secundario
        
        header_layout.addWidget(title_res)
        header_layout.addStretch()
        header_layout.addWidget(btn_home)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.res_container = QWidget()
        self.res_layout = QVBoxLayout(self.res_container)
        self.res_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.res_layout.setSpacing(15)
        self.scroll.setWidget(self.res_container)

        right_panel.addLayout(header_layout)
        right_panel.addSpacing(15)
        right_panel.addWidget(self.scroll)

        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 1)
        return page

    # --- LÓGICA DE LA INTERFAZ ---

    def iniciar_carga(self):
        self.stack.setCurrentIndex(1)
        
        self.thread = QThread()
        self.worker = LoadWorker(self.controller)
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        # Actualizar el texto de la vista con la señal del worker
        self.worker.progress.connect(lambda txt: self.lbl_carga_pasos.setText(txt))
        
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.finished.connect(lambda: self.stack.setCurrentIndex(2))
        
        self.thread.start()

    def seleccionar_imagen(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Cargar Imagen", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if ruta:
            # Mostrar imagen
            pix = QPixmap(ruta).scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.display_query.setPixmap(pix)
            
            # Limpiar resultados
            for i in reversed(range(self.res_layout.count())): 
                widget = self.res_layout.itemAt(i).widget()
                if widget: widget.setParent(None)
            
            # Actualizar status a tiempo real mediante callback
            self.lbl_status.setText("Procesando imagen...")
            QApplication.processEvents()
            
            def status_callback(msg):
                self.lbl_status.setText(msg)
                QApplication.processEvents()
                
            pred, top5, error = self.controller.procesar_consulta(ruta, callback=status_callback)
            self.lbl_status.setText("Consulta finalizada.")
            self.mostrar_resultados(pred, top5)

    def mostrar_resultados(self, pred, top5):
        if not top5:
            # Diseño de alerta minimalista
            frame_alerta = QFrame()
            frame_alerta.setStyleSheet("background-color: #FDEDEC; border-radius: 8px; padding: 20px;")
            lay_alerta = QVBoxLayout(frame_alerta)
            lbl = QLabel(f"BLOQUEO DE SEGURIDAD\n\n{pred}")
            lbl.setStyleSheet("color: #E74C3C; font-weight: bold; font-size: 16px;")
            lay_alerta.addWidget(lbl)
            self.res_layout.addWidget(frame_alerta)
            return

        # Etiqueta de predicción principal
        lbl_pred = QLabel(f"Predicción Principal:  {pred.upper()}")
        lbl_pred.setStyleSheet("font-size: 16px; color: #27AE60; font-weight: bold; margin-bottom: 5px;")
        self.res_layout.addWidget(lbl_pred)

        # Tarjetas de resultados minimalistas
        for r in top5:
            card = QFrame()
            card.setStyleSheet("background-color: #FFFFFF; border: 1px solid #E8E2EE; border-radius: 8px;")
            card.setFixedHeight(100)
            lay = QHBoxLayout(card)
            lay.setContentsMargins(10, 10, 10, 10)
            
            img_path = os.path.join(self.controller.dataset_path, r['clase_recuperada'], r['nombre_imagen'])
            img_lbl = QLabel()
            if os.path.exists(img_path):
                img_lbl.setPixmap(QPixmap(img_path).scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            else:
                img_lbl.setText("No Img")
                
            info_lay = QVBoxLayout()
            lbl_name = QLabel(f"{r['nombre_imagen']}")
            lbl_name.setStyleSheet("font-weight: bold; font-size: 14px;")
            lbl_err = QLabel(f"Error (Distancia): {r['indice_error']}")
            lbl_err.setStyleSheet("color: #7F8C8D; font-size: 12px;")
            
            info_lay.addWidget(lbl_name)
            info_lay.addWidget(lbl_err)
            info_lay.addStretch()
            
            lay.addWidget(img_lbl)
            lay.addLayout(info_lay)
            lay.addStretch()
            
            self.res_layout.addWidget(card)
        
        self.res_layout.addStretch()