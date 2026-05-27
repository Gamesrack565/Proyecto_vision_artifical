import sys
from PyQt6.QtWidgets import QApplication
from src.controlador.controlador import MainController
from src.vista.vista import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # Ruta directa a tu base de datos de imágenes
    ruta_dataset = r"C:\Users\Lizeth\Varios\tropical-fruits-DB-1024x768"
    
    controlador = MainController(ruta_dataset)
    
    # El arranque ahora es inmediato y limpio
    ventana = MainWindow(controlador)
    ventana.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()