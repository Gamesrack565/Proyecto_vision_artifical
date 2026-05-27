import sys
from PyQt6.QtWidgets import QApplication
from src.controlador.controlador import MainController
from src.vista.vista import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # Ruta directa al directorio de frutas tropicales
    ruta_dataset = r"C:\\Users\\Lizeth\\Varios\\tropical-fruits-DB-1024x768"
    
    controlador = MainController(ruta_dataset)
    
    ventana = MainWindow(controlador)
    ventana.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()