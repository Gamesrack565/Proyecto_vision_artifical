import os
from src.controlador.controlador import MainController
from src.vista.vista import MainWindow
from PyQt6.QtWidgets import QApplication
import sys

def main():
    app = QApplication(sys.argv)
    
    ruta_dataset = r"C:\Users\Lizeth\Varios\tropical-fruits-DB-1024x768"
    
    controlador = MainController(ruta_dataset)
    ventana = MainWindow(controlador)
    ventana.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()