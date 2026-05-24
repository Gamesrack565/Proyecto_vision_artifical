import sys
import os
from PyQt6.QtWidgets import QApplication
from src.controlador.controlador import MainController
from src.vista.vista import MainWindow # Importa tu archivo vista.py

def main():
    app = QApplication(sys.argv)
    
    # Asegúrate de que esta ruta sea correcta para tu sistema
    ruta_dataset = os.path.join(os.getcwd(), 'Dataset/tropical-fruits-DB-1024x768')
    
    # Instanciamos el controlador que creamos antes
    controlador = MainController(ruta_dataset)
    
    # Creamos y mostramos la vista
    ventana = MainWindow(controlador)
    ventana.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()