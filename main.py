import os
from src.controlador.controlador import MainController
from src.vista.vista import MainWindow
from PyQt6.QtWidgets import QApplication
import sys

def main():
    app = QApplication(sys.argv)
    
    ruta_dataset = r"C:\Users\Lizeth\Varios\tropical-fruits-DB-1024x768"
    
    controlador = MainController(ruta_dataset)
    
    # ==============================================================
    # MODO EXAMEN
    # ==============================================================
    
    # 1. OBLIGATORIO: Hacemos que la IA "estudie" cargando el CSV primero
    controlador.ejecutar_procesamiento_inicial()
    
    # 2. Ahora sí, aplicamos el examen para generar la matriz
    controlador.generar_matriz_confusion(ruta_carpeta_prueba=ruta_dataset, archivo_salida='mi_matriz_confusion.csv')
    
    # ==============================================================

    # Arrancamos la interfaz gráfica
    ventana = MainWindow(controlador)
    ventana.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()