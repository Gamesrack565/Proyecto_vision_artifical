import os
from src.controlador.controlador import MainController

def main():
    # Define la ruta a tu dataset. Ajusta esta variable según tu sistema.
    # Asumiendo que la carpeta 'dataset' está en el mismo directorio que main.py
    ruta_dataset = os.path.join(os.getcwd(), 'Dataset/tropical-fruits-DB-1024x768')
    
    # Instanciar el controlador
    app_controller = MainController(dataset_path=ruta_dataset)
    
    # Ejecutar la lógica de extracción y creación de la BD
    app_controller.ejecutar_procesamiento_inicial()

if __name__ == "__main__":
    main()