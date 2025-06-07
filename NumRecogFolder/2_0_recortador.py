import os
import cv2
import numpy as np
from pathlib import Path

def recortar_imagenes(carpeta_origen, carpeta_destino):
    """
    Recorta todas las imágenes en la carpeta_origen, eliminando el 20% superior e inferior,
    y guarda los resultados en carpeta_destino como archivos PNG.
    """
    # Crear carpeta de destino si no existe
    os.makedirs(carpeta_destino, exist_ok=True)
    
    # Extensiones de imagen comunes
    extensiones = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Obtener todas las imágenes de la carpeta
    imagenes = []
    for ext in extensiones:
        imagenes.extend(list(Path(carpeta_origen).glob(f'*{ext}')))
        imagenes.extend(list(Path(carpeta_origen).glob(f'*{ext.upper()}')))
    
    print(f"Se encontraron {len(imagenes)} imágenes para procesar")
    
    # Procesar cada imagen
    for i, ruta_imagen in enumerate(imagenes):
        try:
            # Cargar la imagen
            imagen = cv2.imread(str(ruta_imagen))
            
            if imagen is None:
                print(f"No se pudo cargar la imagen: {ruta_imagen}")
                continue
            
            # Obtener dimensiones
            altura, ancho = imagen.shape[:2]
            
            # Calcular los puntos de recorte (20% arriba y 20% abajo)
            inicio_y = int(altura * 0.2)
            fin_y = int(altura * 0.5)
            
            # Recortar la imagen
            imagen_recortada = imagen[inicio_y:fin_y, 0:ancho]
            
            # Cambiar la extensión a PNG
            nombre_sin_extension = ruta_imagen.stem
            nombre_archivo_png = f"{nombre_sin_extension}.png"
            ruta_salida = os.path.join(carpeta_destino, nombre_archivo_png)
            
            # Guardar la imagen recortada como PNG
            cv2.imwrite(ruta_salida, imagen_recortada)
            
            print(f"Procesada imagen {i+1}/{len(imagenes)}: {nombre_archivo_png}")
            
        except Exception as e:
            print(f"Error al procesar {ruta_imagen}: {str(e)}")
    
    print(f"\nProceso completado. Las imágenes recortadas se guardaron en formato PNG en: {carpeta_destino}")

# Ejemplo de uso
if __name__ == "__main__":
    carpeta_origen = "NumRecogFolder/1_ImgRecortesOriginales"  # Carpeta con las imágenes originales
    carpeta_destino = "NumRecogFolder/2_0_ImgRecortadas60Central"  # Carpeta donde se guardarán las imágenes recortadas
    
    recortar_imagenes(carpeta_origen, carpeta_destino)
