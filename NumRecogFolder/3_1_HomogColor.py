import os
import cv2
import numpy as np
from pathlib import Path

COLORS_BASIC = {
    'blanco': [255, 255, 255],
    'negro': [0, 0, 0],
    'rojo': [255, 0, 0],
    'verde': [0, 255, 0],
    'azul': [0, 0, 255],
    'amarillo': [255, 255, 0],
    'naranja': [255, 165, 0],
    'morado': [128, 0, 128],
    'marrón': [139, 69, 19]
}

def find_closest_basic_color(pixel, basic_colors):
    pixel = np.array(pixel, dtype=np.float32)
    min_dist = float('inf')
    closest_color = None
    
    for color_rgb in basic_colors.values():
        color_rgb = np.array(color_rgb, dtype=np.float32)
        dist = np.linalg.norm(pixel - color_rgb)
        if dist < min_dist:
            min_dist = dist
            closest_color = color_rgb
    
    return closest_color

def homogenize_colors_direct(image):
    height, width, channels = image.shape
    pixels = image.reshape(-1, channels)
    
    result_pixels = np.zeros_like(pixels, dtype=np.float32)
    for idx, pixel in enumerate(pixels):
        result_pixels[idx] = find_closest_basic_color(pixel, COLORS_BASIC)
    
    result_image = result_pixels.reshape(height, width, channels)
    
    unique_colors, counts = np.unique(result_image.reshape(-1, channels), axis=0, return_counts=True)
    
    print(f"Combinaciones RGB encontradas:")
    for color, count in zip(unique_colors, counts):
        print(f"RGB{tuple(color.astype(int))}: {count} píxeles")
    
    return result_image.astype(np.uint8)

def process_image(input_path, output_path):
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error leyendo: {input_path}")
        return False
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result_rgb = homogenize_colors_direct(image_rgb)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, result_bgr)
    return True

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if not os.path.exists(input_folder):
        print(f"Carpeta no encontrada: {input_folder}")
        return
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            input_path = os.path.join(input_folder, filename)
            
            # Cambiar la extensión a PNG
            nombre_sin_extension = Path(filename).stem
            nombre_archivo_png = f"{nombre_sin_extension}.png"
            output_path = os.path.join(output_folder, nombre_archivo_png)
            
            print(f"\nProcesando: {filename} -> {nombre_archivo_png}")
            process_image(input_path, output_path)

input_folder = 'NumRecogFolder/3_0_ImgRecortesColoresSimplif'
output_folder = 'NumRecogFolder/3_1_ImgRecortesColoresMasSimplif'

process_folder(input_folder, output_folder)
