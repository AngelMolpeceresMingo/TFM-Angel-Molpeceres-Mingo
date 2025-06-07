import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path

# Definir colores básicos con sus valores RGB
COLORS_BASIC = {
    'blanco': [255, 255, 255],
    'negro': [0, 0, 0],
    'rojo': [255, 0, 0],
    'verde_cesped': [34, 139, 34],      # Verde césped principal
    'verde': [0, 128, 0],               # Verde estándar
    'verde_claro': [50, 205, 50],       # Verde claro
    'azul': [0, 0, 255],
    'amarillo': [255, 255, 0],
    'naranja': [255, 165, 0],
    'morado': [128, 0, 128],
    'marrón': [139, 69, 19],
    'gris': [128, 128, 128]
}

# Umbral de similitud de color (distancia euclidiana en espacio RGB)
SIMILARITY_THRESHOLD = 200  # 200 Ajustado para generalizar más los colores

def is_green_dominant(pixel):
    """Verifica si el canal verde es dominante en el píxel."""
    return pixel[1] > pixel[0] and pixel[1] > pixel[2] and pixel[1] > 60

def find_closest_basic_color(pixel, basic_colors, threshold):
    """Encuentra el color básico más cercano con detección mejorada de verdes."""
    pixel = np.array(pixel, dtype=np.float32)
    
    # Si es verde dominante, asignar directamente verde césped
    if is_green_dominant(pixel):
        return np.array(basic_colors['verde_cesped'], dtype=np.float32)
    
    min_dist = float('inf')
    closest_color = None
    
    for color_name, color_rgb in basic_colors.items():
        if 'verde' in color_name:  # Saltar verdes ya manejados
            continue
        color_rgb = np.array(color_rgb, dtype=np.float32)
        dist = np.sqrt(np.sum((pixel - color_rgb) ** 2))
        if dist < min_dist and dist < threshold:
            min_dist = dist
            closest_color = color_rgb
            
    return closest_color if closest_color is not None else pixel

def homogenize_colors(image):
    """Aplica homogeneización de colores a una imagen basada en colores básicos."""
    height, width, channels = image.shape
    pixels = image.reshape(-1, channels)
    
    kmeans = KMeans(n_clusters=8, random_state=0).fit(pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    quantized = centers[labels].reshape(height, width, channels)
    
    result = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            pixel = quantized[i, j]
            homogenized_color = find_closest_basic_color(pixel, COLORS_BASIC, SIMILARITY_THRESHOLD)
            result[i, j] = homogenized_color
            
    return result.astype(np.uint8)

def process_image(input_path, output_path):
    """Procesa una imagen individual y guarda el resultado."""
    image = cv2.imread(input_path)
    if image is None:
        print(f"No se pudo leer la imagen: {input_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result_rgb = homogenize_colors(image_rgb)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, result_bgr)
    print(f"Procesada: {input_path} -> {output_path}")

def process_folder(input_folder, output_folder):
    """Procesa todas las imágenes en una carpeta de entrada y guarda los resultados en una carpeta de salida como PNG."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            
            # Cambiar la extensión a PNG
            nombre_sin_extension = Path(filename).stem
            nombre_archivo_png = f"{nombre_sin_extension}.png"
            output_path = os.path.join(output_folder, nombre_archivo_png)
            
            process_image(input_path, output_path)

# Parámetros de ejemplo para ejecución
input_folder = 'NumRecogFolder/2_2_ImgColoresExagerados'
output_folder = 'NumRecogFolder/3_0_ImgRecortesColoresSimplif'

# Ejecutar el procesamiento
process_folder(input_folder, output_folder)
