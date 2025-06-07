import os
import cv2
import numpy as np

def get_colors_ordered_by_center(image):
    height, width, _ = image.shape
    center = (width // 2, height // 2)
    pixels = image.reshape(-1, 3)
    coords = [(i % width, i // width) for i in range(len(pixels))]
    distances = [np.sqrt((x - center[0])**2 + (y - center[1])**2) for x, y in coords]
    color_dist = list(zip([tuple(p) for p in pixels], distances))
    color_dist.sort(key=lambda x: x[1])  # Ordenar por distancia al centro
    seen = set()
    ordered_colors = []
    for color, dist in color_dist:
        if color not in seen:
            seen.add(color)
            ordered_colors.append(color)
    return ordered_colors

def process_image(input_path, output_folder):
    image = cv2.imread(input_path)
    if image is None:
        print(f"No se pudo leer la imagen: {input_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image_rgb.shape
    colors_ordered = get_colors_ordered_by_center(image_rgb)
    
    binary_images = []
    for color in colors_ordered[:4]:  # Tomar los 4 colores más centrales
        binary_img = np.ones((height, width), dtype=np.uint8) * 255  # Fondo blanco
        mask = np.all(image_rgb == color, axis=2)
        binary_img[mask] = 0  # Color elegido en negro
        binary_images.append(binary_img)
    
    # Si hay menos de 4 colores únicos, rellena con imágenes blancas hasta tener 4
    while len(binary_images) < 4:
        binary_images.append(np.ones((height, width), dtype=np.uint8) * 255)
    
    # Obtener el nombre base del archivo de entrada sin extensión
    base_filename = os.path.basename(input_path)
    image_name_without_ext = os.path.splitext(base_filename)[0]
    
    # Crear la ruta para la subcarpeta específica de esta imagen
    image_specific_output_folder = os.path.join(output_folder, image_name_without_ext)
    
    # Crear la subcarpeta si no existe
    if not os.path.exists(image_specific_output_folder):
        os.makedirs(image_specific_output_folder)
    
    # Guardar cada imagen binaria en la subcarpeta
    for i, binary_img in enumerate(binary_images):
        output_image_filename = f'componente_{i+1}.png'
        output_path = os.path.join(image_specific_output_folder, output_image_filename)
        cv2.imwrite(output_path, binary_img)
    
    print(f"Procesada '{base_filename}': 4 imágenes guardadas en {image_specific_output_folder}")

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            process_image(input_path, output_folder)

# Ejemplo de uso:
input_folder = 'NumRecogFolder/3_1_ImgRecortesColoresMasSimplif'
output_folder = 'NumRecogFolder/4_ImgColoresAislados'

process_folder(input_folder, output_folder)
