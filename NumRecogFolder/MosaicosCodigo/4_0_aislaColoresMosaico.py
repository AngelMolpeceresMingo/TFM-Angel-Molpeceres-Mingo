import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_colors_ordered_by_center(image):
    height, width, _ = image.shape
    center = (width // 2, height // 2)
    pixels = image.reshape(-1, 3)
    coords = [(i % width, i // width) for i in range(len(pixels))]
    distances = [np.sqrt((x - center[0])**2 + (y - center[1])**2) for x, y in coords]
    color_dist = list(zip([tuple(p) for p in pixels], distances))
    color_dist.sort(key=lambda x: x[1])
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
    for color in colors_ordered[:4]:
        binary_img = np.ones((height, width), dtype=np.uint8) * 255  # Fondo blanco
        mask = np.all(image_rgb == color, axis=2)
        binary_img[mask] = 0  # Color elegido en negro
        binary_images.append(binary_img)
    
    # Si hay menos de 4 colores, rellena con imágenes blancas
    while len(binary_images) < 4:
        binary_images.append(np.ones((height, width), dtype=np.uint8) * 255)
    
    # Crear mosaico 1x4
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i, binary_img in enumerate(binary_images):
        axs[i].imshow(binary_img, cmap='gray')
        axs[i].axis('off')
    plt.tight_layout()
    
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_folder, f'mosaic_{filename}')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Procesada y guardada: {output_path}")

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            process_image(input_path, output_folder)

input_folder = 'player_dataset/3_ImgRecortesColoresSimplif2'
output_folder = 'player_dataset/4_ImgColoresAislados2222'

# Ejecutar el procesamiento
process_folder(input_folder, output_folder)
