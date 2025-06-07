import os
import cv2
import numpy as np

def filter_largest_black_groups(binary_img, max_groups=2, height_tolerance=20, max_area_percentage=0.4, min_column_threshold=0.8):
    height, width = binary_img.shape
    visited = np.zeros_like(binary_img, dtype=bool)
    groups = []
    total_pixels = height * width

    def neighbors(x, y):
        for nx, ny in [(x-1,y),(x+1,y),(x,y-1),(x,y+1), (x-1,y-1), (x-1,y+1), (x+1,y-1), (x+1,y+1)]:
            if 0 <= nx < width and 0 <= ny < height:
                yield nx, ny

    def get_group_center_y(group_pixels):
        y_coords = [y for (x, y) in group_pixels]
        return sum(y_coords) / len(y_coords)

    def groups_at_similar_height(group1, group2, tolerance):
        center_y1 = get_group_center_y(group1)
        center_y2 = get_group_center_y(group2)
        return abs(center_y1 - center_y2) <= tolerance

    def is_valid_group(group_pixels):
        group_size = len(group_pixels)
        if group_size > (total_pixels * max_area_percentage):
            return False
        group_coords = set(group_pixels)
        x_coords = set(x for (x, y) in group_pixels)
        for x in x_coords:
            column_pixels_in_group = sum(1 for y in range(height) if (x, y) in group_coords)
            if column_pixels_in_group > (height * min_column_threshold):
                return False
        # Exclude groups touching the image borders
        for (x, y) in group_pixels:
            if x == 0 or x == width - 1 or y == 0 or y == height - 1:
                return False
        return True

    for y in range(height):
        for x in range(width):
            if binary_img[y, x] == 0 and not visited[y, x]:
                queue = [(x, y)]
                group_pixels = []
                visited[y, x] = True
                while queue:
                    cx, cy = queue.pop(0)
                    group_pixels.append((cx, cy))
                    for nx, ny in neighbors(cx, cy):
                        if binary_img[ny, nx] == 0 and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((nx, ny))
                groups.append(group_pixels)

    valid_groups = [group for group in groups if is_valid_group(group)]
    valid_groups.sort(key=len, reverse=True)
    selected_groups = []
    if len(valid_groups) > 0:
        selected_groups.append(valid_groups[0])
        if len(valid_groups) > 1 and max_groups > 1:
            for i in range(1, len(valid_groups)):
                if groups_at_similar_height(selected_groups[0], valid_groups[i], height_tolerance):
                    selected_groups.append(valid_groups[i])
                    break
    filtered_img = np.ones_like(binary_img) * 255
    for group in selected_groups:
        for (x, y) in group:
            filtered_img[y, x] = 0
    return filtered_img

def process_component(input_image_path, output_image_path):
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"No se pudo leer la imagen: {input_image_path}")
        return
    filtered_img = filter_largest_black_groups(
        img,
        max_groups=2,
        height_tolerance=30,
        max_area_percentage=0.5,
        min_column_threshold=0.8
    )
    cv2.imwrite(output_image_path, filtered_img)

def process_image_folder(input_subfolder, output_subfolder):
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)
    for i in range(1, 5):  # Asume 4 componentes: componente_1.png, ..., componente_4.png
        input_path = os.path.join(input_subfolder, f'componente_{i}.png')
        output_path = os.path.join(output_subfolder, f'componente_{i}.png')
        if os.path.exists(input_path):
            process_component(input_path, output_path)
            print(f"Procesada {input_path} → {output_path}")
        else:
            print(f"¡ADVERTENCIA! No existe {input_path}")

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for root, dirs, _ in os.walk(input_folder):
        for subdir in dirs:
            input_subfolder = os.path.join(root, subdir)
            relative_path = os.path.relpath(input_subfolder, input_folder)
            output_subfolder = os.path.join(output_folder, relative_path)
            process_image_folder(input_subfolder, output_subfolder)

# Ejemplo de uso:
input_folder = 'NumRecogFolder/4_ImgColoresAislados'
output_folder = 'NumRecogFolder/5_ImgNumerosAislados'

process_folder(input_folder, output_folder)