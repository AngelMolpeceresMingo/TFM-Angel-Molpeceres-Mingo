import os
import subprocess
import glob
import time

def localizar_jugadores_imagenes(input_folder, detection_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Analizando carpeta: {input_folder}")
    
    img1_folder = os.path.join(input_folder, "img1")
    
    if not os.path.exists(img1_folder):
        print(f"ERROR: No se encontró carpeta img1 en {input_folder}")
        return False
    
    jpg_files = glob.glob(os.path.join(img1_folder, "*.jpg"))
    if not jpg_files:
        print(f"ERROR: No se encontraron imágenes JPG en {img1_folder}")
        return False
    
    if not os.path.exists(detection_file):
        print(f"ERROR: No se encontró archivo de detecciones: {detection_file}")
        return False
    
    num_images = len(jpg_files)
    print(f"Imágenes encontradas: {num_images} en {img1_folder}")
    print(f"Archivo detecciones: {detection_file}")
    
    output_video = os.path.join(output_folder, "jugadores_localizados.mp4")
    
    print(f"Iniciando localización de jugadores...")
    
    cmd = [
        "python", "PnLCalib/2.1-ExtraeCoordLineas.py",
        "--weights_kp", "SV_kp",
        "--weights_line", "SV_lines", 
        "--pnl_refine",
        "--device", "cpu",
        "--input_folder", img1_folder,
        "--detection_file", detection_file,
        "--save_path", output_video
    ]
    
    print(f"Ejecutando: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    if result.returncode == 0:
        print(f"EXITOSO: Procesamiento completado en {processing_time:.1f} segundos")
        print(f"Video guardado en: {output_video}")
        
        positions_file = output_video.replace('.mp4', '_positions.txt')
        if os.path.exists(positions_file):
            print(f"Posiciones guardadas en: {positions_file}")
            
        return True
    else:
        print(f"ERROR en localización:")
        print(f"Stderr: {result.stderr}")
        if result.stdout:
            print(f"Stdout: {result.stdout}")
        return False

def verificar_estructura_snmot(carpeta):
    img1_path = os.path.join(carpeta, "img1")
    det_path = os.path.join(carpeta, "det")
    detection_file = os.path.join(det_path, "Newdet.txt")
    
    print(f"Verificando estructura de: {carpeta}")
    
    if not os.path.exists(img1_path):
        print(f"ERROR: Falta carpeta img1")
        return False
    else:
        jpg_count = len(glob.glob(os.path.join(img1_path, "*.jpg")))
        print(f"OK: img1 encontrada con {jpg_count} imágenes JPG")
        if jpg_count == 0:
            print(f"ERROR: No hay imágenes JPG en img1")
            return False
    
    if not os.path.exists(det_path):
        print(f"ERROR: Falta carpeta det")
        return False
    else:
        print(f"OK: det encontrada")
    
    if not os.path.exists(detection_file):
        print(f"ERROR: Falta archivo det/Newdet.txt")
        return False
    else:
        try:
            with open(detection_file, 'r') as f:
                lines = f.readlines()
            print(f"OK: Newdet.txt encontrado con {len(lines)} líneas")
        except Exception as e:
            print(f"ERROR: Error leyendo Newdet.txt: {e}")
            return False
    
    print(f"RESULTADO: Estructura SNMOT válida")
    return True

def procesar_carpeta_individual_imagenes(input_path, output_path):
    print("LOCALIZADOR DE JUGADORES EN SECUENCIAS DE IMAGENES")
    print("=" * 55)
    print(f"Carpeta entrada: {input_path}")
    print(f"Carpeta salida: {output_path}")
    
    detection_file = os.path.join(input_path, "det", "Newdet.txt")
    print(f"Archivo detecciones: {detection_file}")
    print("=" * 55)
    
    return localizar_jugadores_imagenes(input_path, detection_file, output_path)

if __name__ == '__main__':
    input_path = "Ruta/A/La/Carpeta/De/La/Secuencia"
    output_path = "Ruta/A/La/Carpeta/De/Salida"

    resultado = procesar_carpeta_individual_imagenes(input_path, output_path)