import cv2
import numpy as np
import pytesseract
import os

# Configurar ruta de Tesseract (obligatorio en Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocesamiento_minimo(image_path):
    """Preprocesamiento mínimo - solo redimensionamiento"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Solo redimensionar manteniendo relación de aspecto
    h, w = img.shape
    escala = 600 / max(w, h)  # Tamaño objetivo para dimensión mayor
    nuevo_w = int(w * escala)
    nuevo_h = int(h * escala)
    img_redim = cv2.resize(img, (nuevo_w, nuevo_h), interpolation=cv2.INTER_CUBIC)
    
    return img_redim

def reconocer_digito_minimo(image_path):
    """Reconocimiento OCR con procesamiento mínimo"""
    try:
        img_procesada = preprocesamiento_minimo(image_path)
        if img_procesada is None:
            return None
        
        # Configuración específica para dígitos individuales
        config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'
        
        texto = pytesseract.image_to_string(img_procesada, config=config)
        digito = ''.join(filter(str.isdigit, texto.strip()))
        
        # Guardar imagen procesada para diagnóstico
        ruta_diagnostico = image_path.replace('.png', '_procesada.png')
        cv2.imwrite(ruta_diagnostico, img_procesada)
        
        return digito if digito else None
    
    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return None

def procesar_carpetas_minimo(input_folder):
    """Procesa todas las subcarpetas con procesamiento mínimo"""
    resultados = {}
    
    for root, dirs, _ in os.walk(input_folder):
        for subdir in dirs:
            subcarpeta = os.path.join(root, subdir)
            ruta_relativa = os.path.relpath(subcarpeta, input_folder)
            
            print(f"\nProcesando: {ruta_relativa}")
            resultados_subcarpeta = {}
            
            for i in range(1, 5):
                archivo = f'componente_{i}.png'
                ruta_imagen = os.path.join(subcarpeta, archivo)
                
                if os.path.exists(ruta_imagen):
                    digito = reconocer_digito_minimo(ruta_imagen)
                    resultados_subcarpeta[archivo] = digito
                    print(f"  {archivo}: {digito if digito else 'No detectado'}")
            
            if resultados_subcarpeta:
                resultados[ruta_relativa] = resultados_subcarpeta
    
    return resultados

# Configuración principal
carpeta_entrada = 'NumRecogFolder/5_ImgNumerosAislados'
archivo_salida = 'NumRecogFolder/resultados_ocr_minimo.txt'

# Ejecutar procesamiento
if os.path.exists(carpeta_entrada):
    resultados_finales = procesar_carpetas_minimo(carpeta_entrada)
    
    # Guardar resultados
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        f.write("RESULTADOS OCR CON PROCESAMIENTO MINIMO\n")
        f.write("=" * 30 + "\n\n")
        
        for subcarpeta, componentes in resultados_finales.items():
            f.write(f"SUBCARPETA: {subcarpeta}\n")
            for componente, digito in componentes.items():
                f.write(f"  {componente}: {digito if digito else 'No detectado'}\n")
            f.write("\n")
    
    print(f"\nProceso completado. Resultados guardados en: {archivo_salida}")
else:
    print(f"Error: No se encuentra la carpeta {carpeta_entrada}")
