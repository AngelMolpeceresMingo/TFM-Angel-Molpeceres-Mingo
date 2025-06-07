import cv2
import pytesseract
import os

# Configurar ruta de Tesseract (obligatorio en Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocesamiento_minimo(image_path):
    """Preprocesamiento mínimo - solo redimensionamiento"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    h, w = img.shape
    escala = 600 / max(w, h)
    nuevo_w = int(w * escala)
    nuevo_h = int(h * escala)
    img_redim = cv2.resize(img, (nuevo_w, nuevo_h), interpolation=cv2.INTER_CUBIC)
    
    return img_redim

def reconocer_digito_todos_algoritmos(image_path):
    """Retorna todas las detecciones de números de todos los algoritmos"""
    img_procesada = preprocesamiento_minimo(image_path)
    if img_procesada is None:
        return []
    
    resultados_detallados = []
    
    # 1. Tesseract - Solo las 2 configuraciones más efectivas
    configs_tesseract = [
        (r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789', 'Tesseract OEM3-PSM10'),
        (r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789', 'Tesseract OEM3-PSM8')
    ]
    
    for config, nombre in configs_tesseract:
        try:
            datos = pytesseract.image_to_data(img_procesada, config=config, output_type=pytesseract.Output.DICT)
            texto = pytesseract.image_to_string(img_procesada, config=config)
            
            digito = ''.join(filter(str.isdigit, texto.strip()))
            
            if digito:
                confianzas = [int(conf) for conf in datos['conf'] if int(conf) > 0]
                confianza_promedio = sum(confianzas) / len(confianzas) if confianzas else 0
                
                resultados_detallados.append({
                    'algoritmo': nombre,
                    'digito': digito,
                    'confianza': round(confianza_promedio, 1)
                })
        except Exception:
            continue
    
    # 2. EasyOCR - El algoritmo más efectivo
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        resultados_easy = reader.readtext(img_procesada, 
                                        allowlist='0123456789',
                                        width_ths=0.7,
                                        height_ths=0.7)
        
        for (bbox, text, confidence) in resultados_easy:
            digito = ''.join(filter(str.isdigit, text))
            if digito and confidence > 0.3:
                resultados_detallados.append({
                    'algoritmo': 'EasyOCR',
                    'digito': digito,
                    'confianza': round(confidence * 100, 1)
                })
    except ImportError:
        print("⚠️  EasyOCR no está instalado. Usando solo Tesseract.")
    except Exception:
        pass
    
    return resultados_detallados

def procesar_carpetas_todos_algoritmos(input_folder):
    """Procesa carpetas mostrando todas las detecciones de números"""
    resultados = {}
    
    for root, dirs, _ in os.walk(input_folder):
        for subdir in dirs:
            subcarpeta = os.path.join(root, subdir)
            ruta_relativa = os.path.relpath(subcarpeta, input_folder)
            
            print(f"\n{'='*50}")
            print(f"PROCESANDO: {ruta_relativa}")
            print(f"{'='*50}")
            
            resultados_subcarpeta = {}
            todos_los_numeros = []
            
            for i in range(1, 5):
                archivo = f'componente_{i}.png'
                ruta_imagen = os.path.join(subcarpeta, archivo)
                
                if os.path.exists(ruta_imagen):
                    print(f"\n📁 {archivo}:")
                    
                    detalles = reconocer_digito_todos_algoritmos(ruta_imagen)
                    resultados_subcarpeta[archivo] = detalles
                    
                    if detalles:
                        for detalle in detalles:
                            print(f"   • {detalle['algoritmo']}: '{detalle['digito']}' (confianza: {detalle['confianza']}%)")
                            todos_los_numeros.append(detalle['digito'])
                    else:
                        print(f"   ❌ Ningún algoritmo detectó dígitos")
            
            # Mostrar TODOS los números detectados separados por comas
            print(f"\n📊 RESUMEN DE {ruta_relativa}:")
            if todos_los_numeros:
                numeros_str = ", ".join(todos_los_numeros)
                print(f"   Todos los números detectados: {numeros_str}")
            else:
                print("   No se detectaron números")
            
            if resultados_subcarpeta:
                resultados[ruta_relativa] = resultados_subcarpeta
    
    return resultados

def mostrar_estadisticas_todos(resultados):
    """Muestra estadísticas del procesamiento con todas las detecciones"""
    total_subcarpetas = len(resultados)
    total_imagenes = sum(len(componentes) for componentes in resultados.values())
    imagenes_exitosas = sum(1 for componentes in resultados.values() 
                           for detalles in componentes.values() if detalles)
    
    # Contar total de detecciones
    total_detecciones = sum(len(detalles) for componentes in resultados.values() 
                           for detalles in componentes.values())
    
    print(f"\n{'='*50}")
    print("📈 ESTADÍSTICAS DEL PROCESAMIENTO (TODAS LAS DETECCIONES)")
    print(f"{'='*50}")
    print(f"📂 Subcarpetas procesadas: {total_subcarpetas}")
    print(f"🖼️  Imágenes procesadas: {total_imagenes}")
    print(f"✅ Imágenes con alguna detección: {imagenes_exitosas}")
    print(f"🔢 Total de detecciones de números: {total_detecciones}")
    print(f"📊 Tasa de éxito: {(imagenes_exitosas/total_imagenes*100):.1f}%" if total_imagenes > 0 else "N/A")

# Configuración principal
carpeta_entrada = 'NumRecogFolder/5_ImgNumerosAislados'
archivo_salida = 'NumRecogFolder/resultados_ocr_todos_algoritmos.txt'

# Ejecutar procesamiento
if os.path.exists(carpeta_entrada):
    print("🚀 PROCESAMIENTO CON TODAS LAS DETECCIONES DE NÚMEROS")
    print("🔧 Algoritmos usados: EasyOCR + Tesseract OEM3-PSM10 + Tesseract OEM3-PSM8")
    print("📋 Manteniendo TODAS las detecciones sin selección")
    
    resultados_finales = procesar_carpetas_todos_algoritmos(carpeta_entrada)
    
    # Guardar resultados
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        f.write("RESULTADOS OCR - TODAS LAS DETECCIONES\n")
        f.write("=" * 40 + "\n")
        f.write("Algoritmos: EasyOCR + Tesseract (PSM10/PSM8)\n")
        f.write("Modo: Todas las detecciones (sin selección)\n\n")
        
        for subcarpeta, componentes in resultados_finales.items():
            f.write(f"SUBCARPETA: {subcarpeta}\n")
            f.write("-" * 30 + "\n")
            
            todos_los_numeros = []
            for componente, detalles in componentes.items():
                f.write(f"  {componente}:\n")
                if detalles:
                    for detalle in detalles:
                        f.write(f"    • {detalle['algoritmo']}: {detalle['digito']} (confianza: {detalle['confianza']}%)\n")
                        todos_los_numeros.append(detalle['digito'])
                else:
                    f.write("    No detectado\n")
            
            if todos_los_numeros:
                f.write(f"  Todos los números detectados: {', '.join(todos_los_numeros)}\n")
            f.write("\n")
    
    mostrar_estadisticas_todos(resultados_finales)
    print(f"\n💾 Resultados guardados en: {archivo_salida}")
else:
    print(f"❌ Error: No se encuentra la carpeta {carpeta_entrada}")
