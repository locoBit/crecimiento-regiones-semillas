# Importamos las librerías necesarias para el procesamiento de imágenes
import cv2
import numpy as np
from collections import deque
import matplotlib # Necesario para la visualización de resultados
# Usamos el backend 'Agg' de Matplotlib para poder generar y guardar las imágenes 
# sin necesidad de una interfaz gráfica interactiva (evitando errores de Tkinter).
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def region_growing_global_v3(image_path, threshold):
    """
    Función principal que implementa el Crecimiento de Regiones de manera global.
    Busca todas las regiones de la imagen.
    """
    # Leo la imagen desde la ruta especificada y la convierto directamente a escala de grises.
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Verifico si la imagen se cargó correctamente.
    if img is None:
        print(f"Error: No se pudo cargar la imagen en {image_path}")
        return None

    # Obtengo las dimensiones (filas y columnas) de la imagen.
    rows, cols = img.shape
    
    # 1. Inicialización para crecimiento global
    # Creo una matriz para guardar las etiquetas (IDs) de las regiones. Uso uint32 
    # para evitar errores de desbordamiento (overflow) si hay demasiadas regiones.
    region_labels = np.zeros((rows, cols), dtype=np.uint32)
    current_region_id = 0 # Inicializo el contador de regiones.
    
    # Lista para almacenar las coordenadas (y, x) de cada píxel semilla.
    seed_points_list = []
    
    # Defino los desplazamientos para la 8-conectividad (vecinos directos y diagonales).
    neighbor_offsets = [
        (-1, 0), (1, 0), (0, -1), (0, 1), 
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    # 2. Bucle principal para encontrar semillas
    # Recorro cada píxel de la imagen (de arriba a abajo, de izquierda a derecha).
    for r in range(rows):
        for c in range(cols):
            
            # Si el píxel aún no pertenece a ninguna región (su etiqueta es 0), 
            # lo tomo como una nueva semilla para una nueva región.
            if region_labels[r, c] == 0:
                
                current_region_id += 1 # Incremento el ID de la nueva región.
                
                # Guardo la coordenada del píxel semilla actual.
                seed_points_list.append((r, c)) # (y, x)
                
                # Inicio la cola con la nueva semilla para el proceso de expansión.
                queue = deque([(r, c)])
                region_labels[r, c] = current_region_id # Etiqueto la semilla.
                
                # Inicializo la suma de intensidad y el contador de píxeles para el criterio adaptativo.
                # Uso np.int64 para evitar el overflow en la suma de intensidades de regiones grandes.
                current_region_sum = np.int64(img[r, c])
                current_region_count = 1

                # 3. Proceso de Crecimiento (Se expande la región desde la semilla)
                while queue:
                    current_y, current_x = queue.popleft() # Saco el píxel actual de la cola.
                    
                    # Calculo la media actual de la región; esto hace el criterio adaptativo.
                    region_mean = current_region_sum / current_region_count 

                    # Reviso a los 8 vecinos del píxel actual.
                    for dy, dx in neighbor_offsets:
                        neighbor_y, neighbor_x = current_y + dy, current_x + dx
                        
                        # Verifico que el vecino esté dentro de los límites de la imagen.
                        if 0 <= neighbor_y < rows and 0 <= neighbor_x < cols:
                            
                            # Verifico que el vecino no haya sido etiquetado todavía (región_ID = 0).
                            if region_labels[neighbor_y, neighbor_x] == 0:
                                
                                neighbor_intensity = img[neighbor_y, neighbor_x]
                                
                                # Criterio de similitud: la diferencia absoluta de intensidad 
                                # entre el vecino y la media actual debe ser menor o igual al umbral.
                                if abs(int(neighbor_intensity) - region_mean) <= threshold:
                                    
                                    # Si cumple el criterio, lo añado a la región y a la cola para expansión.
                                    region_labels[neighbor_y, neighbor_x] = current_region_id
                                    queue.append((neighbor_y, neighbor_x))
                                    
                                    # Actualizo las estadísticas de la región para el cálculo de la media adaptativa.
                                    current_region_sum += neighbor_intensity 
                                    current_region_count += 1
                                    
    print(f"Segmentación completa. Se detectaron {current_region_id} regiones.")
    # Retorno el mapa de etiquetas y la lista de semillas para la visualización.
    return region_labels, seed_points_list

# --- Ejemplo de Uso con Visualización ---

# Defino la ruta de la imagen de entrada y el umbral de similitud.
image_path = 'segmentationImages/cat-original.png' # Se puede cambiar la imagen
# image_path = 'segmentationImages/cars1-original.png' # Ejemplo alternativo 1
# image_path = 'segmentationImages/cars2-original.png' # Ejemplo alternativo 2
intensity_threshold = 30

# 1. Ejecuto la función y capturo el mapa de etiquetas y la lista de semillas.
segmented_map, seeds = region_growing_global_v3(image_path, intensity_threshold)

# 2. Visualizar los resultados
if segmented_map is not None:
    # Leo la imagen original a color para mostrarla.
    original_img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Convierto de BGR (formato de OpenCV) a RGB (formato de Matplotlib).
    if original_img_color is not None:
        original_img_color = cv2.cvtColor(original_img_color, cv2.COLOR_BGR2RGB)
    
    # Creo la figura para los dos subplots.
    plt.figure(figsize=(15, 6))
    
    # Subplot 1: Imagen Original con Semillas
    plt.subplot(1, 2, 1)
    
    if original_img_color is not None:
        plt.imshow(original_img_color)
    else:
        # Si la carga a color falla, intento cargarla en escala de grises.
        original_img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_img_gray is not None:
            plt.imshow(original_img_gray, cmap='gray')
            
    # Preparo las coordenadas (x, y) para plotear las semillas.
    seed_x = [p[1] for p in seeds]
    seed_y = [p[0] for p in seeds]
    
    # Dibujo todas las semillas como puntos rojos pequeños.
    plt.plot(seed_x, seed_y, 'ro', markersize=2) 
    plt.title(f"Imagen Original con {len(seeds)} Semillas (Puntos Rojos)")
    
    # Subplot 2: Mapa de Regiones Segmentadas
    plt.subplot(1, 2, 2)
    # Muestro el mapa de etiquetas usando un mapa de colores (viridis) para diferenciar las regiones.
    plt.imshow(segmented_map, cmap='viridis') 
    plt.colorbar(label='ID de Región')
    plt.title(f"Mapa de Regiones Segmentadas (Umbral: {intensity_threshold})")
    
    # Guardo la imagen de resultado en el directorio 'output'.
    plt.savefig('output/output-segmentation.png')