import cv2
import numpy as np
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def region_growing_global_v3(image_path, threshold):
    """
    Detecta todas las regiones y retorna tanto el mapa de etiquetas como 
    la lista de coordenadas de las semillas utilizadas para iniciar cada región.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: No se pudo cargar la imagen en {image_path}")
        return None

    rows, cols = img.shape
    
    # 1. Inicialización para crecimiento global
    region_labels = np.zeros((rows, cols), dtype=np.uint32)
    current_region_id = 0
    
    # NUEVA CARACTERÍSTICA: Lista para almacenar todas las semillas
    seed_points_list = []
    
    neighbor_offsets = [
        (-1, 0), (1, 0), (0, -1), (0, 1), 
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    # 2. Bucle principal para encontrar semillas
    for r in range(rows):
        for c in range(cols):
            
            # Si el píxel no ha sido etiquetado, úsalo como una nueva semilla.
            if region_labels[r, c] == 0:
                
                current_region_id += 1
                
                # NUEVA CARACTERÍSTICA: Guardar la coordenada de la nueva semilla
                seed_points_list.append((r, c)) # (y, x)
                
                queue = deque([(r, c)])
                region_labels[r, c] = current_region_id
                
                # Criterio adaptativo para la media
                current_region_sum = np.int64(img[r, c])
                current_region_count = 1

                # 3. Proceso de Crecimiento
                while queue:
                    current_y, current_x = queue.popleft()
                    
                    region_mean = current_region_sum / current_region_count 

                    for dy, dx in neighbor_offsets:
                        neighbor_y, neighbor_x = current_y + dy, current_x + dx
                        
                        if 0 <= neighbor_y < rows and 0 <= neighbor_x < cols:
                            
                            if region_labels[neighbor_y, neighbor_x] == 0:
                                
                                neighbor_intensity = img[neighbor_y, neighbor_x]
                                
                                if abs(int(neighbor_intensity) - region_mean) <= threshold:
                                    
                                    region_labels[neighbor_y, neighbor_x] = current_region_id
                                    queue.append((neighbor_y, neighbor_x))
                                    
                                    current_region_sum += neighbor_intensity 
                                    current_region_count += 1
                                    
    print(f"Segmentación completa. Se detectaron {current_region_id} regiones.")
    # La función ahora retorna la lista de semillas junto con el mapa de etiquetas
    return region_labels, seed_points_list

# --- Ejemplo de Uso con Visualización ---

image_path = 'segmentationImages/cat-original.png' # Ajusta tu ruta
# image_path = 'segmentationImages/cars1-original.png' # Ajusta tu ruta
# image_path = 'segmentationImages/cars2-original.png' # Ajusta tu ruta
intensity_threshold = 30

# 1. Ejecutar el algoritmo
segmented_map, seeds = region_growing_global_v3(image_path, intensity_threshold)

# 2. Visualizar los resultados
if segmented_map is not None:
    original_img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Convertir a RGB para que Matplotlib lo muestre correctamente si la original es a color
    if original_img_color is not None:
        original_img_color = cv2.cvtColor(original_img_color, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15, 6))
    
    # Subplot 1: Imagen Original con Semillas
    plt.subplot(1, 2, 1)
    
    if original_img_color is not None:
        plt.imshow(original_img_color)
    else:
        # Si la imagen no se pudo cargar a color, la mostramos en escala de grises
        original_img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_img_gray is not None:
            plt.imshow(original_img_gray, cmap='gray')
        
    # Dibujar todas las semillas
    # seeds es una lista de (y, x), pero plt.plot usa (x, y)
    seed_x = [p[1] for p in seeds]
    seed_y = [p[0] for p in seeds]
    
    # Usamos un color distintivo (rojo 'r') y un tamaño pequeño
    plt.plot(seed_x, seed_y, 'ro', markersize=2) 
    plt.title(f"Imagen Original con {len(seeds)} Semillas (Puntos Rojos)")
    
    # Subplot 2: Mapa de Regiones Segmentadas
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_map, cmap='viridis') 
    plt.colorbar(label='ID de Región')
    plt.title(f"Mapa de Regiones Segmentadas (Umbral: {intensity_threshold})")
    
    # plt.show()
    plt.savefig('output/output-segmentation.png')