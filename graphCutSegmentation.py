"""
Script de Segmentación con GraphCut

Este script realiza segmentación de imágenes usando el algoritmo GrabCut de OpenCV.
El algoritmo GrabCut es un método iterativo para segmentación de primer plano/fondo
basado en cortes de grafo. Requiere un rectángulo inicial que contenga aproximadamente
el objeto de interés.

El script:
1. Carga una imagen y la convierte del espacio de color BGR a RGB
2. Aplica el algoritmo GrabCut para segmentar el objeto en primer plano
3. Visualiza la imagen original, la máscara de segmentación y el resultado final segmentado
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Paso 1: Cargar y preprocesar la imagen
# ============================================================================

# Cargar la imagen original desde archivo
cat_original = cv.imread('./segmentationImages/cat-original.png')

# Convertir de BGR (por defecto en OpenCV) a RGB (por defecto en matplotlib) para visualización correcta
cat_original = cv.cvtColor(cat_original, cv.COLOR_BGR2RGB)

# ============================================================================
# Paso 2: Configurar visualización e inicializar parámetros de GrabCut
# ============================================================================

# Crear una figura con 3 subgráficos dispuestos verticalmente (3 columnas, 1 fila)
plt.figure()

# Subgráfico 1: Mostrar la imagen original
plt.subplot(131)
plt.imshow(cat_original)
plt.title('Imagen Original')
plt.axis('off')

# Subgráfico 2: Mostrará la máscara de segmentación
plt.subplot(132)

# Obtener dimensiones de la imagen (altura, ancho, canales)
rows, cols, _ = cat_original.shape

# Inicializar la máscara: ceros indican píxeles de fondo/primer plano a determinar
# La máscara será actualizada por grabCut con los valores:
#   0 = definitivamente fondo
#   1 = definitivamente primer plano
#   2 = probablemente fondo
#   3 = probablemente primer plano
mask = np.zeros((rows, cols), np.uint8)

# Inicializar modelos para Modelos de Mezcla Gaussiana (GMM) de fondo y primer plano
# Estos serán aprendidos por el algoritmo para modelar distribuciones de color de píxeles
# Forma: (1, 65) - 65 parámetros para el GMM
bgdModel = np.zeros((1, 65), np.float64)  # Modelo de fondo
fgdModel = np.zeros((1, 65), np.float64)   # Modelo de primer plano

# ============================================================================
# Paso 3: Aplicar algoritmo GrabCut
# ============================================================================

# Definir el rectángulo inicial que contiene el objeto de interés
# Formato: (x, y, ancho, alto) donde (x,y) es la esquina superior izquierda
# Este rectángulo debe contener aproximadamente el objeto en primer plano
rect = (300, 100, 450, 480)  # Rectángulo de selección del objeto

# Aplicar algoritmo GrabCut
# Parámetros:
#   - cat_original: imagen de entrada (RGB de 3 canales)
#   - mask: máscara de salida (será modificada)
#   - rect: rectángulo inicial que contiene el objeto
#   - bgdModel: modelo de fondo (será actualizado)
#   - fgdModel: modelo de primer plano (será actualizado)
#   - 1: número de iteraciones a ejecutar
#   - cv.GC_INIT_WITH_RECT: modo de inicialización (usar rectángulo para inicializar)
cv.grabCut(cat_original, mask, rect, bgdModel, fgdModel, 1, cv.GC_INIT_WITH_RECT)

# Mostrar la máscara de segmentación
plt.imshow(mask, cmap='gray')
plt.title('Máscara de Segmentación')
plt.axis('off')

# ============================================================================
# Paso 4: Crear imagen segmentada final
# ============================================================================

plt.subplot(133)

# Crear máscara binaria: convertir valores de máscara GrabCut a binario (0 o 1)
# Valores 0 (definitivamente fondo) y 2 (probablemente fondo) -> 0
# Valores 1 (definitivamente primer plano) y 3 (probablemente primer plano) -> 1
maskGrabCut = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Aplicar la máscara a la imagen original para extraer el primer plano
# maskGrabCut[:,:,np.newaxis] añade una dimensión de canal para coincidir con la forma de la imagen (H, W, 3)
# Esto multiplica cada píxel por el valor de la máscara (0 o 1), enmascarando efectivamente el fondo
cat_segmented = cat_original * maskGrabCut[:, :, np.newaxis]

# Mostrar el resultado segmentado final
plt.imshow(cat_segmented)
plt.title('Imagen Segmentada (Objeto Extraído)')
plt.axis('off')

# Mostrar todos los subgráficos
plt.tight_layout()  # Ajustar diseño para evitar solapamiento
plt.show()