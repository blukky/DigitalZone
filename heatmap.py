import cv2
import numpy as np

a = np.random.randint(1, 100, size=(1080, 1920))
img = cv2.imread('movies2/Антон/movies2_image_0.jpg')
heatmapshow = None
heatmapshow = cv2.normalize(a, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
super_imposed_img = cv2.addWeighted(heatmapshow, 0.3, img, 0.5, 0)
cv2.imwrite('addcolor.png', super_imposed_img)