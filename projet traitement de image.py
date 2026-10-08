import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

# ======================================================
# Configuration
# ======================================================
image_path = r"C:\Users\AHMED\OneDrive\Desktop\projet\src\vache1.jpg"
out_dir = r"C:\Users\AHMED\OneDrive\Desktop\projet\out_auto"
os.makedirs(out_dir, exist_ok=True)

WHITE_SAT_MAX = 30
WHITE_VAL_MIN = 235
CLOSE_KERNEL = (7,7)
ERODE_KERNEL = (5,5)
MIN_AREA_PX = 2000
MARGIN = 40

# ======================================================
# Fonctions de poids (REALISTES)
# ======================================================
def estimate_weight_auto(length_px, height_px):
    """
    Calcule le poids de la vache en utilisant :
    - hauteur réelle moyenne = 1.40 m
    - estimation de la longueur physique
    - formule scientifique adaptée
    """
    REAL_HEIGHT_M = 1.40   # hauteur adulte au garrot standard
    
    # Conversion m/pixel
    meters_per_pixel = REAL_HEIGHT_M / max(1, height_px)

    # Longueur réelle du corps
    length_m = length_px * meters_per_pixel

    # Formule bovine réaliste : 300 × (L - 0.70)
    weight = 300 * max(0, (length_m - 0.70))

    return weight, length_m


# ======================================================
# Fonctions utilitaires
# ======================================================
def smooth_mask_and_get_contours(mask, blur_ksize=(11,11), thresh=127):
    blurred = cv2.GaussianBlur(mask.astype(np.uint8), blur_ksize, 0)
    _, th = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones(CLOSE_KERNEL, np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return th, contours

def contour_simplify_and_smooth(cnt, smoothing_factor=0.01):
    arclen = cv2.arcLength(cnt, True)
    eps = max(1.0, smoothing_factor * arclen)
    approx = cv2.approxPolyDP(cnt, eps, True)
    hull = cv2.convexHull(approx)
    return hull


# ======================================================
# Pipeline principal
# ======================================================
bgr = cv2.imread(image_path)
if bgr is None:
    raise FileNotFoundError(image_path)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
H, W = rgb.shape[:2]

# Masque fond blanc
hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
mask_white = cv2.inRange(
    hsv,
    np.array([0,0,WHITE_VAL_MIN]),
    np.array([180,WHITE_SAT_MAX,255])
)
mask_nonwhite = cv2.bitwise_not(mask_white)

# Morphologie
mask_closed = cv2.morphologyEx(mask_nonwhite, cv2.MORPH_CLOSE, np.ones(CLOSE_KERNEL, np.uint8))
mask_eroded = cv2.erode(mask_closed, np.ones(ERODE_KERNEL, np.uint8), iterations=1)

# Supprimer sol
mask_eroded[int(H*0.85):, :] = 0

# Contours
mask_smooth, contours = smooth_mask_and_get_contours(mask_eroded, blur_ksize=(15,15))
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# CSV
csv_path = os.path.join(out_dir, "measurements_auto.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "index","area_px","length_px","height_px","length_m_estimated","weight_kg",
        "bbox_x","bbox_y","bbox_w","bbox_h","mask","crop","overlay"
    ])

overlay_base = rgb.copy()
cow_idx = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < MIN_AREA_PX:
        continue

    smooth_cnt = contour_simplify_and_smooth(cnt)

    rect = cv2.minAreaRect(smooth_cnt)
    (cx,cy),(w_rect,h_rect),angle = rect
    length_px = max(w_rect, h_rect)
    height_px = min(w_rect, h_rect)   # hauteur verticale

    # Bounding box
    x,y,w_box,h_box = cv2.boundingRect(smooth_cnt)
    x1=max(0,x-MARGIN); y1=max(0,y-MARGIN)
    x2=min(W,x+w_box+MARGIN); y2=min(H,y+h_box+MARGIN)

    # Mask crop
    mask_cow = np.zeros_like(mask_smooth)
    cv2.drawContours(mask_cow,[smooth_cnt],-1,255,-1)
    mask_cow = cv2.GaussianBlur(mask_cow,(9,9),0)
    _,mask_cow = cv2.threshold(mask_cow,127,255,cv2.THRESH_BINARY)

    crop_rgb = rgb[y1:y2, x1:x2]
    crop_mask = mask_cow[y1:y2, x1:x2]
    cow_only = cv2.bitwise_and(crop_rgb, crop_rgb, mask=crop_mask)

    # ======== POIDS RÉALISTE ========
    weight_kg, length_m = estimate_weight_auto(length_px, h_box)

    print(f"Vache {cow_idx} → {weight_kg:.1f} kg  (longueur estimée = {length_m:.2f} m)")

    # Sauvegardes
    mask_path = os.path.join(out_dir, f"cow_{cow_idx:02d}_mask.png")
    crop_path = os.path.join(out_dir, f"cow_{cow_idx:02d}_crop.png")
    only_path = os.path.join(out_dir, f"cow_{cow_idx:02d}_only.png")
    overlay_path = os.path.join(out_dir, f"cow_{cow_idx:02d}_overlay.png")

    cv2.imwrite(mask_path, mask_cow)
    cv2.imwrite(crop_path, cv2.cvtColor(crop_rgb,cv2.COLOR_RGB2BGR))
    cv2.imwrite(only_path, cv2.cvtColor(cow_only,cv2.COLOR_RGB2BGR))

    overlay = overlay_base.copy()
    cv2.drawContours(overlay,[smooth_cnt],-1,(255,0,0),3)
    box = cv2.boxPoints(rect).astype(int)
    cv2.drawContours(overlay,[box],-1,(0,255,0),2)
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))

    # CSV write
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            cow_idx,
            area, length_px, height_px, length_m, weight_kg,
            x1,y1,(x2-x1),(y2-y1),
            mask_path, crop_path, overlay_path
        ])

    cow_idx += 1

print(f"🎉 Terminé ! {cow_idx} vache(s) détectée(s). Mesures + poids sauvegardés dans : {csv_path}")

# Aperçu
show_n = min(4, cow_idx)
fig = plt.figure(figsize=(15,4))
for j in range(show_n):
    path = os.path.join(out_dir, f"cow_{j:02d}_overlay.png")
    ax = fig.add_subplot(1, show_n, j+1)
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    ax.imshow(img); ax.axis('off'); ax.set_title(f"cow_{j:02d}")
plt.show()
