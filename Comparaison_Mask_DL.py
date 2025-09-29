import numpy as np
import napari
import matplotlib.pyplot as plt
from skimage.filters import sobel, laplace
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter
from pathlib import Path
import json

# ======================
# 1. Chargement des masques prédits
# ======================
def load_predictions(unet_path, vit_path):
    """Charge les masques prédits par U-Net et ViT."""
    unet_mask = np.load(unet_path)
    vit_mask = np.load(vit_path)

    # Binarisation (si les masques sont des probabilités)
    unet_mask = (unet_mask > 0.5).astype(np.uint8)
    vit_mask = (vit_mask > 0.5).astype(np.uint8)

    # Vérification des dimensions
    assert unet_mask.shape == vit_mask.shape, \
        f"Les masques n'ont pas la même forme: U-Net={unet_mask.shape}, ViT={vit_mask.shape}"

    return unet_mask, vit_mask

# Chemins vers vos fichiers
unet_path = Path("C:/Users/marie/Documents/Arivis/data/processed/20250225_SCI-9334_18Fev_ZTL_940_D7_Stitch_Linear unmixing.mask_TF500.npy")
vit_path = Path("C:/Users/marie/Documents/Arivis/data/processed/20250225_SCI-9334_18Fev_ZTL_940_D7_Stitch_Linear unmixing.mask_ViT500.npy")

unet_mask, vit_mask = load_predictions(unet_path, vit_path)

# ======================
# 2. Visualisation 3D avec Napari
# ======================
def visualize_3d_comparison(unet, vit):
    """Compare visuellement les masques en 3D."""
    viewer = napari.Viewer()

    # Ajout des masques avec des couleurs distinctes
    viewer.add_labels(unet, name="U-Net")
    viewer.add_labels(vit, name="ViT")

    # Mode de fusion pour superposer
    viewer.layers["ViT"].blending = "additive"
    viewer.layers["ViT"].opacity = 0.7

    # Ajout d'une carte de différence (jaune = désaccord)
    diff_mask = (unet != vit).astype(np.uint8)
    viewer.add_labels(diff_mask, name="Différences")

    napari.run()

print("Lancement de la visualisation 3D (fermez la fenêtre pour continuer)...")
visualize_3d_comparison(unet_mask, vit_mask)

# ======================
# 3. Analyse des contours (netteté)
# ======================
def analyze_edges(mask, sigma=1.0):
    """Calcule la netteté des contours via un filtre Sobel."""
    # Lissage léger pour réduire le bruit
    smoothed = gaussian_filter(mask.astype(float), sigma=sigma)
    # Détection des contours
    edges = sobel(smoothed)
    # Netteté = intensité moyenne des contours
    sharpness = np.mean(edges[edges > 0.1])  # Seuil pour ignorer le bruit
    return sharpness, edges

unet_sharpness, unet_edges = analyze_edges(unet_mask)
vit_sharpness, vit_edges = analyze_edges(vit_mask)

print(f"\n=== Netteté des contours (Sobel) ===")
print(f"U-Net: {unet_sharpness:.3f} | ViT: {vit_sharpness:.3f}")
print(f"Différence: {vit_sharpness - unet_sharpness:.3f} (ViT > 0 = contours plus nets)")

# ======================
# 4. Comparaison slice par slice (2D)
# ======================
def plot_2d_comparison(unet, vit, slice_idx=None):
    """Affiche une comparaison 2D avec contours et différences."""
    if slice_idx is None:
        slice_idx = unet.shape[0] // 2  # Slice centrale

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # U-Net + contours
    axes[0].imshow(unet[slice_idx], cmap="Reds")
    axes[0].imshow(unet_edges[slice_idx], cmap="viridis", alpha=0.5)
    axes[0].set_title(f"U-Net (Slice {slice_idx})\nNetteté: {unet_sharpness:.2f}")

    # ViT + contours
    axes[1].imshow(vit[slice_idx], cmap="Blues")
    axes[1].imshow(vit_edges[slice_idx], cmap="viridis", alpha=0.5)
    axes[1].set_title(f"ViT (Slice {slice_idx})\nNetteté: {vit_sharpness:.2f}")

    # Différences (jaune = désaccord)
    diff = np.abs(unet[slice_idx].astype(int) - vit[slice_idx].astype(int))
    axes[2].imshow(diff, cmap="YlOrBr")
    axes[2].set_title("Différences absolues")

    plt.suptitle("Comparaison U-Net vs ViT (contours en vert)", y=1.02)
    plt.tight_layout()
    plt.show()

print("\nAffichage de la comparaison 2D (slice centrale)...")
plot_2d_comparison(unet_mask, vit_mask)

# ======================
# 5. NOUVELLE VERSION: Analyse des régions 2D/3D
# ======================
def analyze_regions_2d(mask_3d):
    """Analyse les régions slice par slice (2D)."""
    results = {
        "nb_regions_per_slice": [],
        "area_mean_per_slice": [],
        "circularity_mean_per_slice": [],  # Utilise 4π(aire)/périmètre²
    }

    for slice_idx in range(mask_3d.shape[0]):
        slice_2d = mask_3d[slice_idx]
        if np.sum(slice_2d) == 0:  # Ignorer les slices vides
            continue

        labeled_slice = label(slice_2d)
        regions = regionprops(labeled_slice)

        if not regions:
            continue

        # Métriques 2D
        areas = [r.area for r in regions]
        # Circularité = 4π(aire)/périmètre² (0=ligne, 1=cercle)
        circularities = [
            (4 * np.pi * r.area / (r.perimeter ** 2 + 1e-6)) if r.perimeter > 0 else 0
            for r in regions
        ]

        results["nb_regions_per_slice"].append(len(regions))
        results["area_mean_per_slice"].append(np.mean(areas))
        results["circularity_mean_per_slice"].append(np.mean(circularities))

    # Moyennes globales (en ignorant les slices vides)
    if results["nb_regions_per_slice"]:
        results.update({
            "nb_regions_global": np.mean(results["nb_regions_per_slice"]),
            "area_mean_global": np.mean(results["area_mean_per_slice"]),
            "circularity_mean_global": np.mean(results["circularity_mean_per_slice"])
        })
    else:
        results.update({
            "nb_regions_global": 0,
            "area_mean_global": 0,
            "circularity_mean_global": 0
        })

    return results

def analyze_regions_3d(mask_3d):
    """Analyse les régions en 3D avec des métriques compatibles."""
    labeled_mask = label(mask_3d)
    regions = regionprops(labeled_mask)

    if not regions:
        return {
            "nb_regions": 0,
            "volume_mean": 0,
            "solidity_mean": 0,
            "equivalent_diameter_mean": 0
        }

    volumes = [r.area for r in regions]  # 'area' = volume en 3D
    solidities = [r.solidity for r in regions]
    equivalent_diameters = [r.equivalent_diameter for r in regions]

    return {
        "nb_regions": len(regions),
        "volume_mean": np.mean(volumes),
        "solidity_mean": np.mean(solidities),
        "equivalent_diameter_mean": np.mean(equivalent_diameters)
    }

# Analyse 2D et 3D
print("\n=== Analyse des régions 2D (par slice) ===")
unet_regions_2d = analyze_regions_2d(unet_mask)
vit_regions_2d = analyze_regions_2d(vit_mask)

print(f"U-Net: {unet_regions_2d['nb_regions_global']:.1f} régions/slice | Aire moy.: {unet_regions_2d['area_mean_global']:.1f} | Circularité: {unet_regions_2d['circularity_mean_global']:.3f}")
print(f"ViT:  {vit_regions_2d['nb_regions_global']:.1f} régions/slice | Aire moy.: {vit_regions_2d['area_mean_global']:.1f} | Circularité: {vit_regions_2d['circularity_mean_global']:.3f}")

#print("\n=== Analyse des régions 3D ===")
#unet_regions_3d = analyze_regions_3d(unet_mask)
#vit_regions_3d = analyze_regions_3d(vit_mask)

#print(f"U-Net: {unet_regions_3d['nb_regions']} régions | Volume moy.: {unet_regions_3d['volume_mean']:.1f} voxels | Solidity: {unet_regions_3d['solidity_mean']:.3f}")
#print(f"ViT:  {vit_regions_3d['nb_regions']} régions | Volume moy.: {vit_regions_3d['volume_mean']:.1f} voxels | Solidity: {vit_regions_3d['solidity_mean']:.3f}")

# ======================
# 6. Stabilité des prédictions
# ======================
def compare_stability(masks_list):
    """Évalue la variabilité entre plusieurs prédictions d'un même modèle."""
    if len(masks_list) < 2:
        return {"stability": 1.0}  # Pas de variabilité si un seul masque

    # Calcul de la variance voxel-wise
    variance = np.var(masks_list, axis=0)
    mean_variance = np.mean(variance)
    stability = 1 - mean_variance  # 1 = parfaitement stable

    return {"stability": stability, "variance_moyenne": mean_variance}

# ======================
# 7. Export des résultats
# ======================
def save_comparison(unet, vit, unet_edges, vit_edges, output_dir="comparison_results"):
    """Sauvegarde les visualisations et métriques."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Sauvegarde des métriques
    results = {
        "contour_sharpness": {"U-Net": unet_sharpness, "ViT": vit_sharpness},
        "regions_2d": {"U-Net": unet_regions_2d, "ViT": vit_regions_2d},
        #"regions_3d": {"U-Net": unet_regions_3d, "ViT": vit_regions_3d},
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Sauvegarde des images 2D
    slice_idx = unet.shape[0] // 2
    fig, _ = plt.subplots(1, 3, figsize=(15, 5))
    plot_2d_comparison(unet, vit, slice_idx)
    fig.savefig(output_dir / "comparison_2d.png", dpi=300, bbox_inches="tight")

    print(f"\nRésultats sauvegardés dans {output_dir}/")

save_comparison(unet_mask, vit_mask, unet_edges, vit_edges)
