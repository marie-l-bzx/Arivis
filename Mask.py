#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
from skimage import morphology
from aicsimageio import AICSImage
import napari

# -------------------
# Paramètres
# -------------------
DEFAULT_CZI_PATH = Path(__file__).parent / "20250303_SCI-9322_24Fev_ZTL_940_D7_Stitch_Linear unmixing.czi"
CFP_CHANNEL_ZEN = 1  # CFP
GFP_CHANNEL_ZEN = 2  # GFP
RFP_CHANNEL_ZEN = 3  # RFP

THRESH_DURE_MERE = 0.8
THRESH_PERTE_SIGNAL = 0.15

# -------------------
# Chargement
# -------------------
def prepare_image(path):
    img = AICSImage(path)
    data = img.get_image_data("CZYX", S=0, T=0)  # shape: C, Z, Y, X
    return data

if __name__ == "__main__":
    infile = DEFAULT_CZI_PATH
    if not os.path.exists(infile):
        raise FileNotFoundError(f"Fichier introuvable: {infile}")

    print(f"Chargement du fichier: {infile}")
    stack = prepare_image(infile)

    # Extraction des canaux (index Python = Zen - 1)
    stack_cfp = stack[CFP_CHANNEL_ZEN - 1].astype(float)
    stack_gfp = stack[GFP_CHANNEL_ZEN - 1].astype(float)
    stack_rfp = stack[RFP_CHANNEL_ZEN - 1].astype(float)

    # Normalisation
    stack_cfp /= stack_cfp.max()
    stack_gfp /= stack_gfp.max()
    stack_rfp /= stack_rfp.max()

    # -------------------
    # Région totale
    # -------------------
    region_totale = np.ones_like(stack_cfp, dtype=bool)

    # -------------------
    # Masque exclusion (basé sur CFP uniquement)
    # -------------------
    dure_mere_mask = stack_cfp > THRESH_DURE_MERE
    black_mask = stack_cfp == 0
    mean_signal = np.mean(stack_cfp, axis=0)
    low_signal_mask_2d = mean_signal < THRESH_PERTE_SIGNAL
    low_signal_mask = np.broadcast_to(low_signal_mask_2d, stack_cfp.shape)

    exclusion_mask = dure_mere_mask | black_mask | low_signal_mask
    exclusion_mask = morphology.remove_small_objects(exclusion_mask, min_size=50)

    # -------------------
    # Visualisation Napari
    # -------------------
    viewer = napari.Viewer()

    # CFP cyan
    viewer.add_image(
        stack_cfp,
        name="CFP",
        colormap="cyan",
        blending="additive",
        contrast_limits=[0, 1]
    )

    # YFP jaune
    viewer.add_image(
        stack_gfp,
        name="GFP",
        colormap="green",
        blending="additive",
        contrast_limits=[0, 1]
    )

    # RFP rouge
    viewer.add_image(
        stack_rfp,
        name="RFP",
        colormap="red",
        blending="additive",
        contrast_limits=[0, 1]
    )

    # Palette pour labels
    palette_region = {0: (0, 0, 0, 0), 1: (1, 0, 0, 0.3)}  # rouge transparent
    palette_exclusion = {0: (0, 0, 0, 0), 1: (0, 0, 1, 0.3)}  # bleu transparent

    # Région totale
    region_layer = viewer.add_labels(region_totale.astype(np.uint8), name="Région Totale")
    region_layer.color = palette_region

    # Masque exclusion
    exclusion_layer = viewer.add_labels(exclusion_mask.astype(np.uint8), name="Masque Exclusion")
    exclusion_layer.color = palette_exclusion
    
    #--------------------
    #Sauvegard du masque après édition
    #--------------------
    @viewer.bind_key('s') #touche"s" pour sauvegarder
    def save_mask(viewer):
        mask_final = viewer.layers["Masque Exclusion"].data.astype(np.uint8)
        outpath = Path(infile).with_suffix('.npy')
        np.save(outpath, mask_final)
        print(f"Masque sauvegardé sous: {outpath}")

    napari.run()
