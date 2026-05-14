"""
PenguinID Pipeline: Individual recognition via wing patterns.
Includes Dual-Orientation Matching to handle left and right wings automatically.
"""

import os
import gdown
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from wildlife_tools.data import WildlifeDataset
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity

# --- CONFIGURATION ---
FOLDER_ID = '1Sw6TlXLwTD8441nHSrffX-AzAZX_A6lp'
DATA_DIR = './penguin_wings_final'

# 1. DOWNLOAD DATA
if not os.path.exists(DATA_DIR):
    print("Downloading wing database from Google Drive...")
    gdown.download_folder(id=FOLDER_ID, output=DATA_DIR, quiet=False)

# 2. PRE-PROCESSING
transform = T.Compose([
    T.Resize([224, 224]),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# 3. INITIALIZE MODEL & DATABASE
# pylint: disable=no-member
dataset = WildlifeDataset.from_folder(root=DATA_DIR, transform=transform)

print("Loading MegaDescriptor model...")
model = timm.create_model(
    'hf-hub:BVRA/MegaDescriptor-T-224',
    pretrained=True,
    num_classes=0
)
model.eval()
extractor = DeepFeatures(model)

print(f"Extracting fingerprints for {len(dataset)} images...")
database_features = extractor(dataset)

# Initialize similarity helper for identification
similarity_tool = CosineSimilarity()

# 4. DUAL-ORIENTATION IDENTIFICATION
def identify_penguin(image_path):
    """
    Checks the query image and its flipped version against the database.
    Returns the ID with the highest similarity score.
    """
    img_orig = Image.open(image_path).convert('RGB')
    img_flip = img_orig.transpose(Image.FLIP_LEFT_RIGHT)

    tensor_orig = transform(img_orig).unsqueeze(0)
    tensor_flip = transform(img_flip).unsqueeze(0)

    with torch.no_grad():
        feat_orig = model(tensor_orig)
        feat_flip = model(tensor_flip)

    # Calculate similarity scores (using torch.cosine_similarity to satisfy pylint)
    sim_orig = torch.cosine_similarity(feat_orig, database_features)
    sim_flip = torch.cosine_similarity(feat_flip, database_features)

    # Get the best match from both orientations
    max_orig, idx_orig = torch.max(sim_orig, dim=0)
    max_flip, idx_flip = torch.max(sim_flip, dim=0)

    if max_orig > max_flip:
        best_idx = idx_orig
        score = max_orig
        orientation = "Original"
    else:
        best_idx = idx_flip
        score = max_flip
        orientation = "Flipped"

    predicted_id = dataset.labels_string[best_idx.item()]

    return {
        "id": predicted_id,
        "confidence": round(float(score), 4),
        "match_orientation": orientation
    }

if __name__ == "__main__":
    print("System Ready. Place a query image path in identify_penguin() to test.")
    