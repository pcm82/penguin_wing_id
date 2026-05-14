"""
PenguinID Pipeline: Multi-processing Safe Version.
Handles Windows-specific bootstrapping and flat folder datasets.
"""

import os
import gdown
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from wildlife_tools.features import DeepFeatures

# --- CONFIGURATION ---
FOLDER_ID = '1Sw6TlXLwTD8441nHSrffX-AzAZX_A6lp'
DATA_DIR = './penguin_wings_final'

# --- DATASET LOADER ---
class FlatImageDataset(Dataset):
    """Loads images from a flat folder without needing subdirectories."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not self.image_files:
            raise FileNotFoundError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

# --- IDENTIFICATION FUNCTION ---
def identify_penguin(image_path, model, transform, database_features, image_files):
    """Checks query and flipped version against the database."""
    img_orig = Image.open(image_path).convert('RGB')
    img_flip = img_orig.transpose(Image.FLIP_LEFT_RIGHT)

    tensor_orig = transform(img_orig).unsqueeze(0)
    tensor_flip = transform(img_flip).unsqueeze(0)

    with torch.no_grad():
        feat_orig = model(tensor_orig)
        feat_flip = model(tensor_flip)

    sim_orig = torch.cosine_similarity(feat_orig, database_features)
    sim_flip = torch.cosine_similarity(feat_flip, database_features)

    max_orig, idx_orig = torch.max(sim_orig, dim=0)
    max_flip, idx_flip = torch.max(sim_flip, dim=0)

    if max_orig > max_flip:
        best_idx, score, orientation = idx_orig, max_orig, "Original"
    else:
        best_idx, score, orientation = idx_flip, max_flip, "Flipped"

    matched_filename = image_files[best_idx.item()]

    return {
        "matched_file": matched_filename,
        "confidence": round(float(score), 4),
        "match_orientation": orientation
    }

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. DOWNLOAD DATA
    if not os.path.exists(DATA_DIR):
        print("Downloading wing database...")
        gdown.download_folder(id=FOLDER_ID, output=DATA_DIR, quiet=False)
    else:
        print(f"Local database found at {DATA_DIR}. Skipping download.")

    # 2. INITIALIZE DATASET
    img_transform = T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    dataset = FlatImageDataset(root_dir=DATA_DIR, transform=img_transform)

    # 3. LOAD MODEL
    print("Loading MegaDescriptor model...")
    # Using 0 workers for wildlife-tools to avoid any further Windows process issues
    penguin_model = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', pretrained=True, num_classes=0)
    penguin_model.eval()
    
    # num_workers=0 ensures it runs in the main process
    extractor = DeepFeatures(penguin_model, num_workers=0)

    print(f"Extracting fingerprints for {len(dataset)} images...")
    db_features = extractor(dataset)

    print("--- System Ready ---")
    
    # 4. TEST EXAMPLE (Uncomment and add a path to test)
    # test_file = "C:/Users/time4/Desktop/test_penguin.jpg"
    # if os.path.exists(test_file):
    #     result = identify_penguin(test_file, penguin_model, img_transform, db_features, dataset.image_files)
    #     print(f"Match: {result['matched_file']} | Confidence: {result['confidence']}")