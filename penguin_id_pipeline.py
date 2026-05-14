"""
Module for identifying individual King Penguins using biometric wing patterns.
Uses MegaDescriptor for feature extraction and Cosine Similarity for matching.
"""

import os
import gdown
import timm
import torchvision.transforms as T
from wildlife_tools.data import WildlifeDataset
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier

# 1. DOWNLOAD DATA FROM GOOGLE DRIVE
FOLDER_ID = 'YOUR_GOOGLE_DRIVE_FOLDER_ID'
DATA_DIR = './penguin_data'

if not os.path.exists(DATA_DIR):
    print("Downloading photos from Google Drive...")
    gdown.download_folder(id=FOLDER_ID, output=DATA_DIR, quiet=False)

# 2. SETUP DATASET & PRE-PROCESSING
transform = T.Compose([
    T.Resize([224, 224]),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# pylint: disable=no-member
dataset = WildlifeDataset.from_folder(root=DATA_DIR, transform=transform)

# 3. FEATURE EXTRACTION
print("Extracting features using MegaDescriptor...")
MODEL_NAME = 'hf-hub:BVRA/MegaDescriptor-T-224'
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
extractor = DeepFeatures(model)
features = extractor(dataset)

# 4. IDENTIFICATION
def identify_penguin(image_path):
    """
    Takes a new photo and finds the closest match in the database.
    """
    query_dataset = WildlifeDataset([image_path], transform=transform)
    query_features = extractor(query_dataset)

    similarity_func = CosineSimilarity()
    sim_matrix = similarity_func(query_features, features)

    classifier = KnnClassifier(k=1, database_labels=dataset.labels_string)
    prediction = classifier(sim_matrix['cosine'])

    return prediction[0]

if __name__ == "__main__":
    # Example usage placeholder
    pass
