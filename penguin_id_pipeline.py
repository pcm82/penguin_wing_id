import os
import gdown
import numpy as np
import timm
from wildlife_tools.data import WildlifeDataset
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier
import torchvision.transforms as T

# 1. DOWNLOAD DATA FROM GOOGLE DRIVE
# Replace with your shared folder ID (Anyone with the link can view)
FOLDER_ID = 'YOUR_GOOGLE_DRIVE_FOLDER_ID'
DATA_DIR = './penguin_data'

if not os.path.exists(DATA_DIR):
    print("Downloading photos from Google Drive...")
    gdown.download_folder(id=FOLDER_ID, output=DATA_DIR, quiet=False)

# 2. SETUP DATASET & PRE-PROCESSING
# We resize images to 224x224 for the ML model
transform = T.Compose([
    T.Resize([224, 224]),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Assuming your folder structure is: penguin_data/individual_name/photo.jpg
dataset = WildlifeDataset.from_folder(root=DATA_DIR, transform=transform)

# 3. FEATURE EXTRACTION (The "Biometric Fingerprint")
# We use MegaDescriptor, a model trained specifically for individual animal ID
print("Extracting features using MegaDescriptor...")
extractor = DeepFeatures(timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', pretrained=True, num_classes=0))
features = extractor(dataset)

# 4. IDENTIFICATION (Matching a new photo)
def identify_penguin(image_path):
    """
    Takes a new photo and finds the closest match in your database.
    """
    # Load and process the new image
    query_dataset = WildlifeDataset.from_list([image_path], transform=transform)
    query_features = extractor(query_dataset)
    
    # Calculate similarity between new photo and known database
    similarity_func = CosineSimilarity()
    sim_matrix = similarity_func(query_features, features)
    
    # Use K-Nearest Neighbors to find the ID
    classifier = KnnClassifier(k=1, database_labels=dataset.labels_string)
    prediction = classifier(sim_matrix['cosine'])
    
    return prediction[0]

# Example Usage:
# print(f"Identified as: {identify_penguin('new_sighting_wing.jpg')}")