"""
Penguin Wing Identification App
A Streamlit-based tool for biometric identification and dossier management.
"""

import os
import sqlite3
from datetime import datetime
import streamlit as st
import pandas as pd
import torch
import timm
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from wildlife_tools.features import DeepFeatures

# --- CONFIGURATION & PATHS ---
DATA_DIR = './data/wings'
DB_PATH = "./data/database.db"

# Ensure data directory exists
os.makedirs("./data", exist_ok=True)

# --- CACHED AI ENGINE ---
@st.cache_resource
def load_engine():
    """Loads the model once and keeps it in memory."""
    biometric_model = timm.create_model(
        'hf-hub:BVRA/MegaDescriptor-T-224',
        pretrained=True,
        num_classes=0
    )
    biometric_model.eval()
    # Use num_workers=0 for Windows stability
    model_extractor = DeepFeatures(biometric_model, num_workers=0)
    model_transform = T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return biometric_model, model_extractor, model_transform

class FlatImageDataset(Dataset):
    """Dataset for loading images from a flat directory structure."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.image_files[idx])
        img_item = Image.open(path).convert('RGB')
        if self.transform:
            img_item = self.transform(img_item)
        return img_item, self.image_files[idx]

@st.cache_data
def get_db_fingerprints(_extractor, _dataset):
    """Caches the fingerprints of your entire library."""
    features = _extractor(_dataset)
    if not isinstance(features, torch.Tensor):
        return torch.from_numpy(features)
    return features

# --- DATABASE LOGIC ---
def init_db():
    """Initializes the SQLite database with required tables."""
    db_conn = sqlite3.connect(DB_PATH)
    cursor = db_conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS individuals
                 (id TEXT PRIMARY KEY, age TEXT, mother TEXT, father TEXT, notes TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS encounters
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, penguin_id TEXT, 
                  date TEXT, location TEXT)''')
    # Table to track every edit made to a penguin's dossier
    cursor.execute('''CREATE TABLE IF NOT EXISTS change_log
             (id INTEGER PRIMARY KEY AUTOINCREMENT, penguin_id TEXT, 
              field TEXT, old_value TEXT, new_value TEXT, timestamp TEXT)''')
    db_conn.commit()
    db_conn.close()

# --- APP UI ---
st.set_page_config(page_title="King Penguin Dossier", layout="wide")
init_db()

# Initialize AI with unique names
main_model, main_extractor, main_transform = load_engine()
main_dataset = FlatImageDataset(DATA_DIR, transform=main_transform)
db_features = get_db_fingerprints(main_extractor, main_dataset)

st.title("🐧 King Penguin Identification Dossier")

menu = ["Identify & Log", "View Dossier"]
choice = st.sidebar.selectbox("Action", menu)

if choice == "Identify & Log":
    st.header("Step 1: Upload New Sighting")
    up_file = st.file_uploader("Upload Wing Image", type=['jpg', 'jpeg', 'png'])
    if up_file:
        query_img = Image.open(up_file).convert('RGB')
        # Use numeric constant 0 for FLIP_LEFT_RIGHT to avoid Pylint E1101
        query_flip = query_img.transpose(0)
        t_orig = main_transform(query_img).unsqueeze(0)
        t_flip = main_transform(query_flip).unsqueeze(0)
        with torch.no_grad():
            f_orig = main_model(t_orig)
            f_flip = main_model(t_flip)
        sim_orig = torch.cosine_similarity(f_orig, db_features)
        sim_flip = torch.cosine_similarity(f_flip, db_features)
        m_orig, i_orig = torch.max(sim_orig, dim=0)
        m_flip, i_flip = torch.max(sim_flip, dim=0)
        best_score = m_orig if m_orig > m_flip else m_flip
        best_idx = i_orig if m_orig > m_flip else i_flip
        matched_file = main_dataset.image_files[best_idx.item()]
        st.subheader("Step 2: Compare Match")
        col1, col2 = st.columns(2)
        with col1:
            st.image(query_img, caption="New Sighting", use_container_width=True)
        with col2:
            match_path = os.path.join(DATA_DIR, matched_file)
            st.image(Image.open(match_path),
                     caption=f"Match: {matched_file}", use_container_width=True)
        st.metric("AI Confidence Score", f"{best_score.item()*100:.2f}%")
        st.divider()
        st.subheader("Step 3: Verify & Save")
        c_a, c_b = st.columns(2)
        with c_a:
            date_val = st.date_input("Sighting Date", datetime.now())
        with c_b:
            loc_val = st.text_input("Location", "Main Colony")
        if st.checkbox(f"Confirm this is {matched_file}?"):
            if st.button("Save Sighting to Dossier"):
                conn = sqlite3.connect(DB_PATH)
                conn.execute(
                    "INSERT INTO encounters (penguin_id, date, location) VALUES (?,?,?)",
                    (matched_file, str(date_val), loc_val))
                conn.commit()
                conn.close()
                st.success(f"Logged sighting for {matched_file}!")
        else:
            st.info("To register as a new penguin, use the View Dossier tab.")

elif choice == "View Dossier":
    st.header("Search Penguin Records")
    conn = sqlite3.connect(DB_PATH)
    history = pd.read_sql_query("SELECT * FROM encounters", conn)
    st.dataframe(history, use_container_width=True)
    conn.close()
