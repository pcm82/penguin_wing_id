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
                 (id TEXT PRIMARY KEY, display_name TEXT, age TEXT, 
                  mother TEXT, father TEXT, notes TEXT, rep_image TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS encounters
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, penguin_id TEXT, 
                  date TEXT, location TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS pending_changes
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, penguin_id TEXT, 
                  field TEXT, new_value TEXT, user_name TEXT, timestamp TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS change_log 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, penguin_id TEXT, 
                  field TEXT, old_value TEXT, new_value TEXT, timestamp TEXT)''')
    db_conn.commit()
    db_conn.close()

# --- APP UI ---
st.set_page_config(page_title="King Penguin Dossier", layout="wide")
init_db()

# Initialize Navigation and Admin State
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'Identify'
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False

# Global AI objects renamed to avoid Pylint W0621 shadowing [cite: 74]
main_model, main_extractor, main_transform = load_engine()
main_dataset = FlatImageDataset(DATA_DIR, transform=main_transform)
db_features = get_db_fingerprints(main_extractor, main_dataset)

st.title("🐧 King Penguin Identification Dossier")

# Admin Sidebar
with st.sidebar:
    if not st.session_state.is_admin:
        pwd = st.text_input("Admin Password", type="password")
        if st.button("Login"):
            if pwd == "penguinadmin":
                st.session_state.is_admin = True
                st.rerun()
    else:
        if st.button("Logout"):
            st.session_state.is_admin = False
            st.rerun()

# Navigation Bar
st.write("### 🧭 Navigation")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    if st.button("🔍 Identify & Log"):
        st.session_state.current_view = 'Identify'
with c2:
    if st.button("🗂️ View Dossier"):
        st.session_state.current_view = 'Dossier'
with c3:
    if st.button("🕒 View Sightings"):
        st.session_state.current_view = 'Sightings'
with c4:
    if st.button("📜 Change Log"):
        st.session_state.current_view = 'Logs'
with c5:
    if st.button("📝 Edit Dossier"):
        st.session_state.current_view = 'Edit'

st.divider()

# --- RENDER VIEWS ---
conn = sqlite3.connect(DB_PATH)

if st.session_state.current_view == 'Identify':
    st.header("Step 1: Upload New Sighting")
    up_file = st.file_uploader("Upload Wing Image", type=['jpg', 'jpeg', 'png'])
    if up_file:
        query_img = Image.open(up_file).convert('RGB')
        # Numeric 0 for FLIP_LEFT_RIGHT fixes Pylint E1101 [cite: 137, 759]
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
            st.image(Image.open(match_path), caption=f"Match: {matched_file}", use_container_width=True)
        st.metric("AI Confidence Score", f"{best_score.item()*100:.2f}%")
        
        st.divider()
        st.subheader("Step 3: Verify & Save")
        col_a, col_b = st.columns(2)
        with col_a:
            date_val = st.date_input("Sighting Date", datetime.now())
        with col_b:
            loc_val = st.text_input("Location", "Main Colony")
        if st.checkbox(f"Confirm this is {matched_file}?"):
            if st.button("Save Sighting to Dossier"):
                conn.execute(
                    "INSERT INTO encounters (penguin_id, date, location) VALUES (?,?,?)",
                    (matched_file, str(date_val), loc_val))
                conn.commit()
                st.success(f"Logged sighting for {matched_file}!")
        else:
            st.info("To register as a new penguin, use the 'Edit Dossier' tab.")

elif st.session_state.current_view == 'Dossier':
    st.header("🗂️ Individual Dossiers")
    df_inds = pd.read_sql_query("SELECT * FROM individuals", conn)
    st.dataframe(df_inds, use_container_width=True)
    
    # Restrict "Auto-Register" to Admin only 
    if st.session_state.is_admin:
        if st.button("🚀 Auto-Register All Wing Photos"):
            curr_ids = df_inds['id'].tolist() if not df_inds.empty else []
            new_count = 0
            for fname in os.listdir(DATA_DIR):
                if fname not in curr_ids and fname.lower().endswith(('.png', '.jpg')):
                    conn.execute(
                        "INSERT INTO individuals (id, display_name, age, mother, father, notes) VALUES (?,?,?,?,?,?)",
                        (fname, fname, "Adult", "Unknown", "Unknown", "Auto-Imported"))
                    new_count += 1
            conn.commit()
            st.success(f"Registered {new_count} new entries!")
            st.rerun()
    else:
        st.info("Admin login required to initialize the database gallery.")

elif st.session_state.current_view == 'Sightings':
    st.header("🕒 Sighting History")
    df_encs = pd.read_sql_query("SELECT * FROM encounters ORDER BY date DESC", conn)
    st.dataframe(df_encs, use_container_width=True)

elif st.session_state.current_view == 'Logs':
    st.header("📜 Audit Trail")
    if st.session_state.is_admin:
        df_logs = pd.read_sql_query("SELECT * FROM change_log ORDER BY timestamp DESC", conn)
        st.dataframe(df_logs, use_container_width=True)
    else:
        st.warning("Admin access required to view system logs.")

elif st.session_state.current_view == 'Edit':
    st.header("📝 Edit Metadata")
    df_ids = pd.read_sql_query("SELECT id FROM individuals", conn)
    if not df_ids.empty:
        target = st.selectbox("Select Penguin to Edit", df_ids['id'].tolist())
        curr = pd.read_sql_query(f"SELECT * FROM individuals WHERE id='{target}'", conn).iloc[0]
        
        with st.form("edit_form"):
            n_name = st.text_input("Display Name", curr['display_name'] if curr['display_name'] else "")
            n_age = st.selectbox("Age", ["Chick", "Juvenile", "Adult"], 
                                index=["Chick", "Juvenile", "Adult"].index(curr['age']))
            n_mom = st.text_input("Mother ID", curr['mother'])
            n_dad = st.text_input("Father ID", curr['father'])
            n_notes = st.text_area("Notes", curr['notes'])
            
            if st.form_submit_button("Submit Changes"):
                if st.session_state.is_admin:
                    # Direct admin update
                    conn.execute(
                        "UPDATE individuals SET display_name=?, age=?, mother=?, father=?, notes=? WHERE id=?",
                        (n_name, n_age, n_mom, n_dad, n_notes, target))
                    conn.commit()
                    st.success("Changes saved by admin.")
                    st.rerun()
                else:
                    # Submit for review
                    conn.execute(
                        "INSERT INTO pending_changes (penguin_id, field, new_value, timestamp) VALUES (?,?,?,?)",
                        (target, "display_name", n_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    conn.commit()
                    st.info("Proposals submitted for admin review.")
    else:
        st.warning("No individuals found. Please register them in the 'Dossier' tab.")

conn.close()