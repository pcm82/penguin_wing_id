import os
import torch
import timm
import pandas as pd
import streamlit as st
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from datetime import datetime
from sqlalchemy import create_engine, text

# --- CONFIGURATION ---
# Using your provided Supabase URI for persistent cloud storage
DB_URI = "postgresql://postgres:penguindatabase2026!@db.xozmbgbkbdzugsagwghf.supabase.co:5432/postgres"
DATA_DIR = './data/wings'
PORTRAIT_DIR = './data/portraits'

# Ensure local directories exist for temporary processing
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PORTRAIT_DIR, exist_ok=True)

# Database Engine
engine = create_engine(DB_URI)

# --- AI ENGINE (PURE PYTORCH) ---
@st.cache_resource
def load_biometric_model():
    """
    Loads the MegaDescriptor-T-224 model directly via timm.
    This bypasses the 'wildlife-tools' and 'faiss-gpu' dependency errors.
    """
    # Create the model using the Hugging Face hub reference
    model = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', pretrained=True, num_classes=0)
    model.eval()
    
    # Standard normalization for the MegaDescriptor model
    transform = T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return model, transform

def extract_embedding(image, model, transform):
    """Converts a PIL image into a normalized biometric vector (embedding)."""
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img_tensor)
    # L2 Normalization allows us to use simple matrix multiplication for similarity
    return F.normalize(embedding, p=2, dim=1)

@st.cache_data
def get_fingerprint_library(_model, _transform):
    """Generates a searchable bank of vectors from the reference wings folder."""
    valid_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not valid_files:
        return [], None
    
    vectors = []
    for fname in valid_files:
        img = Image.open(os.path.join(DATA_DIR, fname)).convert('RGB')
        vectors.append(extract_embedding(img, _model, _transform))
    
    # Concatenate all individual vectors into a single searchable tensor
    return valid_files, torch.cat(vectors)

# --- APP INITIALIZATION ---
st.set_page_config(page_title="King Penguin CMS", layout="wide")

# Load AI components once
main_model, main_transform = load_biometric_model()
filenames, library_vectors = get_fingerprint_library(main_model, main_transform)

# Initialize Session States
if 'current_view' not in st.session_state: 
    st.session_state.current_view = 'Dossier'
if 'is_admin' not in st.session_state: 
    st.session_state.is_admin = False

# Sidebar: Authentication
with st.sidebar:
    st.title("🔐 Authentication")
    if not st.session_state.is_admin:
        pwd = st.text_input("Admin Password", type="password")
        if st.button("Login"):
            if pwd == "penguinadmin":
                st.session_state.is_admin = True
                st.rerun()
            else: 
                st.error("Invalid Credentials")
    else:
        st.success("Admin Session Active")
        if st.button("Logout"):
            st.session_state.is_admin = False
            st.rerun()

# --- NAVIGATION ---
st.title("🐧 King Penguin Research Portal")
n1, n2, n3, n4, n5 = st.columns(5)
nav_items = [("🔍 Identify", "Identify"), ("🗂️ Dossier", "Dossier"), 
             ("🕒 Sightings", "Sightings"), ("📝 Edit", "Edit"), ("📜 Logs", "Logs")]

for i, (label, view) in enumerate(nav_items):
    if eval(f"n{i+1}").button(label):
        st.session_state.current_view = view

st.divider()

# --- VIEWS ---

if st.session_state.current_view == 'Identify':
    st.header("New Sighting Identification")
    up_file = st.file_uploader("Upload Wing Image", type=['jpg', 'png'])
    
    if up_file and library_vectors is not None:
        q_img = Image.open(up_file).convert('RGB')
        q_vec = extract_embedding(q_img, main_model, main_transform)
        
        # MANUAL COSINE SIMILARITY: Compare uploaded vector against library vectors
        # Resulting score ranges from -1 to 1 (usually 0.6+ for a good match)
        similarities = torch.mm(q_vec, library_vectors.t())
        confidence, best_idx = torch.max(similarities, dim=1)
        
        match_id = filenames[best_idx.item()]
        
        # Retrieve the individual's Display Name from Supabase
        with engine.connect() as conn:
            res = conn.execute(text("SELECT display_name FROM individuals WHERE id=:id"), {"id": match_id}).fetchone()
        match_name = res[0] if res else match_id

        # Display Comparison
        c1, c2 = st.columns(2)
        c1.image(q_img, caption="Recent Sighting", use_container_width=True)
        c2.image(os.path.join(DATA_DIR, match_id), caption=f"Match: {match_name}", use_container_width=True)
        st.metric("AI Confidence Score", f"{confidence.item()*100:.2f}%")

elif st.session_state.current_view == 'Dossier':
    st.header("🗂️ Population Gallery")
    with engine.connect() as conn:
        df_pop = pd.read_sql_query("SELECT * FROM individuals", conn)
    
    # RESTRICTED ADMIN SECTION
    if df_pop.empty:
        st.info("No individuals found in database.")
        if st.session_state.is_admin:
            if st.button("🚀 Auto-Register reference photos"):
                with engine.connect() as conn:
                    for fname in filenames:
                        conn.execute(text(
                            "INSERT INTO individuals (id, display_name, age, mother, father, notes) "
                            "VALUES (:id, :dn, 'Adult', 'Unknown', 'Unknown', 'Initial Import') "
                            "ON CONFLICT (id) DO NOTHING"), {"id": fname, "dn": fname})
                    conn.commit()
                st.success("Successfully registered filenames as base identities.")
                st.rerun()
        else:
            st.warning("Admin login required to initialize the population database.")
    else:
        # Render the gallery in a responsive grid
        for i in range(0, len(df_pop), 4):
            cols = st.columns(4)
            for j in range(4):
                if i + j < len(df_pop):
                    p = df_pop.iloc[i + j]
                    with cols[j]:
                        # Prefer a portrait if one has been uploaded, otherwise show reference wing
                        img_path = p['rep_image'] if p['rep_image'] else os.path.join(DATA_DIR, p['id'])
                        if os.path.exists(img_path):
                            st.image(img_path, use_container_width=True)
                        st.write(f"**{p['display_name']}**")
                        st.caption(f"Internal ID: {p['id']}")

elif st.session_state.current_view == 'Sightings':
    st.header("🕒 Sighting History")
    with engine.connect() as conn:
        df_sight = pd.read_sql_query("SELECT * FROM encounters ORDER BY id DESC", conn)
    st.dataframe(df_sight, use_container_width=True, hide_index=True)

elif st.session_state.current_view == 'Edit':
    st.header("📝 Metadata Management")
    with engine.connect() as conn:
        df_map = pd.read_sql_query("SELECT id, display_name FROM individuals", conn)
    
    if not df_map.empty:
        # Create a mapping for selecting by name rather than technical ID
        name_to_id = dict(zip(df_map['display_name'], df_map['id']))
        choice = st.selectbox("Select Individual", list(name_to_id.keys()))
        target_id = name_to_id[choice]
        
        with engine.connect() as conn:
            curr = conn.execute(text("SELECT * FROM individuals WHERE id=:id"), {"id": target_id}).fetchone()
        
        with st.form("metadata_form"):
            new_dn = st.text_input("Friendly Name", curr[1])
            new_age = st.selectbox("Life Stage", ["Chick", "Juvenile", "Adult"], index=2)
            new_notes = st.text_area("Field Notes", curr[5])
            
            if st.form_submit_button("Save Changes"):
                if st.session_state.is_admin:
                    with engine.connect() as conn:
                        conn.execute(text(
                            "UPDATE individuals SET display_name=:dn, age=:ag, notes=:nt WHERE id=:id"),
                            {"dn": new_dn, "ag": new_age, "nt": new_notes, "id": target_id})
                        conn.commit()
                    st.success("Metadata updated in Supabase.")
                else:
                    st.warning("Admin access is required to modify research data.")
    else:
        st.warning("Database is currently empty. Please register individuals first.")

elif st.session_state.current_view == 'Logs':
    st.header("📜 Audit Logs")
    if st.session_state.is_admin:
        with engine.connect() as conn:
            df_log = pd.read_sql_query("SELECT * FROM change_log ORDER BY id DESC", conn)
        st.dataframe(df_log, use_container_width=True)
    else:
        st.info("Log in as Admin to view the system audit trail.")