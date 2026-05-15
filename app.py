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
from sqlalchemy.pool import NullPool

# --- 1. DATABASE CONFIGURATION ---
# Optimized for Supabase Transaction Pooler (Port 6543)
if "DATABASE_URL" in st.secrets:
    DB_URI = st.secrets["DATABASE_URL"]
else:
    # Ensure '!' is encoded as '%21' in the password
    DB_URI = "postgresql://postgres.xozmbgbkbdzugsagwghf:penguindatabase2026%21@aws-1-us-east-2.pooler.supabase.com:6543/postgres"

engine = create_engine(
    DB_URI,
    poolclass=NullPool,
    connect_args={"sslmode": "require"}
)

DATA_DIR = './data/wings'
os.makedirs(DATA_DIR, exist_ok=True)

# --- 2. DATABASE SCHEMA INITIALIZATION ---
def init_db():
    """Builds the database tables if they do not exist."""
    with engine.connect() as conn:
        # Create individuals table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS individuals (
                id TEXT PRIMARY KEY,
                display_name TEXT,
                age TEXT DEFAULT 'Adult',
                mother TEXT DEFAULT 'Unknown',
                father TEXT DEFAULT 'Unknown',
                notes TEXT,
                rep_image TEXT
            )
        """))
        # Create sightings table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS encounters (
                id SERIAL PRIMARY KEY,
                penguin_id TEXT REFERENCES individuals(id),
                date_observed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                location TEXT DEFAULT 'Unknown',
                observer TEXT DEFAULT 'Unknown',
                notes TEXT
            )
        """))
        # Create audit logs table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS change_log (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                action TEXT,
                details TEXT
            )
        """))
        conn.commit()

# Run initialization immediately
try:
    init_db()
except Exception as e:
    st.error(f"Database Schema Setup Error: {e}")

# --- 3. AI ENGINE (PURE PYTORCH) ---
@st.cache_resource
def load_biometric_model():
    model = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', pretrained=True, num_classes=0)
    model.eval()
    transform = T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return model, transform

def extract_embedding(image, model, transform):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img_tensor)
    return F.normalize(embedding, p=2, dim=1)

@st.cache_data
def get_fingerprint_library(_model, _transform):
    valid_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not valid_files: return [], None
    vectors = []
    for fname in valid_files:
        img = Image.open(os.path.join(DATA_DIR, fname)).convert('RGB')
        vectors.append(extract_embedding(img, _model, _transform))
    return valid_files, torch.cat(vectors)

# --- 4. APP INITIALIZATION ---
st.set_page_config(page_title="King Penguin CMS", layout="wide")
main_model, main_transform = load_biometric_model()
filenames, library_vectors = get_fingerprint_library(main_model, main_transform)

if 'current_view' not in st.session_state: st.session_state.current_view = 'Dossier'
if 'is_admin' not in st.session_state: st.session_state.is_admin = False

with st.sidebar:
    st.title("🔐 Admin Access")
    if not st.session_state.is_admin:
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if pwd == "penguinadmin":
                st.session_state.is_admin = True
                st.rerun()
            else: st.error("Access Denied")
    else:
        st.success("Admin Session Active")
        if st.button("Logout"):
            st.session_state.is_admin = False
            st.rerun()

# Navigation
st.title("🐧 Research Portal")
n1, n2, n3, n4, n5 = st.columns(5)
navs = [("🔍 Identify", "Identify"), ("🗂️ Dossier", "Dossier"), ("🕒 Sightings", "Sightings"), ("📝 Edit", "Edit"), ("📜 Logs", "Logs")]
for i, (label, view) in enumerate(navs):
    if eval(f"n{i+1}").button(label): st.session_state.current_view = view
st.divider()

# --- 5. VIEW LOGIC ---

if st.session_state.current_view == 'Identify':
    st.header("Sighting Identification")
    up_file = st.file_uploader("Upload Image", type=['jpg', 'png'])
    if up_file and library_vectors is not None:
        q_img = Image.open(up_file).convert('RGB')
        q_vec = extract_embedding(q_img, main_model, main_transform)
        similarities = torch.mm(q_vec, library_vectors.t())
        score, best_idx = torch.max(similarities, dim=1)
        match_id = filenames[best_idx.item()]
        
        with engine.connect() as conn:
            res = conn.execute(text("SELECT display_name FROM individuals WHERE id=:id"), {"id": match_id}).fetchone()
        match_name = res[0] if res else match_id
        
        c1, c2 = st.columns(2)
        c1.image(q_img, caption="Query", use_container_width=True)
        c2.image(os.path.join(DATA_DIR, match_id), caption=f"Match: {match_name}", use_container_width=True)
        st.metric("Confidence", f"{score.item()*100:.2f}%")

elif st.session_state.current_view == 'Dossier':
    st.header("Population Dossier")
    try:
        with engine.connect() as conn:
            df_pop = pd.read_sql_query("SELECT * FROM individuals", conn)
        
        if df_pop.empty:
            st.info("Population empty.")
            if st.session_state.is_admin:
                if st.button("🚀 Auto-Register Photos"):
                    with engine.connect() as conn:
                        for f in filenames:
                            conn.execute(text("INSERT INTO individuals (id, display_name) VALUES (:id, :dn)"), {"id": f, "dn": f})
                        conn.commit()
                    st.rerun()
        else:
            for i in range(0, len(df_pop), 4):
                cols = st.columns(4)
                for j in range(4):
                    if i+j < len(df_pop):
                        p = df_pop.iloc[i+j]
                        img_path = os.path.join(DATA_DIR, p['id'])
                        if os.path.exists(img_path): cols[j].image(img_path, use_container_width=True)
                        cols[j].write(f"**{p['display_name']}**")
    except Exception as e: st.error(f"Dossier Error: {e}")

elif st.session_state.current_view == 'Sightings':
    st.header("Sighting History")
    try:
        with engine.connect() as conn:
            df_sight = pd.read_sql_query("SELECT * FROM encounters ORDER BY id DESC", conn)
        st.dataframe(df_sight, use_container_width=True, hide_index=True)
    except Exception as e: st.error(f"Sightings Error: {e}")

elif st.session_state.current_view == 'Edit':
    st.header("Manage Metadata")
    if st.session_state.is_admin:
        try:
            with engine.connect() as conn:
                df_map = pd.read_sql_query("SELECT id, display_name FROM individuals", conn)
            if not df_map.empty:
                choice = st.selectbox("Select Penguin", df_map['display_name'])
                target_id = df_map[df_map['display_name'] == choice]['id'].values[0]
                with st.form("meta_form"):
                    new_dn = st.text_input("New Display Name")
                    if st.form_submit_button("Save"):
                        with engine.connect() as conn:
                            conn.execute(text("UPDATE individuals SET display_name=:dn WHERE id=:id"), {"dn": new_dn, "id": target_id})
                            conn.commit()
                        st.success("Metadata updated.")
            else: st.info("No records to edit.")
        except Exception as e: st.error(f"Edit Error: {e}")
    else: st.warning("Admin access required.")

elif st.session_state.current_view == 'Logs':
    st.header("Audit Logs")
    if st.session_state.is_admin:
        try:
            with engine.connect() as conn:
                df_log = pd.read_sql_query("SELECT * FROM change_log ORDER BY id DESC", conn)
            st.dataframe(df_log, use_container_width=True)
        except Exception as e: st.error(f"Logs Error: {e}")
    else: st.info("Admin login required.")