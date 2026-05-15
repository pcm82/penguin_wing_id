import os
import time
import torch
import timm
import pandas as pd
import streamlit as st
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from supabase import create_client

# --- 1. CONFIG & CLIENTS ---
URL = st.secrets["SUPABASE_URL"]
KEY = st.secrets["SUPABASE_KEY"]
DB_URI = st.secrets["DATABASE_URL"]

engine = create_engine(DB_URI, poolclass=NullPool, connect_args={"sslmode": "require"})
sb_storage = create_client(URL, KEY)

DATA_DIR = './data/wings'

# --- 2. DATABASE SCHEMA ---
def init_db():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS individuals (
                id TEXT PRIMARY KEY,
                display_name TEXT,
                age TEXT DEFAULT 'Unknown',
                notes TEXT,
                rep_image_url TEXT,
                tags TEXT
            )
        """))
        conn.commit()

init_db()

# --- 3. AI ENGINE (Cached for speed) ---
@st.cache_resource
def load_model():
    model = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', pretrained=True, num_classes=0)
    model.eval()
    return model, T.Compose([
        T.Resize([224, 224]), T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

# --- 4. APP UI ---
st.set_page_config(page_title="King Penguin Cloud CMS", layout="wide")

if 'current_view' not in st.session_state: st.session_state.current_view = 'Dossier'
if 'is_admin' not in st.session_state: st.session_state.is_admin = False
if 'az_filter' not in st.session_state: st.session_state.az_filter = "All"

# Sidebar
with st.sidebar:
    st.title("🔐 Admin Access")
    if not st.session_state.is_admin:
        if st.text_input("Password", type="password") == "penguinadmin":
            st.session_state.is_admin = True
            st.rerun()
    else:
        st.success("Admin Active")
        if st.button("Logout"): st.session_state.is_admin = False; st.rerun()

st.title("🐧 Penguin Research Portal")
n1, n2, n3, n4, n5 = st.columns(5)
navs = [("🔍 Identify", "Identify"), ("🗂️ Dossier", "Dossier"), ("🕒 Sightings", "Sightings"), ("📝 Edit", "Edit"), ("📜 Logs", "Logs")]
for i, (label, view) in enumerate(navs):
    if eval(f"n{i+1}").button(label): st.session_state.current_view = view

st.divider()

# --- 5. DOSSIER WITH SEARCH & A-Z SLIDER ---
if st.session_state.current_view == 'Dossier':
    # A. Search and Filter Controls
    c1, c2 = st.columns([1, 3])
    with c1:
        search_query = st.text_input("🔍 Search by Name, ID, or Notes", "").lower()
    with c2:
        st.write("Filter by Initial:")
        alphabet = ["All"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        az_choice = st.select_slider("A-Z Navigation", options=alphabet, value=st.session_state.az_filter)
        st.session_state.az_filter = az_choice

    # B. Fetch Data from Supabase
    with engine.connect() as conn:
        df = pd.read_sql_query("SELECT * FROM individuals ORDER BY display_name ASC", conn)

    # C. Apply Filters in Memory
    if not df.empty:
        # A-Z Filtering
        if st.session_state.az_filter != "All":
            df = df[df['display_name'].str.upper().str.startswith(st.session_state.az_filter)]
        
        # Search Filtering
        if search_query:
            df = df[
                df['display_name'].str.lower().contains(search_query) | 
                df['id'].str.lower().contains(search_query) |
                df['notes'].str.lower().contains(search_query, na=False)
            ]

        # D. Display Results
        if df.empty:
            st.warning("No penguins match your current filters.")
        else:
            st.write(f"Showing {len(df)} individuals")
            for i in range(0, len(df), 4):
                cols = st.columns(4)
                for j in range(4):
                    if i + j < len(df):
                        row = df.iloc[i + j]
                        with cols[j]:
                            # Use Supabase URL, fallback to local path if URL is missing
                            img_url = row['rep_image_url'] if row['rep_image_url'] else os.path.join(DATA_DIR, row['id'])
                            st.image(img_url, use_container_width=True)
                            st.subheader(row['display_name'])
                            st.caption(f"ID: {row['id']} | Age: {row['age']}")
                            if st.button(f"View Profile", key=f"view_{row['id']}"):
                                st.session_state.selected_penguin = row['id']
                                # Handle drill-down logic here
    else:
        st.info("The database is currently empty. Head to the 'Edit' tab or use Admin tools to sync photos.")

# --- 6. ADMIN MIGRATION TOOL (IN EDIT TAB) ---
elif st.session_state.current_view == 'Edit':
    st.header("📝 Database Management")
    if st.session_state.is_admin:
        with st.expander("🚀 Cloud Migration Tool (Run once)"):
            st.write("This will upload your GitHub photos to Supabase Storage and link them to the database.")
            if st.button("Start Migration"):
                local_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                progress = st.progress(0)
                
                with engine.connect() as conn:
                    for idx, fname in enumerate(local_files):
                        path = os.path.join(DATA_DIR, fname)
                        with open(path, 'rb') as f:
                            try:
                                sb_storage.storage.from_('penguin-photos').upload(fname, f)
                            except: pass # Skip if already uploaded
                        
                        public_url = sb_storage.storage.from_('penguin-photos').get_public_url(fname)
                        conn.execute(text("""
                            INSERT INTO individuals (id, display_name, rep_image_url) 
                            VALUES (:id, :dn, :url) ON CONFLICT (id) DO UPDATE SET rep_image_url = :url
                        """), {"id": fname, "dn": fname, "url": public_url})
                        progress.progress((idx + 1) / len(local_files))
                    conn.commit()
                st.success("Migration complete!")