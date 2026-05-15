import os
import time
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
if "DATABASE_URL" in st.secrets:
    DB_URI = st.secrets["DATABASE_URL"]
else:
    # Local fallback
    DB_URI = "postgresql://postgres.xozmbgbkbdzugsagwghf:penguindatabase2026%21@aws-1-us-east-2.pooler.supabase.com:6543/postgres"

engine = create_engine(DB_URI, poolclass=NullPool, connect_args={"sslmode": "require"})

DATA_DIR = './data/wings'
os.makedirs(DATA_DIR, exist_ok=True)

# --- 2. DATABASE SCHEMA SETUP ---
def init_db():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS individuals (
                id TEXT PRIMARY KEY,
                display_name TEXT,
                age TEXT DEFAULT 'Adult',
                notes TEXT,
                rep_image TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS encounters (
                id SERIAL PRIMARY KEY,
                penguin_id TEXT REFERENCES individuals(id),
                date_observed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                location TEXT DEFAULT 'Unknown',
                notes TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS change_log (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                action TEXT,
                details TEXT
            )
        """))
        conn.commit()

try:
    init_db()
except Exception as e:
    st.error(f"Schema Setup Error: {e}")

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

# --- 4. UTILITY FUNCTIONS ---
def generate_unique_id():
    """Generates an immutable ID based on the current epoch."""
    return f"PENG-{int(time.time())}"

def merge_penguins(dup_id, target_id):
    """Moves sightings to target and removes duplicate record."""
    with engine.connect() as conn:
        conn.execute(text("UPDATE encounters SET penguin_id=:t WHERE penguin_id=:d"), {"t": target_id, "d": dup_id})
        conn.execute(text("DELETE FROM individuals WHERE id=:d"), {"d": dup_id})
        conn.execute(text("INSERT INTO change_log (action, details) VALUES ('MERGE', :det)"), 
                     {"det": f"Merged {dup_id} into {target_id}"})
        conn.commit()

# --- 5. APP INITIALIZATION ---
st.set_page_config(page_title="King Penguin CMS", layout="wide")
main_model, main_transform = load_biometric_model()
filenames, library_vectors = get_fingerprint_library(main_model, main_transform)

if 'current_view' not in st.session_state: st.session_state.current_view = 'Dossier'
if 'is_admin' not in st.session_state: st.session_state.is_admin = False
if 'selected_penguin' not in st.session_state: st.session_state.selected_penguin = None

with st.sidebar:
    st.title("🔐 Admin Login")
    if not st.session_state.is_admin:
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if pwd == "penguinadmin":
                st.session_state.is_admin = True
                st.rerun()
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
    if eval(f"n{i+1}").button(label):
        st.session_state.current_view = view
        st.session_state.selected_penguin = None

st.divider()

# --- 6. VIEW LOGIC ---

if st.session_state.current_view == 'Identify':
    st.header("Search & Register")
    up_file = st.file_uploader("Upload Image", type=['jpg', 'png'])
    if up_file and library_vectors is not None:
        q_img = Image.open(up_file).convert('RGB')
        q_vec = extract_embedding(q_img, main_model, main_transform)
        similarities = torch.mm(q_vec, library_vectors.t())
        score, best_idx = torch.max(similarities, dim=1)
        match_id = filenames[best_idx.item()]
        
        with engine.connect() as conn:
            res = conn.execute(text("SELECT display_name FROM individuals WHERE id=:id"), {"id": match_id}).fetchone()
        match_name = res[0] if res else "Unknown"

        c1, c2 = st.columns(2)
        c1.image(q_img, caption="Recent Sighting", use_container_width=True)
        c2.image(os.path.join(DATA_DIR, match_id), caption=f"Top Match: {match_name}", use_container_width=True)
        st.metric("AI Confidence", f"{score.item()*100:.2f}%")
        
        if st.session_state.is_admin:
            if st.button("🆕 Register as New Individual"):
                new_id = generate_unique_id()
                with engine.connect() as conn:
                    conn.execute(text("INSERT INTO individuals (id, display_name) VALUES (:id, :dn)"), 
                                 {"id": new_id, "dn": f"New Penguin {new_id}"})
                    conn.commit()
                st.success(f"Registered as {new_id}!")

elif st.session_state.current_view == 'Dossier':
    if st.session_state.selected_penguin:
        p_id = st.session_state.selected_penguin
        with engine.connect() as conn:
            p = conn.execute(text("SELECT * FROM individuals WHERE id=:id"), {"id": p_id}).fetchone()
        
        if st.button("⬅️ Back"):
            st.session_state.selected_penguin = None
            st.rerun()
            
        st.header(f"Profile: {p[1]}")
        c1, c2 = st.columns([1, 2])
        with c1:
            img = p[4] if p[4] else os.path.join(DATA_DIR, p[0])
            if os.path.exists(img): st.image(img, use_container_width=True)
            st.write(f"**Age:** {p[2]} years")
        with c2:
            st.subheader("Associated Records")
            # Logic to find images starting with the same ID prefix
            base = p[0].split('.')[0]
            related = [f for f in filenames if f.startswith(base)]
            grid = st.columns(3)
            for idx, r in enumerate(related):
                r_path = os.path.join(DATA_DIR, r)
                with grid[idx % 3]:
                    st.image(r_path, use_container_width=True)
                    if st.session_state.is_admin:
                        if st.button("Set Profile", key=f"p_{r}"):
                            with engine.connect() as conn:
                                conn.execute(text("UPDATE individuals SET rep_image=:img WHERE id=:id"), {"img": r_path, "id": p_id})
                                conn.commit()
                            st.rerun()
    else:
        st.header("Population Gallery")
        with engine.connect() as conn:
            df_pop = pd.read_sql_query("SELECT * FROM individuals", conn)
        if df_pop.empty:
            if st.session_state.is_admin and st.button("🚀 Initial Sync"):
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
                        row = df_pop.iloc[i+j]
                        with cols[j]:
                            img = row['rep_image'] if row['rep_image'] else os.path.join(DATA_DIR, row['id'])
                            if os.path.exists(img): st.image(img, use_container_width=True)
                            if st.button(f"View {row['display_name']}", key=f"v_{row['id']}"):
                                st.session_state.selected_penguin = row['id']
                                st.rerun()

elif st.session_state.current_view == 'Edit':
    st.header("Bookkeeping")
    if st.session_state.is_admin:
        with engine.connect() as conn:
            df_m = pd.read_sql_query("SELECT id, display_name FROM individuals", conn)
        
        t1, t2 = st.tabs(["Update Data", "Merge/Consolidate"])
        with t1:
            choice = st.selectbox("Select Individual", df_m['display_name'])
            target_id = df_m[df_m['display_name'] == choice]['id'].values[0]
            with st.form("edit"):
                new_name = st.text_input("New Name")
                new_age = st.text_input("Age (Years)")
                if st.form_submit_button("Save"):
                    with engine.connect() as conn:
                        conn.execute(text("UPDATE individuals SET display_name=:n, age=:a WHERE id=:i"), 
                                     {"n": new_name, "a": new_age, "i": target_id})
                        conn.commit()
                    st.success("Record updated.")
        with t2:
            st.warning("Warning: Merging will delete the duplicate and move all history to the target.")
            dup = st.selectbox("Duplicate Record (Delete)", df_m['display_name'])
            target = st.selectbox("Primary Record (Keep)", df_m['display_name'])
            if st.button("Consolidate Records"):
                if dup != target:
                    d_id = df_m[df_m['display_name'] == dup]['id'].values[0]
                    t_id = df_m[df_m['display_name'] == target]['id'].values[0]
                    merge_penguins(d_id, t_id)
                    st.success("Merged successfully.")
                    st.rerun()
    else: st.warning("Admin login required.")