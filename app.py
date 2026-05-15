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
# We use NullPool because Supabase handles the pooling on its end via port 6543
if "DATABASE_URL" in st.secrets:
    DB_URI = st.secrets["DATABASE_URL"]
else:
    # Fallback for local dev - ensure the '!' in your password is encoded as '%21'
    DB_URI = "postgresql://postgres.xozmbgbkbdzugsagwghf:penguindatabase2026%21@aws-1-us-east-2.pooler.supabase.com:6543/postgres?prepared_statements=false"

engine = create_engine(
    DB_URI,
    poolclass=NullPool,
    connect_args={"sslmode": "require"}
)

DATA_DIR = './data/wings'
PORTRAIT_DIR = './data/portraits'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PORTRAIT_DIR, exist_ok=True)

# --- 2. AI ENGINE (PURE PYTORCH) ---
@st.cache_resource
def load_biometric_model():
    """Loads MegaDescriptor directly via timm to avoid dependency conflicts."""
    model = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', pretrained=True, num_classes=0)
    model.eval()
    
    transform = T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return model, transform

def extract_embedding(image, model, transform):
    """Converts an image into a normalized biometric vector."""
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img_tensor)
    return F.normalize(embedding, p=2, dim=1)

@st.cache_data
def get_fingerprint_library(_model, _transform):
    """Scans local wings folder and builds a searchable tensor bank."""
    valid_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not valid_files:
        return [], None
    
    vectors = []
    for fname in valid_files:
        img = Image.open(os.path.join(DATA_DIR, fname)).convert('RGB')
        vectors.append(extract_embedding(img, _model, _transform))
    
    return valid_files, torch.cat(vectors)

# --- 3. APP INITIALIZATION ---
st.set_page_config(page_title="King Penguin CMS", layout="wide")

main_model, main_transform = load_biometric_model()
filenames, library_vectors = get_fingerprint_library(main_model, main_transform)

if 'current_view' not in st.session_state: st.session_state.current_view = 'Dossier'
if 'is_admin' not in st.session_state: st.session_state.is_admin = False

# Sidebar Login
with st.sidebar:
    st.title("🔐 Authentication")
    if not st.session_state.is_admin:
        pwd = st.text_input("Admin Password", type="password")
        if st.button("Login"):
            if pwd == "penguinadmin":
                st.session_state.is_admin = True
                st.rerun()
            else: st.error("Invalid Credentials")
    else:
        st.success("Admin Active")
        if st.button("Logout"):
            st.session_state.is_admin = False
            st.rerun()

# Navigation
st.title("🐧 King Penguin Research Portal")
n1, n2, n3, n4, n5 = st.columns(5)
nav_items = [("🔍 Identify", "Identify"), ("🗂️ Dossier", "Dossier"), 
             ("🕒 Sightings", "Sightings"), ("📝 Edit", "Edit"), ("📜 Logs", "Logs")]

for i, (label, view) in enumerate(nav_items):
    if eval(f"n{i+1}").button(label):
        st.session_state.current_view = view

st.divider()

# --- 4. VIEWS ---

if st.session_state.current_view == 'Identify':
    st.header("New Sighting Identification")
    up_file = st.file_uploader("Upload Wing Photo", type=['jpg', 'png'])
    
    if up_file and library_vectors is not None:
        q_img = Image.open(up_file).convert('RGB')
        q_vec = extract_embedding(q_img, main_model, main_transform)
        
        # Biometric Matching (Cosine Similarity via Matrix Multiplication)
        similarities = torch.mm(q_vec, library_vectors.t())
        confidence, best_idx = torch.max(similarities, dim=1)
        
        match_id = filenames[best_idx.item()]
        
        with engine.connect() as conn:
            res = conn.execute(text("SELECT display_name FROM individuals WHERE id=:id"), {"id": match_id}).fetchone()
        match_name = res[0] if res else match_id

        c1, c2 = st.columns(2)
        c1.image(q_img, caption="Recent Sighting", use_container_width=True)
        c2.image(os.path.join(DATA_DIR, match_id), caption=f"Top Match: {match_name}", use_container_width=True)
        st.metric("AI Confidence", f"{confidence.item()*100:.2f}%")

elif st.session_state.current_view == 'Dossier':
    st.header("🗂️ Population Gallery")
    try:
        with engine.connect() as conn:
            df_pop = pd.read_sql_query("SELECT * FROM individuals", conn)
        
        if df_pop.empty:
            st.info("No individuals registered in Supabase.")
            # ONLY ADMINS CAN REGISTER
            if st.session_state.is_admin:
                if st.button("🚀 Auto-Register reference photos"):
                    with engine.connect() as conn:
                        for fname in filenames:
                            conn.execute(text(
                                "INSERT INTO individuals (id, display_name, age) "
                                "VALUES (:id, :dn, 'Adult') ON CONFLICT DO NOTHING"), 
                                {"id": fname, "dn": fname})
                        conn.commit()
                    st.rerun()
            else:
                st.warning("Admin login required to initialize population data.")
        else:
            for i in range(0, len(df_pop), 4):
                cols = st.columns(4)
                for j in range(4):
                    if i + j < len(df_pop):
                        p = df_pop.iloc[i + j]
                        with cols[j]:
                            img_p = p['rep_image'] if p['rep_image'] else os.path.join(DATA_DIR, p['id'])
                            if os.path.exists(img_p): st.image(img_p, use_container_width=True)
                            st.write(f"**{p['display_name']}**")
                            st.caption(f"ID: {p['id']}")
    except Exception as e:
        st.error(f"Database Error: {e}")

elif st.session_state.current_view == 'Sightings':
    st.header("🕒 Sighting History")
    with engine.connect() as conn:
        df_sight = pd.read_sql_query("SELECT * FROM encounters ORDER BY id DESC", conn)
    st.dataframe(df_sight, use_container_width=True, hide_index=True)

elif st.session_state.current_view == 'Edit':
    st.header("📝 Metadata Management")
    if not st.session_state.is_admin:
        st.warning("Admin access required to edit records.")
    else:
        with engine.connect() as conn:
            df_map = pd.read_sql_query("SELECT id, display_name FROM individuals", conn)
        
        if not df_map.empty:
            choice = st.selectbox("Select Individual", df_map['display_name'])
            target_id = df_map[df_map['display_name'] == choice]['id'].values[0]
            
            with engine.connect() as conn:
                curr = conn.execute(text("SELECT * FROM individuals WHERE id=:id"), {"id": target_id}).fetchone()
            
            with st.form("edit_form"):
                new_dn = st.text_input("Display Name", curr[1])
                new_age = st.selectbox("Age Class", ["Chick", "Juvenile", "Adult"], index=2)
                new_notes = st.text_area("Notes", curr[5])
                if st.form_submit_button("Save"):
                    with engine.connect() as conn:
                        conn.execute(text("UPDATE individuals SET display_name=:dn, age=:ag, notes=:nt WHERE id=:id"),
                                     {"dn": new_dn, "ag": new_age, "nt": new_notes, "id": target_id})
                        conn.commit()
                    st.success("Updated.")
        else: st.info("No data to edit.")

elif st.session_state.current_view == 'Logs':
    st.header("📜 Audit Logs")
    if st.session_state.is_admin:
        with engine.connect() as conn:
            df_log = pd.read_sql_query("SELECT * FROM change_log ORDER BY id DESC", conn)
        st.dataframe(df_log, use_container_width=True)
    else:
        st.info("Admin login required.")