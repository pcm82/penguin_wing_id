"""
Penguin Wing Identification App - Supabase PostgreSQL Edition
"""

import os
import streamlit as st
import pandas as pd
import torch
import timm
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from wildlife_tools.features import DeepFeatures
from sqlalchemy import create_engine, text
from datetime import datetime

# --- CONFIGURATION ---
DB_URI = "postgresql://postgres:penguindatabase2026!@db.xozmbgbkbdzugsagwghf.supabase.co:5432/postgres"
DATA_DIR = './data/wings'
PORTRAIT_DIR = './data/portraits'

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PORTRAIT_DIR, exist_ok=True)

# --- DATABASE ENGINE ---
engine = create_engine(DB_URI)

def init_db():
    """Initializes the Supabase PostgreSQL tables."""
    with engine.connect() as conn:
        conn.execute(text('''CREATE TABLE IF NOT EXISTS individuals
                     (id TEXT PRIMARY KEY, display_name TEXT, age TEXT, 
                      mother TEXT, father TEXT, notes TEXT, rep_image TEXT)'''))
        conn.execute(text('''CREATE TABLE IF NOT EXISTS encounters
                     (id SERIAL PRIMARY KEY, penguin_id TEXT, date TEXT, location TEXT)'''))
        conn.execute(text('''CREATE TABLE IF NOT EXISTS pending_changes
                     (id SERIAL PRIMARY KEY, penguin_id TEXT, field TEXT, new_value TEXT, timestamp TEXT)'''))
        conn.execute(text('''CREATE TABLE IF NOT EXISTS change_log 
                     (id SERIAL PRIMARY KEY, penguin_id TEXT, event_type TEXT, field TEXT, 
                      old_value TEXT, new_value TEXT, timestamp TEXT)'''))
        conn.commit()

def log_event(p_id, e_type, field="N/A", old="N/A", new="N/A"):
    """Records system activity in the cloud audit log."""
    with engine.connect() as conn:
        conn.execute(text(
            "INSERT INTO change_log (penguin_id, event_type, field, old_value, new_value, timestamp) "
            "VALUES (:p, :e, :f, :o, :n, :t)"),
            {"p": p_id, "e": e_type, "f": field, "o": str(old), "n": str(new), 
             "t": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        )
        conn.commit()

# --- CACHED AI ENGINE ---
@st.cache_resource
def load_engine():
    biometric_model = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', pretrained=True, num_classes=0)
    biometric_model.eval()
    model_extractor = DeepFeatures(biometric_model, num_workers=0)
    model_transform = T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return biometric_model, model_extractor, model_transform

class FlatImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg'))]
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.image_files[idx])
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.image_files[idx]

@st.cache_data
def get_db_fingerprints(_extractor, _dataset):
    features = _extractor(_dataset)
    return torch.from_numpy(features) if not isinstance(features, torch.Tensor) else features

# --- APP SETUP ---
st.set_page_config(page_title="King Penguin CMS", layout="wide")
init_db()

if 'current_view' not in st.session_state: st.session_state.current_view = 'Dossier'
if 'is_admin' not in st.session_state: st.session_state.is_admin = False

with st.sidebar:
    st.title("🔐 Admin Login")
    if not st.session_state.is_admin:
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if pwd == "penguinadmin":
                st.session_state.is_admin = True
                st.rerun()
            else: st.error("Invalid")
    else:
        if st.button("Logout"):
            st.session_state.is_admin = False
            st.rerun()

# --- NAVIGATION ---
st.title("🐧 King Penguin Research Portal")
n1, n2, n3, n4, n5, n6 = st.columns(6)
views = [("🔍 Identify", "Identify"), ("🗂️ Dossier", "Dossier"), ("🕒 Sightings", "Sightings"), 
         ("📝 Edit", "Edit"), ("⏳ Pending", "Pending"), ("📜 Logs", "Logs")]
for i, (label, view) in enumerate(views):
    if eval(f"n{i+1}").button(label): st.session_state.current_view = view

st.divider()
model_obj, extractor_obj, transform_obj = load_engine()
main_dataset = FlatImageDataset(DATA_DIR, transform=transform_obj)
db_features = get_db_fingerprints(extractor_obj, main_dataset)

# --- VIEWS ---
if st.session_state.current_view == 'Identify':
    st.header("New Sighting Identification")
    up_file = st.file_uploader("Upload Image", type=['jpg', 'png'])
    
    with st.expander("➕ Log Sighting Manually"):
        with engine.connect() as conn:
            df_map = pd.read_sql_query("SELECT id, display_name FROM individuals", conn)
        if not df_map.empty:
            name_to_id = dict(zip(df_map['display_name'], df_map['id']))
            manual_name = st.selectbox("Select Penguin", list(name_to_id.keys()))
            manual_id = name_to_id[manual_name]
            if st.button("Log Manual Sighting"):
                with engine.connect() as conn:
                    conn.execute(text("INSERT INTO encounters (penguin_id, date, location) VALUES (:p, :d, :l)"),
                                 {"p": manual_id, "d": str(datetime.now().date()), "l": "Manual Entry"})
                    conn.commit()
                log_event(manual_id, "SIGHTING", "Manual", "None", "Logged")
                st.success(f"Logged {manual_name}")

    if up_file:
        q_img = Image.open(up_file).convert('RGB')
        t_q = transform_obj(q_img).unsqueeze(0)
        with torch.no_grad(): f_q = model_obj(t_q)
        sim = torch.cosine_similarity(f_q, db_features)
        v, idx = torch.max(sim, dim=0)
        match_id = main_dataset.image_files[idx.item()]
        
        with engine.connect() as conn:
            res = conn.execute(text("SELECT display_name FROM individuals WHERE id=:id"), {"id": match_id}).fetchone()
        match_name = res[0] if res else match_id

        c1, c2 = st.columns(2)
        c1.image(q_img, caption="Query", width='stretch')
        c2.image(os.path.join(DATA_DIR, match_id), caption=f"Match: {match_name}", width='stretch')
        st.metric("Confidence", f"{v.item()*100:.2f}%")

elif st.session_state.current_view == 'Dossier':
    st.header("🗂️ Population Gallery")
    with engine.connect() as conn:
        df_pop = pd.read_sql_query("SELECT * FROM individuals", conn)
    
    if df_pop.empty:
        if st.button("🚀 Auto-Register Photos"):
            with engine.connect() as conn:
                for fname in main_dataset.image_files:
                    conn.execute(text("INSERT INTO individuals (id, display_name, age, mother, father, notes) "
                                      "VALUES (:id, :dn, 'Adult', 'Unknown', 'Unknown', 'Import') ON CONFLICT (id) DO NOTHING"),
                                 {"id": fname, "dn": fname})
                conn.commit()
            st.rerun()
    else:
        for r_idx in range(0, len(df_pop), 4):
            cols = st.columns(4)
            for c_idx, g_col in enumerate(cols):
                if r_idx + c_idx < len(df_pop):
                    p = df_pop.iloc[r_idx + c_idx]
                    with g_col:
                        img_path = p['rep_image'] if p['rep_image'] else os.path.join(DATA_DIR, p['id'])
                        st.image(img_path, width='stretch')
                        st.write(f"**{p['display_name']}**")

elif st.session_state.current_view == 'Sightings':
    st.header("🕒 Sightings")
    with engine.connect() as conn:
        df_sight = pd.read_sql_query("SELECT * FROM encounters ORDER BY id DESC", conn)
    st.dataframe(df_sight, width='stretch')

elif st.session_state.current_view == 'Edit':
    st.header("📝 Edit Metadata")
    with engine.connect() as conn:
        df_map = pd.read_sql_query("SELECT id, display_name FROM individuals", conn)
    if not df_map.empty:
        name_to_id = dict(zip(df_map['display_name'], df_map['id']))
        target_name = st.selectbox("Target Penguin", list(name_to_id.keys()))
        target_id = name_to_id[target_name]
        
        with engine.connect() as conn:
            curr = conn.execute(text("SELECT * FROM individuals WHERE id=:id"), {"id": target_id}).fetchone()
        
        with st.form("edit_meta"):
            n_name = st.text_input("New Display Name", curr[1])
            n_age = st.selectbox("Age", ["Chick", "Juvenile", "Adult"], index=0)
            n_mom = st.text_input("Mother", curr[3])
            if st.form_submit_button("Submit"):
                if st.session_state.is_admin:
                    with engine.connect() as conn:
                        conn.execute(text("UPDATE individuals SET display_name=:dn, age=:ag, mother=:mo WHERE id=:id"),
                                     {"dn": n_name, "ag": n_age, "mo": n_mom, "id": target_id})
                        conn.commit()
                    log_event(target_id, "DIRECT_EDIT", "Name", curr[1], n_name)
                    st.success("Admin Updated")
                else:
                    with engine.connect() as conn:
                        conn.execute(text("INSERT INTO pending_changes (penguin_id, field, new_value, timestamp) VALUES (:id, 'display_name', :nv, :t)"),
                                     {"id": target_id, "nv": n_name, "t": datetime.now().strftime("%Y-%m-%d")})
                        conn.commit()
                    log_event(target_id, "PROPOSAL", "Name", curr[1], n_name)
                    st.info("Proposed")

elif st.session_state.current_view == 'Pending':
    st.header("⏳ Pending Proposals")
    if not st.session_state.is_admin:
        st.warning("Admin Only")
    else:
        with engine.connect() as conn:
            df_p = pd.read_sql_query("SELECT * FROM pending_changes", conn)
        for _, row in df_p.iterrows():
            with st.container(border=True):
                st.write(f"Penguin: {row['penguin_id']} | New {row['field']}: {row['new_value']}")
                if st.button(f"Accept ##{row['id']}"):
                    with engine.connect() as conn:
                        conn.execute(text(f"UPDATE individuals SET {row['field']}=:nv WHERE id=:id"), 
                                     {"nv": row['new_value'], "id": row['penguin_id']})
                        conn.execute(text("DELETE FROM pending_changes WHERE id=:id"), {"id": row['id']})
                        conn.commit()
                    log_event(row['penguin_id'], "ACCEPTED", row['field'], "Pending", row['new_value'])
                    st.rerun()

elif st.session_state.current_view == 'Logs':
    st.header("📜 Audit Logs")
    if not st.session_state.is_admin:
        st.warning("Admin Only")
    else:
        with engine.connect() as conn:
            df_l = pd.read_sql_query("SELECT * FROM change_log ORDER BY id DESC", conn)
        st.dataframe(df_l, width='stretch')