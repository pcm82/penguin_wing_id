"""
Penguin Wing Identification App - Full CMS & Audit Edition
Includes Database Migrations for 'individuals' and 'change_log' tables.
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
PORTRAIT_DIR = './data/portraits'
DB_PATH = "./data/database.db"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PORTRAIT_DIR, exist_ok=True)

# --- CACHED AI ENGINE ---
@st.cache_resource
def load_engine():
    """Loads the biometric model and keeps it in memory."""
    biometric_model = timm.create_model(
        'hf-hub:BVRA/MegaDescriptor-T-224',
        pretrained=True,
        num_classes=0
    )
    biometric_model.eval()
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

# --- DATABASE LOGIC & MIGRATION ---
def init_db():
    """Initializes SQLite and Migrates columns if they are missing."""
    db_conn = sqlite3.connect(DB_PATH)
    cursor = db_conn.cursor()
    
    # Core Tables
    cursor.execute('''CREATE TABLE IF NOT EXISTS individuals
                 (id TEXT PRIMARY KEY, display_name TEXT, age TEXT, 
                  mother TEXT, father TEXT, notes TEXT, rep_image TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS encounters
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, penguin_id TEXT, 
                  date TEXT, location TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS pending_changes
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, penguin_id TEXT, 
                  field TEXT, new_value TEXT, timestamp TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS change_log 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, penguin_id TEXT, 
                  event_type TEXT, field TEXT, old_value TEXT, new_value TEXT, 
                  timestamp TEXT)''')
    
    # Migration Check: Individuals Table
    cursor.execute("PRAGMA table_info(individuals)")
    ind_cols = [info[1] for info in cursor.fetchall()]
    if "display_name" not in ind_cols:
        cursor.execute("ALTER TABLE individuals ADD COLUMN display_name TEXT")
    if "rep_image" not in ind_cols:
        cursor.execute("ALTER TABLE individuals ADD COLUMN rep_image TEXT")
        
    # Migration Check: Change Log Table (FIXES YOUR ERROR)
    cursor.execute("PRAGMA table_info(change_log)")
    log_cols = [info[1] for info in cursor.fetchall()]
    if "event_type" not in log_cols:
        cursor.execute("ALTER TABLE change_log ADD COLUMN event_type TEXT")
    
    db_conn.commit()
    db_conn.close()

# --- HELPER: LOGGING ---
def log_event(p_id, e_type, field="N/A", old="N/A", new="N/A"):
    """Helper to record system activity."""
    with sqlite3.connect(DB_PATH) as l_conn:
        l_conn.execute(
            "INSERT INTO change_log (penguin_id, event_type, field, old_value, new_value, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (p_id, e_type, field, str(old), str(new), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )

# --- APP SETUP ---
st.set_page_config(page_title="King Penguin Research CMS", layout="wide")
init_db()

if 'current_view' not in st.session_state:
    st.session_state.current_view = 'Dossier'
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False

# --- ADMIN LOGIN ---
with st.sidebar:
    st.title("🔐 Authentication")
    if not st.session_state.is_admin:
        pwd = st.text_input("Admin Password", type="password")
        if st.button("Login"):
            if pwd == "penguinadmin":
                st.session_state.is_admin = True
                st.rerun()
            else:
                st.error("Invalid Password")
    else:
        st.success("Admin Mode Active")
        if st.button("Logout"):
            st.session_state.is_admin = False
            st.rerun()

# --- NAVIGATION ---
st.title("🐧 King Penguin Research Portal")
n1, n2, n3, n4, n5, n6 = st.columns(6)
views = [("🔍 Identify", "Identify"), ("🗂️ Dossier", "Dossier"), ("🕒 Sightings", "Sightings"), 
         ("📝 Edit", "Edit"), ("⏳ Pending", "Pending"), ("📜 Logs", "Logs")]
for i, (label, view) in enumerate(views):
    if eval(f"n{i+1}").button(label):
        st.session_state.current_view = view

st.divider()
main_model, main_extractor, main_transform = load_engine()
main_dataset = FlatImageDataset(DATA_DIR, transform=main_transform)
db_features = get_db_fingerprints(main_extractor, main_dataset)
conn = sqlite3.connect(DB_PATH)

# --- VIEWS ---

if st.session_state.current_view == 'Identify':
    st.header("New Sighting Identification")
    up_file = st.file_uploader("Upload Image", type=['jpg', 'png'])
    
    with st.expander("➕ Log Sighting Manually (No Image)"):
        df_ids = pd.read_sql_query("SELECT id FROM individuals", conn)
        manual_id = st.selectbox("Select Known Penguin", df_ids['id'].tolist(), key="man_sel")
        m_date = st.date_input("Date", datetime.now(), key="man_date")
        m_loc = st.text_input("Location", "Main Colony", key="man_loc")
        if st.button("Log Manual Sighting"):
            conn.execute("INSERT INTO encounters (penguin_id, date, location) VALUES (?,?,?)", 
                         (manual_id, str(m_date), m_loc))
            log_event(manual_id, "SIGHTING", "Manual", "None", f"Loc: {m_loc}")
            conn.commit()
            st.success(f"Logged manual sighting for {manual_id}!")

    if up_file:
        q_img = Image.open(up_file).convert('RGB')
        t_orig = main_transform(q_img).unsqueeze(0)
        with torch.no_grad():
            f_orig = main_model(t_orig)
        sim = torch.cosine_similarity(f_orig, db_features)
        v, idx = torch.max(sim, dim=0)
        match_id = main_dataset.image_files[idx.item()]
        
        c1, c2 = st.columns(2)
        c1.image(q_img, caption="Query", use_container_width=True)
        c2.image(os.path.join(DATA_DIR, match_id), caption=f"Match: {match_id}", use_container_width=True)
        st.metric("AI Confidence Score", f"{v.item()*100:.2f}%")
        
        if st.checkbox(f"Confirm match with {match_id}?"):
            s_date = st.date_input("Sighting Date", datetime.now())
            s_loc = st.text_input("Sighting Location", "Main Colony")
            if st.button("Save Sighting"):
                conn.execute("INSERT INTO encounters (penguin_id, date, location) VALUES (?,?,?)", 
                             (match_id, str(s_date), s_loc))
                log_event(match_id, "SIGHTING", "Photo", "None", f"Loc: {s_loc}")
                conn.commit()
                st.success("Sighting Logged and Indexed!")

elif st.session_state.current_view == 'Dossier':
    st.header("🗂️ Population Gallery")
    df_pop = pd.read_sql_query("SELECT * FROM individuals", conn)
    if df_pop.empty:
        st.info("No individuals registered.")
        if st.button("🚀 Auto-Register Photos"):
            for fname in main_dataset.image_files:
                conn.execute("INSERT OR IGNORE INTO individuals (id, age, mother, father, notes) VALUES (?,?,?,?,?)", 
                             (fname, "Adult", "Unknown", "Unknown", "Auto-Import"))
                log_event(fname, "SYSTEM", "Register", "None", "Imported")
            conn.commit()
            st.rerun()
    else:
        for r_idx in range(0, len(df_pop), 4):
            cols = st.columns(4)
            for c_idx, g_col in enumerate(cols):
                if r_idx + c_idx < len(df_pop):
                    p = df_pop.iloc[r_idx + c_idx]
                    with g_col:
                        img_p = p['rep_image'] if p['rep_image'] else os.path.join(DATA_DIR, p['id'])
                        st.image(img_p, use_container_width=True)
                        st.write(f"**{p['display_name'] or p['id']}**")
                        st.caption(f"Age: {p['age']}")

elif st.session_state.current_view == 'Sightings':
    st.header("🕒 Sighting Records")
    df_sight = pd.read_sql_query("SELECT * FROM encounters ORDER BY date DESC", conn)
    for _, row in df_sight.iterrows():
        with st.container(border=True):
            sc1, sc2 = st.columns([1, 4])
            img_path = os.path.join(DATA_DIR, row['penguin_id'])
            if os.path.exists(img_path):
                sc1.image(img_path, width=150)
            else: sc1.write("🖼️ (No Image)")
            sc2.write(f"**Penguin ID:** {row['penguin_id']}")
            sc2.write(f"📅 **Date:** {row['date']} | 📍 **Location:** {row['location']}")

elif st.session_state.current_view == 'Edit':
    st.header("📝 Edit or Propose Metadata")
    df_ids = pd.read_sql_query("SELECT id FROM individuals", conn)
    if not df_ids.empty:
        target = st.selectbox("Target Penguin", df_ids['id'].tolist())
        curr = pd.read_sql_query(f"SELECT * FROM individuals WHERE id='{target}'", conn).iloc[0]
        with st.form("edit_form"):
            n_name = st.text_input("Display Name", curr['display_name'] if curr['display_name'] else "")
            n_age = st.selectbox("Age", ["Chick", "Juvenile", "Adult"], index=0)
            n_mom = st.text_input("Mother", curr['mother'] if curr['mother'] else "Unknown")
            n_dad = st.text_input("Father", curr['father'] if curr['father'] else "Unknown")
            n_note = st.text_area("Notes", curr['notes'] if curr['notes'] else "")
            new_pic = st.file_uploader("Upload Portrait", type=['jpg', 'png'])
            
            if st.form_submit_button("Submit"):
                r_path = curr['rep_image']
                if new_pic:
                    r_path = os.path.join(PORTRAIT_DIR, f"{target}_rep.jpg")
                    Image.open(new_pic).save(r_path)
                
                if st.session_state.is_admin:
                    conn.execute("UPDATE individuals SET display_name=?, age=?, mother=?, father=?, notes=?, rep_image=? WHERE id=?", 
                                 (n_name, n_age, n_mom, n_dad, n_note, r_path, target))
                    log_event(target, "DIRECT_EDIT", "Dossier", "Manual", "Updated by Admin")
                    conn.commit()
                    st.success("Admin Update Saved.")
                else:
                    for k, v in [("display_name", n_name), ("age", n_age), ("mother", n_mom), ("notes", n_note)]:
                        if str(v) != str(curr[k]):
                            conn.execute("INSERT INTO pending_changes (penguin_id, field, new_value, timestamp) VALUES (?,?,?,?)", 
                                         (target, k, v, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                            log_event(target, "PROPOSAL", k, curr[k], v)
                    conn.commit()
                    st.info("Proposals submitted for review.")
    else: st.warning("No individuals found.")

elif st.session_state.current_view == 'Pending':
    st.header("⏳ Pending Proposals")
    if not st.session_state.is_admin:
        st.warning("Admin Access Required.")
    else:
        df_p = pd.read_sql_query("SELECT * FROM pending_changes", conn)
        if df_p.empty: st.info("No pending changes.")
        else:
            for _, row in df_p.iterrows():
                with st.container(border=True):
                    st.write(f"**Penguin:** {row['penguin_id']} | **Field:** {row['field']} | **New Value:** `{row['new_value']}`")
                    b1, b2 = st.columns(2)
                    if b1.button(f"✅ Accept ##{row['id']}"):
                        old_q = pd.read_sql_query(f"SELECT {row['field']} FROM individuals WHERE id='{row['penguin_id']}'", conn)
                        old_val = old_q.iloc[0][0] if not old_q.empty else "N/A"
                        conn.execute(f"UPDATE individuals SET {row['field']}=? WHERE id=?", (row['new_value'], row['penguin_id']))
                        log_event(row['penguin_id'], "ACCEPTED", row['field'], old_val, row['new_value'])
                        conn.execute("DELETE FROM pending_changes WHERE id=?", (row['id'],))
                        conn.commit(); st.rerun()
                    if b2.button(f"❌ Reject ##{row['id']}"):
                        log_event(row['penguin_id'], "REJECTED", row['field'], "N/A", row['new_value'])
                        conn.execute("DELETE FROM pending_changes WHERE id=?", (row['id'],))
                        conn.commit(); st.rerun()

elif st.session_state.current_view == 'Logs':
    st.header("📜 System Audit Trail")
    if not st.session_state.is_admin:
        st.warning("Admin Access Required.")
    else:
        df_l = pd.read_sql_query("SELECT * FROM change_log ORDER BY timestamp DESC", conn)
        event_filter = st.multiselect("Filter by Event:", ["SIGHTING", "PROPOSAL", "ACCEPTED", "REJECTED", "DIRECT_EDIT", "SYSTEM"], default=["SIGHTING", "PROPOSAL", "ACCEPTED", "REJECTED", "DIRECT_EDIT"])
        st.dataframe(df_l[df_l['event_type'].isin(event_filter)], use_container_width=True, hide_index=True)

conn.close()