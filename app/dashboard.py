"""
app/dashboard.py — Smart Attendance Dashboard (Advanced UI with Plotly & Registration)
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np, datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(page_title="Smart Attendance System", page_icon="🎓", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#0a0e1a;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1117 0%,#161b27 100%);border-right:1px solid #21262d;}
[data-testid="stSidebar"] *{color:#c9d1d9;}
.header-wrap{background:linear-gradient(135deg,#1f6feb 0%,#8b5cf6 50%,#ec4899 100%);border-radius:20px;padding:32px 40px;margin-bottom:28px;position:relative;overflow:hidden;}
.header-wrap h1{color:#fff;font-size:2.2rem;font-weight:800;margin:0;letter-spacing:-0.5px;}
.header-wrap p{color:rgba(255,255,255,0.75);margin:8px 0 0;font-size:14px;}
.kpi{background:#161b27;border:1px solid #21262d;border-radius:14px;padding:20px 22px;transition:transform .2s;}
.kpi:hover{transform:translateY(-2px);}
.kpi .icon{font-size:1.8rem;margin-bottom:8px;}
.kpi .val{font-size:2rem;font-weight:800;color:#fff;line-height:1;}
.kpi .lbl{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-top:4px;}
.kpi .delta{font-size:12px;margin-top:6px;}
.badge-green{background:#0d4429;color:#3fb950;border-radius:6px;padding:2px 8px;font-size:12px;font-weight:600;}
.badge-red{background:#4c1717;color:#f85149;border-radius:6px;padding:2px 8px;font-size:12px;font-weight:600;}
.badge-blue{background:#0c2d6b;color:#58a6ff;border-radius:6px;padding:2px 8px;font-size:12px;font-weight:600;}
.card{background:#161b27;border:1px solid #21262d;border-radius:14px;padding:24px;margin-bottom:18px;}
.section-title{font-size:1.1rem;font-weight:700;color:#e6edf3;margin-bottom:16px;display:flex;align-items:center;gap:8px;}
.alert-s{background:#0d4429;border:1px solid #238636;color:#3fb950;border-radius:10px;padding:14px 18px;margin:10px 0;}
.alert-w{background:#3a1c00;border:1px solid #9e6a03;color:#e3b341;border-radius:10px;padding:14px 18px;margin:10px 0;}
.alert-i{background:#0c2d6b;border:1px solid #1f6feb;color:#58a6ff;border-radius:10px;padding:14px 18px;margin:10px 0;}
.person-card{background:#1c2333;border:1px solid #30363d;border-radius:12px;padding:16px 20px;text-align:center;transition:all .2s;}
.person-card:hover{border-color:#58a6ff;transform:translateY(-3px);}
.person-card .pname{font-weight:700;color:#e6edf3;font-size:15px;}
.person-card .pcount{font-size:2rem;font-weight:800;color:#58a6ff;}
.person-card .ppct{font-size:12px;color:#8b949e;}
stButton>button{background:linear-gradient(135deg,#1f6feb,#8b5cf6)!important;color:white!important;border:none!important;border-radius:10px!important;font-weight:600!important;}
div[data-testid="stDataFrame"]{border-radius:10px;overflow:hidden;}
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    plot_bgcolor='#161b27', paper_bgcolor='#0a0e1a',
    font=dict(color='#c9d1d9', family='Inter'),
    xaxis=dict(gridcolor='#21262d', linecolor='#30363d'),
    yaxis=dict(gridcolor='#21262d', linecolor='#30363d'),
)

def safe_fetch_all():
    try:
        from database.db import fetch_all
        return fetch_all()
    except Exception as e:
        st.error(f"DB: {e}"); return []

def safe_fetch_by_date(d):
    try:
        from database.db import fetch_by_date
        return fetch_by_date(d)
    except Exception as e:
        st.error(f"DB: {e}"); return []

def safe_fetch_range(start, end):
    try:
        from database.db import fetch_all
        all_data = fetch_all()
        df = pd.DataFrame(all_data, columns=["id","name","date","time"])
        df['date'] = pd.to_datetime(df['date']).dt.date
        mask = (df['date'] >= start) & (df['date'] <= end)
        return df.loc[mask]
    except Exception as e:
        st.error(f"DB: {e}"); return pd.DataFrame()

def safe_insert(name, date, time):
    try:
        from database.db import insert_attendance
        insert_attendance(name, date, time); return True
    except Exception as e:
        st.error(f"Save: {e}"); return False

if "last_seen" not in st.session_state:
    st.session_state.last_seen = {}

def is_new_session(name):
    now = datetime.datetime.now()
    last = st.session_state.last_seen.get(name)
    if last and (now - last).seconds < 60: return False
    st.session_state.last_seen[name] = now; return True

@st.cache_resource(show_spinner="⚙️ Loading AI models…")
def load_model():
    from utils.face_loader import load_faces
    from utils.recognition import detect_faces, recognize
    return load_faces, detect_faces, recognize

@st.cache_resource(show_spinner="📂 Loading embeddings…")
def get_embeddings():
    lf,_,_ = load_model(); return lf()

# ── SIDEBAR ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 🎓 Attendance System")
    st.markdown("<p style='color:#8b949e;font-size:12px;margin-top:-8px;'>Internship Project</p>", unsafe_allow_html=True)
    st.markdown("---")
    option = st.selectbox("Navigation", [
        "📅 Daily Attendance", "📈 Monthly Analytics",
        "👤 Person-wise Count", "👥 Register New Face",
        "🚨 Alerts", "📷 Live Camera"
    ], label_visibility="collapsed")
    # Accuracy settings for the presentation
    st.markdown("⚙️ **System Sensitivity**")
    rec_threshold = st.sidebar.slider(
        "Match Strictness", 
        0.1, 1.5, 1.0, 0.1, 
        help="Adjusts AI precision: Lower (0.6) is very strict for high security. Higher (1.2) is more relaxed. 1.0 is the recommended balance."
    )
    st.markdown("---")
    st.markdown(f"<div style='color:#8b949e;font-size:11px;'>🕐 {datetime.datetime.now().strftime('%d %b %Y, %I:%M %p')}</div>", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
  <h1>📊 Smart Attendance Dashboard</h1>
  <p>🤖 FaceNet AI &nbsp;•&nbsp; 🗄️ SQLite &nbsp;•&nbsp; 📱 Twilio SMS &nbsp;•&nbsp; 🎥 Live Camera &nbsp;•&nbsp; 📊 Plotly Analytics</p>
</div>
""", unsafe_allow_html=True)

# ── Top KPI row ───────────────────────────────────────────────────────
all_rows = safe_fetch_all()
df_all = pd.DataFrame(all_rows, columns=["id","name","date","time"]) if all_rows else pd.DataFrame(columns=["id","name","date","time"])
total = len(all_rows)
today_str = str(datetime.date.today())
today_c = len([r for r in all_rows if str(r[2]) == today_str])
unique_p = len(set(r[1] for r in all_rows)) if all_rows else 0
total_days = df_all["date"].nunique() if len(df_all) else 0
top_person = df_all["name"].value_counts().idxmax() if len(df_all) else "—"

c1,c2,c3,c4 = st.columns(4)
for col, icon, val, lbl, badge_cls, badge_txt in [
    (c1,"📋",total,"Total Records","badge-blue","all time"),
    (c2,"✅",today_c,"Present Today","badge-green","today"),
    (c3,"👥",unique_p,"Registered People","badge-blue","enrolled"),
    (c4,"🏆",top_person,"Top Attender","badge-green","leader"),
]:
    col.markdown(f"""
    <div class="kpi">
      <div class="icon">{icon}</div>
      <div class="val">{val}</div>
      <div class="lbl">{lbl}</div>
      <div class="delta"><span class="{badge_cls}">{badge_txt}</span></div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── CONTENT ──────────────────────────────────────────────────────────
if "Daily" in option:
    st.markdown('<div class="section-title">📅 Attendance Records & Filtering</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Single Date", "Date Range"])
    with tab1:
        col1, col2 = st.columns([1,3])
        with col1: sel_date = st.date_input("Select Date", value=datetime.date.today())
        rows = safe_fetch_by_date(str(sel_date))
        df = pd.DataFrame(rows, columns=["ID","Name","Date","Time"]) if rows else pd.DataFrame(columns=["ID","Name","Date","Time"])
        if len(df):
            st.dataframe(df, use_container_width=True)
            st.markdown(f'<div class="alert-s">✅ <b>Total Present: {len(df)}</b> on {sel_date}</div>', unsafe_allow_html=True)
            csv = df.to_csv(index=False).encode(); st.download_button("📥 Export CSV", csv, f"attendance_{sel_date}.csv", "text/csv")
        else: st.markdown(f'<div class="alert-i">ℹ️ No records for <b>{sel_date}</b></div>', unsafe_allow_html=True)
    with tab2:
        c1, c2 = st.columns(2)
        start_d = c1.date_input("Start Date", value=datetime.date.today() - datetime.timedelta(days=7))
        end_d = c2.date_input("End Date", value=datetime.date.today())
        df_range = safe_fetch_range(start_d, end_d)
        if not df_range.empty:
            st.dataframe(df_range, use_container_width=True)
            st.markdown(f'<div class="alert-s">📅 Found <b>{len(df_range)}</b> records from {start_d} to {end_d}</div>', unsafe_allow_html=True)
        else: st.markdown('<div class="alert-i">ℹ️ No records found in this range.</div>', unsafe_allow_html=True)

elif "Monthly" in option:
    st.markdown('<div class="section-title">📈 Monthly Analytics</div>', unsafe_allow_html=True)
    if len(df_all):
        df_all["month"] = pd.to_datetime(df_all["date"], errors="coerce").dt.to_period("M").astype(str)
        monthly = df_all.groupby("month").size().reset_index(name="Count")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["Count"], mode='lines+markers', line=dict(color='#1f6feb', width=3), marker=dict(size=8, color='#8b5cf6', line=dict(color='white', width=2)), fill='tozeroy', fillcolor='rgba(31,111,235,0.1)', name='Attendance'))
        fig.update_layout(title="Monthly Attendance Trend", **PLOTLY_THEME, height=380)
        st.plotly_chart(fig, use_container_width=True)
        df_all["day"] = pd.to_datetime(df_all["date"], errors="coerce").dt.day_name()
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        day_counts = df_all["day"].value_counts().reindex(day_order, fill_value=0).reset_index(); day_counts.columns = ["Day","Count"]
        fig2 = px.bar(day_counts, x="Day", y="Count", color="Count", color_continuous_scale=["#1f6feb","#8b5cf6","#ec4899"], title="Attendance by Day of Week")
        fig2.update_layout(**PLOTLY_THEME, height=320, coloraxis_showscale=False); st.plotly_chart(fig2, use_container_width=True)
    else: st.markdown('<div class="alert-i">ℹ️ No data yet.</div>', unsafe_allow_html=True)

elif "Person" in option:
    st.markdown('<div class="section-title">👤 Person-wise Attendance</div>', unsafe_allow_html=True)
    if len(df_all):
        counts = df_all["name"].value_counts().reset_index(); counts.columns = ["Name","Count"]
        max_days = total_days if total_days > 0 else 1
        counts["Percentage"] = (counts["Count"] / max_days * 100).clip(upper=100).round(1)
        cols = st.columns(min(len(counts), 4))
        for i, row in counts.iterrows():
            pct = row["Percentage"]; badge = "badge-green" if pct >= 75 else "badge-red"
            cols[i % len(cols)].markdown(f'<div class="person-card"><div class="pcount">{int(row["Count"])}</div><div class="pname">👤 {row["Name"]}</div><div class="ppct">days present</div><div style="margin-top:8px;"><span class="{badge}">{pct}%</span></div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        colors = ['#3fb950' if p >= 75 else '#f85149' for p in counts["Percentage"]]
        fig = go.Figure(go.Bar(x=counts["Name"], y=counts["Count"], marker_color=colors, text=counts["Count"], textposition='outside', textfont=dict(color='white')))
        fig.update_layout(title="Attendance Count per Person", **PLOTLY_THEME, height=380, showlegend=False); st.plotly_chart(fig, use_container_width=True)
        fig2 = px.pie(counts, values="Count", names="Name", title="Attendance Distribution", color_discrete_sequence=px.colors.sequential.Plasma_r)
        fig2.update_layout(**PLOTLY_THEME, height=350); st.plotly_chart(fig2, use_container_width=True)
    else: st.markdown('<div class="alert-i">ℹ️ No data yet.</div>', unsafe_allow_html=True)

elif "Register" in option:
    st.markdown('<div class="section-title">👤 Register New Face</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**1. Enter Personal Details**")
        new_name = st.text_input("Person Name", placeholder="e.g. John Doe", key="reg_name").strip().lower()
        st.markdown("**2. Guidelines**")
        st.info("• Face the camera clearly\n• Good lighting required\n• Multiple photos help accuracy")
    with col2:
        if new_name:
            st.markdown(f"**3. Capture Multiple Photos for {new_name}**")
            reg_method = st.radio("Choose Method", ["Live Camera", "Upload Photo"], horizontal=True)
            reg_photo = None
            if reg_method == "Live Camera": reg_photo = st.camera_input("Capture Shot", key="reg_cam")
            else: reg_photo = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
            if reg_photo:
                save_dir = f"data/known_faces/{new_name}"
                os.makedirs(save_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(save_dir, f"{timestamp}.jpg")
                img_bytes = reg_photo.getvalue() if hasattr(reg_photo, 'getvalue') else reg_photo.getbuffer()
                with open(save_path, "wb") as f: f.write(img_bytes)
                st.toast(f"✅ Photo {len(os.listdir(save_dir))} saved!")
                st.markdown(f'<div class="alert-s">📸 Photo saved! You can take <b>more photos</b> from different angles.</div>', unsafe_allow_html=True)
                num_photos = len(os.listdir(save_dir)); st.metric("Photos Captured", num_photos)
                if st.button("🚀 Finalize & Train AI", type="primary", use_container_width=True):
                    st.cache_resource.clear(); st.success(f"Registered {new_name} with {num_photos} photos! Returning..."); st.rerun()
        else: st.warning("Please enter a name first to start registration.")
    st.markdown("---")
    st.markdown('<div class="section-title">📂 Manage Registered Faces</div>', unsafe_allow_html=True)
    known_path = "data/known_faces"
    if os.path.exists(known_path):
        students = sorted([d for d in os.listdir(known_path) if os.path.isdir(os.path.join(known_path, d))])
        if students:
            col_sel, col_btn = st.columns([3, 1])
            to_delete = col_sel.selectbox("Select Student to Remove", students)
            if col_btn.button("🗑️ Delete Person", type="primary", use_container_width=True):
                import shutil; shutil.rmtree(os.path.join(known_path, to_delete))
                st.cache_resource.clear(); st.success(f"Removed {to_delete}! Reloading..."); st.rerun()
        else: st.info("No registered students found.")

elif "Alert" in option:
    st.markdown('<div class="section-title">🚨 Alerts System</div>', unsafe_allow_html=True)
    if len(df_all):
        counts = df_all["name"].value_counts().reset_index(); counts.columns = ["name","count"]
        max_c = int(counts["count"].max()); min_days = st.slider("🎯 Minimum Attendance Threshold", 1, max(max_c,30), 5)
        low = counts[counts["count"] < min_days]; ok = counts[counts["count"] >= min_days]
        ca, cb = st.columns(2); ca.metric("⚠️ Below Threshold", len(low), delta=f"-{len(low)}", delta_color="inverse"); cb.metric("✅ Meeting Target", len(ok), delta=f"+{len(ok)}")
        if not low.empty:
            fig = go.Figure()
            for _, row in low.iterrows(): fig.add_trace(go.Bar(name=row["name"], x=[row["name"]], y=[row["count"]], marker_color='#f85149', text=[int(row["count"])], textposition='outside'))
            fig.add_hline(y=min_days, line_dash="dash", line_color="#e3b341", annotation_text=f"Threshold: {min_days}")
            fig.update_layout(title="Low Attendance Students", **PLOTLY_THEME, height=320, showlegend=False); st.plotly_chart(fig, use_container_width=True)
            st.markdown("**⚠️ Low Attendance List:**"); st.dataframe(low.reset_index(drop=True), use_container_width=True)
        try: from config import TWILIO_TO_NUMBER as dn
        except: dn = "+91"
        phone = st.text_input("📱 Recipient Phone Number", value=dn)
        if st.button("📤 Send SMS Alerts", type="primary", use_container_width=True):
            if low.empty: st.info("No low-attendance students.")
            else:
                from utils.notify import send_bulk_sms
                results = send_bulk_sms(low["name"].tolist(), phone.strip())
                for name, sid in results:
                    if sid: st.markdown(f'<div class="alert-s">✅ SMS sent for <b>{name}</b> — <code>{sid}</code></div>', unsafe_allow_html=True)
                    else: st.markdown(f'<div class="alert-w">❌ Failed for {name}</div>', unsafe_allow_html=True)
    else: st.markdown('<div class="alert-i">ℹ️ No data yet.</div>', unsafe_allow_html=True)

elif "Camera" in option:
    st.markdown('<div class="section-title">📷 Live AI Camera</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    lf_fn, det_fn, rec_fn = load_model()
    known_emb, labels = get_embeddings()
    if len(known_emb) == 0: st.markdown('<div class="alert-w">⚠️ No known faces. Add images to <code>data/known_faces/</code></div>', unsafe_allow_html=True)
    img_file = st.camera_input("Camera", label_visibility="collapsed")
    if img_file:
        pil = Image.open(img_file).convert("RGB"); rgb = np.array(pil); bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = det_fn(bgr); marked = []; face_info = []
        for (x1,y1,x2,y2) in faces:
            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0: continue
            name = rec_fn(crop, known_emb, labels, distance_threshold=rec_threshold) if len(known_emb) > 0 else "Unknown"
            face_info.append(name); color = (0,255,100) if name != "Unknown" else (255,60,60)
            cv2.rectangle(bgr,(x1,y1),(x2,y2),color,3); cv2.putText(bgr, name,(x1,max(y1-12,0)),cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2)
            if name != "Unknown" and is_new_session(name):
                now = datetime.datetime.now()
                if safe_insert(name, str(now.date()), now.strftime("%H:%M:%S")):
                    marked.append((name, now.strftime("%H:%M:%S")))
                    try:
                        from utils.notify import send_sms
                        send_sms(name, now.strftime("%H:%M:%S"))
                    except: pass
        result = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); st.image(result, use_container_width=True)
        if marked:
            for name, t in marked: st.markdown(f'<div class="alert-s">✅ <b>Attendance marked for {name}</b> at {t} </div>', unsafe_allow_html=True)
        elif faces:
            for n in face_info:
                if n == "Unknown": st.markdown('<div class="alert-w">⚠️ Face detected but not recognized. Please register first.</div>', unsafe_allow_html=True)
                else: st.markdown(f'<div class="alert-i">ℹ️ <b>{n}</b> already marked (cooldown active).</div>', unsafe_allow_html=True)
        else: st.markdown('<div class="alert-w">⚠️ No face detected. Please face the camera clearly.</div>', unsafe_allow_html=True)