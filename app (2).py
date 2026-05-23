import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Citi Bike Dashboard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color Palette ──────────────────────────────────────────────────────────────
COLOR_BLUE   = '#4C72B0'
COLOR_ORANGE = '#DD8452'
COLOR_GREEN  = '#55A868'
COLOR_RED    = '#C44E52'
COLOR_PURPLE = '#8172B2'
PALETTE      = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_RED, COLOR_PURPLE, '#937860']

# ── Constants ──────────────────────────────────────────────────────────────────
TRIP_MIN       = 60
TRIP_MAX       = 86400
REFERENCE_YEAR = 2018
BY_MIN         = 1928
BY_MAX         = 2002
TOLERANCE      = 60

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #F0F4FA;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes, filename):
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))

    df['starttime'] = pd.to_datetime(df['starttime'], errors='coerce')
    df['stoptime']  = pd.to_datetime(df['stoptime'],  errors='coerce')
    df = df.dropna(subset=['starttime', 'stoptime'])
    df = df[df['stoptime'] > df['starttime']]
    df = df[(df['tripduration'] >= TRIP_MIN) & (df['tripduration'] <= TRIP_MAX)].copy()
    df = df[(df['birth_year'] >= BY_MIN) & (df['birth_year'] <= BY_MAX)].copy()

    for col in ['usertype', 'gender', 'start_station_name', 'end_station_name']:
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower()

    # Fix tripduration inconsistency
    df['_dur_check'] = (df['stoptime'] - df['starttime']).dt.total_seconds().round(0).astype(int)
    df['_dur_diff']  = (df['tripduration'] - df['_dur_check']).abs()
    df.loc[df['_dur_diff'] > TOLERANCE, 'tripduration'] = df.loc[df['_dur_diff'] > TOLERANCE, '_dur_check']
    df.drop(columns=['_dur_check', '_dur_diff'], inplace=True)

    # Feature Engineering
    df['start_month']        = df['starttime'].dt.month
    df['start_month_name']   = df['starttime'].dt.strftime('%B')
    df['start_weekday']      = df['starttime'].dt.dayofweek
    df['start_weekday_name'] = df['starttime'].dt.strftime('%A')
    df['start_hour']         = df['starttime'].dt.hour
    df['is_weekend']         = df['start_weekday'].isin([5, 6])
    df['tripduration_min']   = (df['tripduration'] / 60).round(2)

    def cat_hour(h):
        if h < 6:   return 'Dini Hari'
        elif h < 12: return 'Pagi'
        elif h < 17: return 'Siang'
        elif h < 21: return 'Sore'
        else:        return 'Malam'

    def cat_dur(m):
        if m < 5:   return 'Sangat Pendek (<5m)'
        elif m < 15: return 'Pendek (5-15m)'
        elif m < 30: return 'Sedang (15-30m)'
        elif m < 60: return 'Panjang (30-60m)'
        else:        return 'Sangat Panjang (>60m)'

    def cat_age(a):
        if a < 25:  return 'Remaja/Dewasa Muda (<25)'
        elif a < 35: return 'Dewasa 25-34'
        elif a < 45: return 'Dewasa 35-44'
        elif a < 55: return 'Dewasa 45-54'
        elif a < 65: return 'Menengah 55-64'
        else:        return 'Lansia 65+'

    df['time_of_day']      = df['start_hour'].apply(cat_hour)
    df['duration_category'] = df['tripduration_min'].apply(cat_dur)
    df['age']              = REFERENCE_YEAR - df['birth_year']
    df['age_group']        = df['age'].apply(cat_age)
    df['is_round_trip']    = (df['start_station_name'] == df['end_station_name'])

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## 🚲 NYC Citi Bike")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file is None:
    st.title("🚲 NYC Citi Bike Trips — Dashboard")
    st.info(
        "**Selamat datang!**\n\n"
        "Upload file CSV dataset **NYC Citi Bike** di sidebar kiri untuk memulai analisis."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset", "NYC Citi Bike Trips")
    c2.metric("Tahun Referensi", "2018")
    c3.metric("Halaman Analisis", "6 halaman")
    st.stop()

# Load data (cache by filename+size to avoid re-processing)
file_bytes = uploaded_file.read()
with st.spinner("⏳ Memuat dan membersihkan data..."):
    df = load_and_clean(file_bytes, uploaded_file.name)

# ── Sidebar Filters ────────────────────────────────────────────────────────────
st.sidebar.markdown("### 🔽 Filter")

usertypes   = sorted(df['usertype'].dropna().unique())
sel_user    = st.sidebar.multiselect("Tipe Pengguna", usertypes, default=list(usertypes))

months      = sorted(df['start_month'].unique())
month_label = df[['start_month','start_month_name']].drop_duplicates().sort_values('start_month')
month_map   = dict(zip(month_label['start_month'], month_label['start_month_name']))
sel_months  = st.sidebar.multiselect("Bulan", months, default=list(months),
                                      format_func=lambda x: month_map.get(x, x))

genders   = sorted(df['gender'].dropna().unique())
sel_gender = st.sidebar.multiselect("Gender", genders, default=list(genders))

st.sidebar.markdown("---")
page = st.sidebar.radio("📌 Navigasi", [
    "📊 Overview",
    "⏱️ Pola Waktu",
    "👥 Pengguna",
    "📍 Stasiun",
    "📈 Tren Musiman",
    "🤖 ML Clustering",
])

# Apply filter
dff = df[
    df['usertype'].isin(sel_user) &
    df['start_month'].isin(sel_months) &
    df['gender'].isin(sel_gender)
].copy()

if dff.empty:
    st.warning("⚠️ Tidak ada data yang cocok dengan filter. Longgarkan filter di sidebar.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Overview Dataset")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Trip",      f"{len(dff):,}")
    c2.metric("Median Durasi",   f"{dff['tripduration_min'].median():.1f} mnt")
    c3.metric("Subscriber",      f"{(dff['usertype']=='subscriber').sum():,}")
    c4.metric("Customer",        f"{(dff['usertype']=='customer').sum():,}")
    c5.metric("Round Trip",      f"{dff['is_round_trip'].sum():,}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Proporsi Tipe Pengguna")
        ut = dff['usertype'].value_counts().reset_index()
        ut.columns = ['usertype', 'count']
        fig = px.pie(ut, names='usertype', values='count',
                     color_discrete_sequence=[COLOR_BLUE, COLOR_ORANGE],
                     hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Distribusi Gender")
        g = dff['gender'].value_counts().reset_index()
        g.columns = ['gender', 'count']
        fig = px.bar(g, x='gender', y='count', color='gender',
                     color_discrete_sequence=PALETTE, text='count')
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(showlegend=False, xaxis_title='Gender', yaxis_title='Jumlah Trip')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Distribusi Kategori Durasi")
    dur_order = ['Sangat Pendek (<5m)', 'Pendek (5-15m)', 'Sedang (15-30m)',
                 'Panjang (30-60m)', 'Sangat Panjang (>60m)']
    dur_c = dff['duration_category'].value_counts().reindex(dur_order).dropna().reset_index()
    dur_c.columns = ['kategori', 'count']
    fig = px.bar(dur_c, x='kategori', y='count', color='kategori',
                 color_discrete_sequence=PALETTE, text='count')
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig.update_layout(showlegend=False, xaxis_title='', yaxis_title='Jumlah Trip')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Statistik Deskriptif")
    st.dataframe(dff[['tripduration_min','age','start_hour']].describe().round(2),
                 use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — POLA WAKTU
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⏱️ Pola Waktu":
    st.title("⏱️ Pola Waktu Perjalanan")

    st.subheader("Rush-Hour: Jumlah Trip per Jam — Subscriber vs Customer")
    fig = go.Figure()
    for ut, color in zip(['subscriber', 'customer'], [COLOR_BLUE, COLOR_ORANGE]):
        sub = dff[dff['usertype'] == ut].groupby('start_hour').size().reset_index(name='n_trip')
        if sub.empty: continue
        fig.add_trace(go.Scatter(
            x=sub['start_hour'], y=sub['n_trip'], name=ut.capitalize(),
            mode='lines+markers', line=dict(color=color, width=2.5),
            fill='tozeroy', fillcolor=color.replace(')', ',0.1)').replace('rgb','rgba') if 'rgb' in color else color + '20',
        ))
    fig.add_vrect(x0=7, x1=9,  fillcolor='grey',  opacity=0.1, annotation_text='Peak Pagi')
    fig.add_vrect(x0=17, x1=19, fillcolor='green', opacity=0.1, annotation_text='Peak Sore')
    fig.update_layout(xaxis_title='Jam', yaxis_title='Jumlah Trip',
                      xaxis=dict(tickmode='linear', tick0=0, dtick=1), height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.info("**Interpretasi:** Subscriber mendominasi jam peak pagi (07-09) dan sore (17-19) — pola komuter. Customer terdistribusi merata sepanjang hari — pola rekreasi/wisata.")

    st.markdown("---")
    st.subheader("Heatmap Volume Trip: Hari × Jam")
    tab_wd, tab_we = st.tabs(["📅 Weekday", "🌅 Weekend"])

    for tab, days, label in zip(
        [tab_wd, tab_we],
        [['Monday','Tuesday','Wednesday','Thursday','Friday'], ['Saturday','Sunday']],
        ['Weekday', 'Weekend']
    ):
        with tab:
            sub = dff[dff['start_weekday_name'].isin(days)]
            if sub.empty:
                st.warning("Tidak ada data.")
                continue
            hm = (sub.groupby(['start_weekday_name','start_hour']).size()
                    .unstack(fill_value=0)
                    .reindex([d for d in days if d in sub['start_weekday_name'].unique()]))
            fig = px.imshow(hm, color_continuous_scale='Blues',
                            labels=dict(x='Jam', y='Hari', color='Jumlah Trip'),
                            title=f'Heatmap Volume — {label}', aspect='auto')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    st.info("**Interpretasi:** Weekday memiliki dua titik panas di jam 08:00 dan 17-18:00. Weekend lebih merata (10:00-18:00), mencerminkan aktivitas santai.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PENGGUNA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Pengguna":
    st.title("👥 Analisis Pengguna")

    st.subheader("Distribusi Durasi: Subscriber vs Customer")
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Subscriber', 'Customer'])
    for i, (ut, color) in enumerate(zip(['subscriber','customer'], [COLOR_BLUE, COLOR_ORANGE]), 1):
        d = dff[dff['usertype'] == ut]['tripduration_min']
        if d.empty: continue
        counts, bins = np.histogram(d.clip(0, 60), bins=40)
        fig.add_trace(go.Bar(x=bins[:-1], y=counts, name=ut.capitalize(),
                             marker_color=color, opacity=0.85, showlegend=False), row=1, col=i)
        fig.add_vline(x=d.median(), line_color=COLOR_RED, line_dash='dash',
                      annotation_text=f'Median={d.median():.1f}m', row=1, col=i)
        fig.add_vline(x=d.mean(),   line_color=COLOR_GREEN, line_dash='dot',
                      annotation_text=f'Mean={d.mean():.1f}m', row=1, col=i)
    fig.update_layout(height=400, xaxis_title='Durasi (menit)', xaxis2_title='Durasi (menit)')
    st.plotly_chart(fig, use_container_width=True)

    dur_sub = dff[dff['usertype']=='subscriber']['tripduration_min']
    dur_cus = dff[dff['usertype']=='customer']['tripduration_min']
    if len(dur_sub) > 1 and len(dur_cus) > 1:
        _, p_mw = stats.mannwhitneyu(dur_sub, dur_cus, alternative='two-sided')
        c1, c2, c3 = st.columns(3)
        c1.metric("Subscriber – Median", f"{dur_sub.median():.2f} mnt")
        c2.metric("Customer – Median",   f"{dur_cus.median():.2f} mnt")
        c3.metric("Rasio Durasi (Cust/Sub)", f"{dur_cus.median()/dur_sub.median():.1f}x")
        if p_mw < 0.05:
            st.success(f"✅ Mann-Whitney: **SIGNIFIKAN** (p={p_mw:.6f}) — perbedaan durasi bermakna secara statistik.")
        else:
            st.warning(f"⚠️ Mann-Whitney: Tidak Signifikan (p={p_mw:.6f})")

    st.markdown("---")
    st.subheader("Median Durasi per Kelompok Usia & Gender")
    age_order = ['Remaja/Dewasa Muda (<25)','Dewasa 25-34','Dewasa 35-44',
                 'Dewasa 45-54','Menengah 55-64','Lansia 65+']
    agg = (dff[dff['gender'].isin(['male','female'])]
             .groupby(['age_group','gender'])['tripduration_min']
             .median().reset_index())
    agg['age_group'] = pd.Categorical(agg['age_group'], categories=age_order, ordered=True)
    agg = agg.sort_values('age_group')
    fig = px.bar(agg, x='age_group', y='tripduration_min', color='gender', barmode='group',
                 color_discrete_map={'male': COLOR_BLUE, 'female': COLOR_ORANGE},
                 labels={'tripduration_min': 'Median Durasi (mnt)', 'age_group': 'Kelompok Usia'},
                 text_auto='.1f')
    fig.update_layout(height=420, xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)
    st.info("**Interpretasi:** Lansia 65+ bersepeda lebih lama (rekreasi). Perempuan secara konsisten berdurasi sedikit lebih panjang di hampir semua kelompok usia.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — STASIUN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📍 Stasiun":
    st.title("📍 Analisis Stasiun")

    n_top = st.slider("Top N Stasiun", 5, 20, 10)

    tab_dep, tab_arr = st.tabs(["🚀 Keberangkatan", "🏁 Kedatangan"])

    for tab, col, label, color in zip(
        [tab_dep, tab_arr],
        ['start_station_name', 'end_station_name'],
        ['Keberangkatan', 'Kedatangan'],
        [COLOR_BLUE, COLOR_ORANGE]
    ):
        with tab:
            if col not in dff.columns:
                st.warning("Kolom tidak tersedia.")
                continue
            top = dff[col].value_counts().head(n_top).reset_index()
            top.columns = ['station', 'n_trip']
            top['station_short'] = top['station'].str[:35]
            fig = px.bar(top.sort_values('n_trip'), x='n_trip', y='station_short',
                         orientation='h', color_discrete_sequence=[color],
                         labels={'n_trip': 'Jumlah Trip', 'station_short': ''},
                         title=f'Top {n_top} Stasiun {label}',
                         text='n_trip')
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Round-Trip per Usertype")
    agg_rt = dff.groupby('usertype').agg(
        n_trip=('is_round_trip','count'), n_roundtrip=('is_round_trip','sum')
    )
    agg_rt['pct_roundtrip'] = (agg_rt['n_roundtrip'] / agg_rt['n_trip'] * 100).round(1)
    st.dataframe(agg_rt, use_container_width=True)
    st.info("**Interpretasi:** Ketidakseimbangan keberangkatan vs kedatangan = sinyal utama kebutuhan *rebalancing* armada sepeda.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — TREN MUSIMAN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Tren Musiman":
    st.title("📈 Tren Musiman")

    trend = (dff.groupby(['start_month','start_month_name'])
                .agg(n_trip=('tripduration_min','count'),
                     median_dur=('tripduration_min','median'))
                .reset_index().sort_values('start_month'))
    trend['rolling_3m'] = trend['n_trip'].rolling(3, center=True).mean().round(0)

    if trend.empty:
        st.warning("Tidak ada data.")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=trend['start_month_name'], y=trend['n_trip'],
                             name='Jumlah Trip', marker_color=COLOR_BLUE, opacity=0.8,
                             text=trend['n_trip'], texttemplate='%{text:,}',
                             textposition='outside'), secondary_y=False)
        fig.add_trace(go.Scatter(x=trend['start_month_name'], y=trend['median_dur'],
                                  name='Median Durasi (mnt)', mode='lines+markers',
                                  line=dict(color=COLOR_RED, width=2.5),
                                  marker=dict(symbol='diamond', size=9)), secondary_y=True)
        fig.update_layout(title='Pola Musiman: Volume Trip & Durasi Median per Bulan',
                          height=450, legend=dict(x=0.01, y=0.99))
        fig.update_yaxes(title_text='Jumlah Trip', secondary_y=False)
        fig.update_yaxes(title_text='Median Durasi (mnt)', secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        peak = trend.loc[trend['n_trip'].idxmax()]
        low  = trend.loc[trend['n_trip'].idxmin()]
        c1, c2, c3 = st.columns(3)
        c1.metric("📈 Bulan Puncak",    peak['start_month_name'], f"{int(peak['n_trip']):,} trip")
        c2.metric("📉 Bulan Terendah",  low['start_month_name'],  f"{int(low['n_trip']):,} trip")
        c3.metric("Rasio Puncak/Terendah", f"{peak['n_trip']/low['n_trip']:.1f}x")

        st.info("**Interpretasi:** Volume meningkat di musim semi/panas, turun di musim dingin. Durasi lebih panjang saat volume rendah — pengguna *dedicated* tetap bersepeda meski cuaca buruk.")

        st.markdown("---")
        st.subheader("📊 Tabel Tren Bulanan")
        st.dataframe(trend[['start_month_name','n_trip','median_dur','rolling_3m']]
                     .rename(columns={'start_month_name':'Bulan','n_trip':'Jumlah Trip',
                                      'median_dur':'Median Durasi (mnt)','rolling_3m':'Rolling 3-Bulan'}),
                     use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ML CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Clustering":
    st.title("🤖 Segmentasi Pengguna — K-Means Clustering")

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    features = ['tripduration_min','age','start_hour','start_weekday']
    df_ml    = dff[features].dropna().copy()
    SAMPLE   = min(30000, len(df_ml))

    if len(df_ml) < 200:
        st.warning("Data terlalu sedikit untuk clustering. Longgarkan filter di sidebar.")
        st.stop()

    st.markdown(f"**Sample:** {SAMPLE:,} dari {len(df_ml):,} baris")
    df_s   = df_ml.sample(n=SAMPLE, random_state=42)
    scaler = StandardScaler()
    X      = scaler.fit_transform(df_s)

    k_range = range(2, 7)
    inertia, sil = [], []

    with st.spinner("Menghitung optimal k..."):
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertia.append(km.inertia_)
            sil.append(silhouette_score(X, km.labels_, sample_size=3000, random_state=42))

    best_k = list(k_range)[sil.index(max(sil))]

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(k_range), y=inertia, mode='lines+markers',
                                  line=dict(color=COLOR_BLUE, width=2.5),
                                  marker=dict(size=8)))
        fig.add_vline(x=best_k, line_color=COLOR_RED, line_dash='dash',
                      annotation_text=f'Optimal k={best_k}')
        fig.update_layout(title='Elbow Method', xaxis_title='k', yaxis_title='Inertia', height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(k_range), y=sil, mode='lines+markers',
                                  line=dict(color=COLOR_ORANGE, width=2.5),
                                  marker=dict(symbol='diamond', size=8)))
        fig.add_vline(x=best_k, line_color=COLOR_RED, line_dash='dash',
                      annotation_text=f'Optimal k={best_k}')
        fig.update_layout(title='Silhouette Score', xaxis_title='k',
                          yaxis_title='Silhouette', height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.success(f"✅ Cluster optimal: **k = {best_k}** (Silhouette = {max(sil):.4f})")

    km_final      = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_s          = df_s.copy()
    df_s['cluster'] = km_final.fit_predict(X).astype(str)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Jam vs Durasi (per Cluster)")
        fig = px.scatter(df_s.sample(min(5000, len(df_s))),
                         x='start_hour', y='tripduration_min', color='cluster',
                         color_discrete_sequence=PALETTE, opacity=0.4,
                         labels={'start_hour':'Jam','tripduration_min':'Durasi (mnt)','cluster':'Cluster'})
        fig.update_traces(marker_size=4)
        fig.update_layout(yaxis_range=[0,80], height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Usia vs Durasi (per Cluster)")
        fig = px.scatter(df_s.sample(min(5000, len(df_s))),
                         x='age', y='tripduration_min', color='cluster',
                         color_discrete_sequence=PALETTE, opacity=0.4,
                         labels={'age':'Usia','tripduration_min':'Durasi (mnt)','cluster':'Cluster'})
        fig.update_traces(marker_size=4)
        fig.update_layout(yaxis_range=[0,80], height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Profil Tiap Cluster")
    profile = df_s.groupby('cluster')[features].agg(['mean','median']).round(2)
    st.dataframe(profile, use_container_width=True)

    st.info(
        f"**Interpretasi:** K-Means mengidentifikasi **{best_k} segmen pengguna** "
        "berdasarkan durasi, usia, jam keberangkatan, dan hari. "
        "Setiap cluster mencerminkan pola perilaku unik untuk dasar strategi pemasaran."
    )
