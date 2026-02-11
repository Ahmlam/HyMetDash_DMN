# ==========================================================
# HyMetDash (SODEXAM/DMN/DEDE) — app_v2.py  (Carte OSM fixée)
# - Carte OSM : contour CIV + sous-bassins (Fanfar 1 & 2) colorés par vigilance
# - Stations cliquables (popup : station, sous-bassin, alerte)
# - Légende dynamique (comptes 🟢🟡🟠🔴)
# - Corrige: DataSourceError, geometry not set, choropleth/scattermapbox
# ==========================================================

import os
import io
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from PIL import Image

# ====== CONFIG GÉNÉRALE ======
st.set_page_config(page_title="HyMetDash — SODEXAM", layout="wide", page_icon="🌦️")
st.markdown(
    "<h2 style='text-align:center;color:#065535;'>🌍 Tableau de Bord Hydrométéorologique — HyMetDash 🌦️</h2>",
    unsafe_allow_html=True,
)

# === CHEMINS DES FICHIERS ===
DATA_EXCEL_DIR = "data/Donnees_observées_1"
STATIONS_FILE  = "data/Sous_bassins_Fanfar_Civ/522_Stations_CI.xlsx"
BASINS_DIR     = "data/Sous_bassins_Fanfar_Civ"
CIV_PATH       = "data/Sous_bassins_Fanfar_Civ/gadm36_CIV_4.shp"
LOGO_PATH      = "data/logo_SODEXAM.png"

# =======================
# UTILITAIRES & CALCULS
# =======================
def add_logo(fig):
    if os.path.exists(LOGO_PATH):
        fig.add_layout_image(
            dict(source=Image.open(LOGO_PATH),
                 xref="paper", yref="paper",
                 x=0, y=1.05, sizex=0.20, sizey=0.20,
                 xanchor="left", yanchor="bottom", layer="above")
        )
    return fig

def export_plot(fig, fname_prefix):
    import plotly.io as pio
    for fmt, mime in [("png","image/png"),("jpg","image/jpeg"),("pdf","application/pdf")]:
        buf = io.BytesIO()
        pio.write_image(fig, buf, format=fmt, scale=3)  # ~4K
        st.download_button(f"💾 Export {fmt.upper()}",
                           buf.getvalue(), file_name=f"{fname_prefix}.{fmt}", mime=mime,
                           key=f"dl_{fname_prefix}_{fmt}")
        buf.close()

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.strip().lower() for c in df.columns})

def _infer_value_column(df: pd.DataFrame):
    for c in df.columns:
        if c.lower() in ("parametre", "paramètre", "param", "valeur", "value"):
            return c
    ignore = {"date","année","annee","mois","jour","year","month","day","station","param"}
    nums = [c for c in df.columns if c.lower() not in ignore and pd.api.types.is_numeric_dtype(df[c])]
    return nums[0] if nums else None

def debit_pluie_debit(pluie_mm, C_runoff, area_km2):
    """Q_rr (m3/s) = C * P(mm/j) * A(km2) / 86.4"""
    if pluie_mm is None:
        return np.nan
    return (C_runoff * pluie_mm * area_km2) / 86.4

def derive_parameters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    # ETP (Hargreaves)
    if "tmin" in df.columns and "tmax" in df.columns:
        tmean = (df["tmin"] + df["tmax"]) / 2.0
        dtr = (df["tmax"] - df["tmin"]).clip(lower=0)
        df["etp"] = 0.0023 * (tmean + 17.8) * np.sqrt(dtr) * 0.408
    else:
        df["etp"] = np.nan
    # Rayonnement via insolation (kWh/m²/j) — schéma simple
    if "insolation" in df.columns:
        df["rayon_global"] = 0.8 * df["insolation"]
        df["rayon_direct"] = 0.7 * df["rayon_global"]
        df["rayon_diffus"] = 0.3 * df["rayon_global"]
    else:
        df["rayon_global"] = np.nan
        df["rayon_direct"] = np.nan
        df["rayon_diffus"] = np.nan
    # Bilan hydrique
    if "pluie" in df.columns:
        df["bilan_hydrique"] = df["pluie"].fillna(0) - df["etp"].fillna(0)
    else:
        df["bilan_hydrique"] = np.nan
    return df

# =======================
# CHARGEMENT (cache)
# =======================
@st.cache_data(show_spinner=True)
def load_stations() -> pd.DataFrame:
    df = pd.read_excel(STATIONS_FILE)
    return df.rename(columns={"Stations":"station","Longitude":"lon","Latitude":"lat"})

@st.cache_data(show_spinner=True)
def load_observed_wide() -> pd.DataFrame:
    if not os.path.exists(DATA_EXCEL_DIR):
        return pd.DataFrame()
    by_station = {}
    for fn in os.listdir(DATA_EXCEL_DIR):
        if not fn.lower().endswith(".xlsx") or fn.startswith("~$"):
            continue
        param_name = os.path.splitext(fn)[0].strip().lower()
        try:
            xls = pd.ExcelFile(os.path.join(DATA_EXCEL_DIR, fn))
        except Exception:
            continue
        for sheet in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet)
                if df is None or df.empty:
                    continue
                df = _norm_cols(df)
                if "date" in df.columns:
                    d = pd.to_datetime(df["date"], errors="coerce")
                else:
                    y = pd.to_numeric(df.get("annee") if "annee" in df.columns else df.get("année"), errors="coerce")
                    m = pd.to_numeric(df.get("mois"), errors="coerce")
                    dd = pd.to_numeric(df.get("jour"), errors="coerce")
                    d = pd.to_datetime(dict(year=y, month=m, day=dd), errors="coerce")
                val_col = _infer_value_column(df)
                if val_col is None:
                    continue
                sub = pd.DataFrame({"date": d, param_name: pd.to_numeric(df[val_col], errors="coerce")})
                sub = sub.dropna(subset=["date"]).sort_values("date")
                if sheet not in by_station:
                    by_station[sheet] = sub.set_index("date")
                else:
                    by_station[sheet] = by_station[sheet].join(sub.set_index("date"), how="outer")
            except Exception:
                continue
    frames = []
    for stn, dfx in by_station.items():
        dfx = dfx.sort_index().reset_index()
        dfx["station"] = stn
        frames.append(dfx)
    if not frames:
        return pd.DataFrame()
    big = pd.concat(frames, ignore_index=True)
    big["date"] = pd.to_datetime(big["date"], errors="coerce")
    big = big.dropna(subset=["date"])
    return big

@st.cache_data(show_spinner=True)
def load_basins_gdf() -> gpd.GeoDataFrame:
    # Lire explicitement les 2 couches
    g1 = gpd.read_file(BASINS_DIR, layer="Sous-Bassins-Fanfar-1")
    g2 = gpd.read_file(BASINS_DIR, layer="Sous-Bassins-Fanfar-2")
    g = pd.concat([g1, g2], ignore_index=True)
    if not isinstance(g, gpd.GeoDataFrame):
        g = gpd.GeoDataFrame(g, geometry="geometry")
    if g.crs is None or g.crs.to_epsg() != 4326:
        g = g.to_crs(epsg=4326)
    # Créer un identifiant stable pour matcher features
    g = g.reset_index(drop=True)
    g["fid"] = g.index.astype(str)
    # Tenter de déduire un nom de sous-bassin
    name_candidates = [c for c in g.columns if c.upper() in ("NAME","NOM","BASIN","SOUS_BASSIN","LAYER","SUBID")]
    g["basin_name"] = g[name_candidates[0]].astype(str) if name_candidates else g["fid"]
    return g

@st.cache_data(show_spinner=True)
def load_civ_boundary() -> gpd.GeoDataFrame:
    civ = gpd.read_file(CIV_PATH)
    if civ.crs is None or civ.crs.to_epsg() != 4326:
        civ = civ.to_crs(epsg=4326)
    return civ

# =======================
# PANNEAU LATÉRAL
# =======================
try:
    st.sidebar.image(LOGO_PATH, use_container_width=True)
except TypeError:
    st.sidebar.image(LOGO_PATH, use_column_width=True)

st.sidebar.header("🧭 Panneau de Contrôle")
section = st.sidebar.radio(
    "Aller à :",
    ["Carte / Spatialisation", "Stations Observées / Graphiques", "Débit et Paramètres hydrauliques", "Prévisions"]
)

# =======================
# CHARGEMENT DONNÉES
# =======================
with st.spinner("Chargement des données..."):
    df_all = load_observed_wide()
    basins = load_basins_gdf()
    civ = load_civ_boundary()
    stations_df = load_stations()

if df_all.empty:
    st.error("Aucune donnée observée trouvée. Vérifiez le dossier Excel.")
    st.stop()

df_all = derive_parameters(df_all)

# =======================
# ONGLET — CARTE / SPATIALISATION (corrigé)
# =======================
if section == "Carte / Spatialisation":
    st.subheader("🗺️ Carte OSM — Sous-bassins et Stations avec vigilance")

    # Paramètres d’alerte (globaux)
    st.markdown("**Code couleur alerte :** 🟢 Vert = aucun impact · 🟡 Jaune = faible impact · 🟠 Orange = impact moyen · 🔴 Rouge = impact élevé:")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rain_thr_2d = st.number_input("Seuil pluie 2 jours (mm)", 0.0, 600.0, 50.0, 50.0)
    with c2:
        q_thr = st.number_input("Seuil débit (m³/s)", 0.0, 1000.0, 10.0, 5.0)
    with c3:
        C_runoff = st.slider("Coef. ruissellement C", 0.0, 1.0, 0.5, 0.05)
    with c4:
        area_km2_default = st.number_input("Aire sous-bassin (km²)", 0.0, 10000.0, 5000.0, 10.0)

    # Calculs alerte par station (dernière valeur)
    df_all = df_all.sort_values("date")
    df_all["pluie_2d"] = df_all.groupby("station")["pluie"].transform(lambda s: s.fillna(0).rolling(2, min_periods=1).sum())
    df_all["Q_rr"] = debit_pluie_debit(df_all.get("pluie", np.nan), C_runoff, area_km2_default)

    def alert_color(pl2d, q):
        if (pd.notna(pl2d) and pl2d >= rain_thr_2d) and (pd.notna(q) and q >= q_thr):
            return "red"
        elif (pd.notna(pl2d) and pl2d >= 0.75*rain_thr_2d) or (pd.notna(q) and q >= 0.75*q_thr):
            return "orange"
        elif (pd.notna(pl2d) and pl2d >= 0.5*rain_thr_2d) or (pd.notna(q) and q >= 0.5*q_thr):
            return "yellow"
        return "green"

    latest = df_all.groupby("station").tail(1).copy()
    latest["alert_color"] = latest.apply(lambda r: alert_color(r.get("pluie_2d"), r.get("Q_rr")), axis=1)

    # GeoDataFrame des stations
    gdf_st = gpd.GeoDataFrame(
        stations_df.copy(),
        geometry=gpd.points_from_xy(stations_df["lon"], stations_df["lat"]),
        crs="EPSG:4326"
    )
    # Attacher couleurs d’alerte aux stations (merge par station)
    gdf_st = gdf_st.merge(latest[["station","alert_color","pluie_2d","Q_rr"]], on="station", how="left")

    # Jointure spatiale stations ↔ sous-bassins
    gdf_st = gdf_st.sjoin(basins[["fid","basin_name","geometry"]], how="left", predicate="intersects")

    # Sévérité max par sous-bassin
    sev_map = {"green": 0, "yellow": 1, "orange": 2, "red": 3}
    inv_sev_map = {v: k for k, v in sev_map.items()}
    basin_sev = (
        gdf_st.dropna(subset=["fid"])
             .assign(sev=lambda d: d["alert_color"].map(sev_map).fillna(0).astype(int))
             .groupby("fid")["sev"].max()
             .reindex(basins["fid"], fill_value=0)
    )
    basins["alert_color"] = basin_sev.map(inv_sev_map)

    # Comptages pour légende
    counts = basins["alert_color"].value_counts().to_dict()
    n_green = counts.get("green", 0)
    n_yellow = counts.get("yellow", 0)
    n_orange = counts.get("orange", 0)
    n_red = counts.get("red", 0)

    st.markdown(
        f"""
        <div style="padding:6px 10px;border:1px solid #eee;border-radius:8px;background:#fff;display:inline-block;">
        <b>Légende (Sous-bassins):</b>
        <span style="margin-left:10px;">🟢 aucun impact: {n_green}</span>
        <span style="margin-left:10px;">🟡 faible impact: {n_yellow}</span>
        <span style="margin-left:10px;">🟠 impact moyen: {n_orange}</span>
        <span style="margin-left:10px;">🔴 impact élevé: {n_red}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sous-bassins colorés (catégoriel)
    color_map = {"green":"#4caf50","yellow":"#ffeb3b","orange":"#ff9800","red":"#f44336"}
    basins_geojson = basins.drop(columns=[], errors="ignore").copy()
    # Assure une clé d'identité stable
    #basins_geojson = basins_geojson.copy()
    basins_geojson["fid"] = basins_geojson["fid"].astype(str)

    fig = px.choropleth_map(
        basins_geojson,
        geojson=basins_geojson.__geo_interface__,
        locations="fid",
        featureidkey="properties.fid",
        color="alert_color",
        color_discrete_map=color_map,
        map_style="open-street-map",
        center={"lat": 7.5, "lon": -5.5},
        zoom=6,
        opacity=0.45,
        hover_name="basin_name",
        hover_data={"alert_color": True, "fid": False},
        title="Vigilance sur les Sous-bassins — Côte d’Ivoire"
    )

    # Ajout contour national (tracé lignes)
    # On simplifie : on utilise le bounding linework via choropleth "transparent" + ligne séparée
    # Pour plus de robustesse, on trace le contour avec Scattermapbox pour chaque anneau extérieur
    def add_boundary_scatter(fig, gdf, line_color="black", line_width=1.2, name="Contour national"):
        from shapely.geometry import Polygon, MultiPolygon, LinearRing
        for geom in gdf.geometry:
            if geom is None:
                continue
            if geom.geom_type == "Polygon":
                polys = [geom]
            elif geom.geom_type == "MultiPolygon":
                polys = list(geom.geoms)
            else:
                continue
            for poly in polys:
                xs, ys = poly.exterior.xy
                fig.add_trace(go.Scattermap(
                    lon=list(xs), lat=list(ys), mode="lines",
                    line=dict(color=line_color, width=line_width), name=name, hoverinfo="skip"
                ))
        return fig

    fig = add_boundary_scatter(fig, civ, line_color="black", line_width=1.5, name="Contour CIV")

    # Infobulles stations (station, sous-bassin, alerte + métriques récentes)
    # Couleur du point = alerte station
    station_marker_colors = gdf_st["alert_color"].map(color_map).fillna("#2196f3")
    fig.add_trace(go.Scattermap(
        lon=gdf_st["lon"],
        lat=gdf_st["lat"],
        mode="markers",
        marker=dict(size=10, color=station_marker_colors, opacity=0.9),
        text=[
            (
                f"<b>{r['station']}</b><br>"
                f"Sous-bassin : {r.get('basin_name','N/A')}<br>"
                f"Alerte : {str(r.get('alert_color','green')).upper()}<br>"
                f"Pluie 2j : {0 if pd.isna(r.get('pluie_2d')) else round(float(r.get('pluie_2d')),1)} mm<br>"
                f"Q_rr : {0 if pd.isna(r.get('Q_rr')) else round(float(r.get('Q_rr')),2)} m³/s"
            )
            for _, r in gdf_st.iterrows()
        ],
        hoverinfo="text",
        name="Stations"
    ))

    add_logo(fig)
    st.plotly_chart(fig, use_container_width=True, key="map_main")
    export_plot(fig, "carte_spatialisation")

    # Liste en-tête des sous-bassins en vigilance (jaune/orange/rouge) avec stations associées
    alert_subs = basins[basins["alert_color"].isin(["yellow","orange","red"])]
    if not alert_subs.empty:
        # Stations dans ces sous-bassins
        alert_station_list = gdf_st[gdf_st["fid"].isin(alert_subs["fid"])]
        grp = (alert_station_list.groupby(["basin_name","alert_color"])["station"]
               .apply(lambda s: ", ".join(sorted(s.unique())))
               .reset_index())
        st.markdown("**🔔 Les sous-bassins et leurs stations en vigilance sont :**")
        for _, row in grp.iterrows():
            badge = {"yellow":"🟡","orange":"🟠","red":"🔴"}.get(row["alert_color"], "⚠️")
            st.write(f"{badge} **{row['basin_name']}** → {row['station']}")
    else:
        st.info("Aucun sous-bassin en vigilance (jaune/orange/rouge).")

# =======================
# ONGLET — STATIONS (inchangé)
# =======================
elif section == "Stations Observées / Graphiques":
    st.subheader("📊 Visualisation par station / paramètre")
    station_sel = st.selectbox("Station :", sorted(df_all["station"].unique()))
    num_cols = [c for c in df_all.columns if c not in ("station","date") and pd.api.types.is_numeric_dtype(df_all[c])]
    preferred = ["pluie","etp","bilan_hydrique","rayon_global","rayon_direct","rayon_diffus","tmax","tmin","vent","insolation"]
    ordered = [p for p in preferred if p in num_cols] + [c for c in num_cols if c not in preferred]
    param_sel = st.selectbox("Paramètre :", ordered)

    df_st = df_all[df_all["station"]==station_sel].dropna(subset=["date"]).sort_values("date")
    if df_st.empty:
        st.warning("Pas de données pour cette station.")
        st.stop()

    min_d = df_st["date"].min().date()
    max_d = df_st["date"].max().date()
    dr = st.date_input("🗓️ Période", [min_d, max_d], format="DD/MM/YYYY")
    if isinstance(dr, (tuple, list)) and len(dr)==2:
        start_d, end_d = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
    else:
        start_d, end_d = pd.to_datetime(min_d), pd.to_datetime(max_d)
    data = df_st[(df_st["date"]>=start_d)&(df_st["date"]<=end_d)].copy()

    graph_type = st.selectbox("Type de graphique", ["Courbe","Histogramme","Box-plot","Nuage de points","Camembert","3D"])
    color_map = {
        "pluie":"#1f77b4", "tmax":"#d62728", "tmin":"#6baed6", "rayon_global":"#ffd166",
        "rayon_direct":"#faae2b", "rayon_diffus":"#bdbdbd", "etp":"#ab47bc", "bilan_hydrique":"#2e8b57",
        "vent":"#17becf", "insolation":"#ffcc66"
    }
    base_color = color_map.get(param_sel, "#0a6fb6")

    if graph_type=="Courbe":
        fig = px.line(data, x="date", y=param_sel, markers=True, color_discrete_sequence=[base_color],
                      title=f"{param_sel} — {station_sel}")
    elif graph_type=="Histogramme":
        fig = px.bar(data, x="date", y=param_sel, color_discrete_sequence=[base_color],
                     title=f"{param_sel} — {station_sel}")
    elif graph_type=="Box-plot":
        fig = px.box(data, y=param_sel, color_discrete_sequence=[base_color],
                     title=f"Distribution {param_sel} — {station_sel}")
    elif graph_type=="Nuage de points":
        fig = px.scatter(data, x="date", y=param_sel, color_discrete_sequence=[base_color],
                         title=f"{param_sel} — {station_sel}")
    elif graph_type=="Camembert":
        data["mois"] = data["date"].dt.month
        df_mean = data.groupby("mois")[param_sel].mean().reset_index()
        fig = px.pie(df_mean, values=param_sel, names="mois", title=f"{param_sel} — {station_sel}")
    else:
        fig = px.scatter_3d(data, x="date", y="station", z=param_sel, color=param_sel,
                            title=f"{param_sel} — {station_sel}")

    add_logo(fig)
    st.plotly_chart(fig, use_container_width=True)
    export_plot(fig, f"{station_sel}_{param_sel}")

    st.download_button("📥 Export CSV (période/param)",
                       data.to_csv(index=False).encode("utf-8"),
                       file_name=f"{station_sel}_{param_sel}_{start_d.date()}_{end_d.date()}.csv",
                       mime="text/csv")

# =======================
# ONGLET — DÉBIT (inchangé)
# =======================
elif section== "Débit — Pluie en Débit (C,A) & Manning/Strickler":
    st.subheader("💧 Débit — Pluie en Débit (C,A) & Manning/Strickler")

    station_q = st.selectbox("Station (pluie pour Q_rr) :", sorted(df_all["station"].unique()))
    dfq = df_all[df_all["station"]==station_q].dropna(subset=["date"]).sort_values("date")

    c1, c2, c3 = st.columns(3)
    with c1:
        C_run = st.slider("Coefficient de ruissellement C", 0.0, 1.0, 0.5, 0.05)
    with c2:
        area_km2 = st.number_input("Aire du sous-bassin (km²)", 0.0, 1e6, 5000.0, 10.0)
    with c3:
        agg = st.selectbox("Agrégation", ["journalière","hebdomadaire","mensuelle"])

    dfq["Q_rr"] = debit_pluie_debit(dfq.get("pluie", np.nan), C_run, area_km2)

    if agg=="hebdomadaire":
        q_plot = dfq.set_index("date")["Q_rr"].resample("W").mean().reset_index()
    elif agg=="mensuelle":
        q_plot = dfq.set_index("date")["Q_rr"].resample("MS").mean().reset_index()
    else:
        q_plot = dfq[["date","Q_rr"]].copy()

    fig_q = px.line(q_plot, x="date", y="Q_rr", markers=True, color_discrete_sequence=["#2e8b57"],
                    title=f"Débit estimé (C={C_run:.2f}, A={area_km2:.0f} km²) — {station_q}")
    add_logo(fig_q)
    st.plotly_chart(fig_q, use_container_width=True)
    export_plot(fig_q, f"Qrr_{station_q}")

    st.markdown("---")
    st.subheader("⚙️ Estimation hydraulique — Manning/Strickler")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        use_n = st.toggle("Saisir n (Manning) ?", value=True)
    with c2:
        n_manning = st.number_input("n (Manning)", 0.010, 0.150, 0.035, 0.001)
    with c3:
        k_strick = st.number_input("k (Strickler, m^(1/3)/s)", 5.0, 150.0, 28.6, 0.1)
    with c4:
        use_k = st.toggle("Saisir k (Strickler) ?", value=False)

    c5,c6,c7 = st.columns(3)
    with c5:
        A_m2 = st.number_input("Aire mouillée A (m²)", 0.0, 1e5, 50.0, 1.0)
    with c6:
        R_m  = st.number_input("Rayon hydraulique R (m)", 0.0, 1000.0, 1.0, 0.01)
    with c7:
        S    = st.number_input("Pente énergétique S (m/m)", 0.0, 1.0, 0.001, 0.0001, format="%.4f")

    k_use = (1.0 / n_manning) if (use_n and not use_k) else (k_strick if use_k else None)
    Q_man = (k_use * A_m2 * (R_m ** (2/3.0)) * (S ** 0.5)) if k_use and A_m2 and R_m and S else np.nan
    st.metric("Débit (Manning/Strickler) estimé", f"{(Q_man if Q_man==Q_man else 0):.3f} m³/s")
    st.info("Rappel : k = 1/n. Si les deux sont fournis, priorité donnée à k (Strickler).")
    
#labels = config_language[current_lang]

# === Onglet "Prévisions" pour HyMetDash ===

elif section == "Prévisions":
    st.subheader("Prévision des 10 prochains jours")

    # Lecture des stations (coordonnées) 
    stations_df = pd.read_excel(STATIONS_FILE)
    station_list = stations_df['Stations'].dropna().unique().tolist()

    # Génération de la plage de dates (10 jours à 3h)
    start_time = pd.Timestamp.now().floor('H')
    dates = pd.date_range(start_time, periods=80, freq='3H')

    # Paramètres fictifs disponibles
    param_list = ['Precipitation', 'Temperature']

    # Génération de données factices (à remplacer plus tard par GFS réel)
    @st.cache_data
    def generate_forecast_data():
        forecast_data = []
        for station in station_list:
            for t in dates:
                precip_val = np.random.uniform(0, 20)
                hour_frac = (t.hour + t.minute/60) / 24.0 * 2 * np.pi
                temp_base = 22 + 6 * np.sin(hour_frac)
                temp_val = temp_base + np.random.normal(0, 1)
                forecast_data.append((station, t, 'Precipitation', round(precip_val, 2)))
                forecast_data.append((station, t, 'Temperature', round(temp_val, 2)))
        return pd.DataFrame(forecast_data, columns=["Station", "Datetime", "Parameter", "Value"])

    forecast_df = generate_forecast_data()

    # Interface utilisateur
    selected_station = st.selectbox("Station", options=station_list)
    selected_param = st.selectbox("Paramètre", options=param_list)

    all_dates = sorted({d.date() for d in dates})
    date_options = ["Toutes les dates"] + [str(d) for d in all_dates]
    selected_date_option = st.selectbox("Date", options=date_options)

    forecast_filtered = forecast_df[(forecast_df['Station'] == selected_station) &
                                    (forecast_df['Parameter'] == selected_param)]

    if selected_date_option != "Toutes les dates":
        filter_date = pd.to_datetime(selected_date_option).date()
        data_to_plot = forecast_filtered[ forecast_filtered['Datetime'].dt.date == filter_date ]
    else:
        data_to_plot = forecast_filtered

    # Affichage graphique
    if selected_date_option == "Toutes les dates":
        pivot = data_to_plot.pivot_table(index=data_to_plot['Datetime'].dt.date,
                                         columns=data_to_plot['Datetime'].dt.hour,
                                         values='Value')
        heatmap_fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[f"{h}h" for h in pivot.columns],
            y=pivot.index.astype(str).tolist(),
            colorscale='YlGnBu',
            colorbar_title=selected_param
        ))
        heatmap_fig.update_layout(title=f"{selected_param} - {selected_station} (10 jours)")
        st.plotly_chart(heatmap_fig, use_container_width=True)
    else:
        fig = px.line(data_to_plot, x='Datetime', y='Value',
                      title=f"{selected_param} - {selected_station} ({selected_date_option})")
        st.plotly_chart(fig, use_container_width=True)

    # Export CSV
    csv_data = data_to_plot[['Datetime','Value']].to_csv(index=False).encode('utf-8')
    st.download_button(label="Télécharger les données CSV", data=csv_data,
                       file_name=f"forecast_{selected_station}_{selected_param}.csv", mime="text/csv")

    # === Nouvelle sous-section : Débits prévus ===
    st.markdown("---")
    st.subheader("💧 Débits prévus (à partir des précipitations GFS)")

    station_q = selected_station

    dfq = forecast_df[(forecast_df['Station'] == station_q) & (forecast_df['Parameter'] == 'Precipitation')].copy()
    dfq = dfq.rename(columns={"Datetime": "date", "Value": "pluie"})

    c1, c2, c3 = st.columns(3)
    with c1:
        C_run = st.slider("Coefficient de ruissellement C", 0.0, 1.0, 0.5, 0.05)
    with c2:
        area_km2 = st.number_input("Aire du sous-bassin (km²)", 0.0, 1e6, 5000.0, 10.0)
    with c3:
        agg = st.selectbox("Agrégation", ["3h", "journalière", "hebdomadaire", "mensuelle"], index=1)

    dfq["Q_rr"] = debit_pluie_debit(dfq["pluie"], C_run, area_km2)

    if agg == "journalière":
        q_plot = dfq.set_index("date")["Q_rr"].resample("D").mean().reset_index()
    elif agg == "hebdomadaire":
        q_plot = dfq.set_index("date")["Q_rr"].resample("W").mean().reset_index()
    elif agg == "mensuelle":
        q_plot = dfq.set_index("date")["Q_rr"].resample("MS").mean().reset_index()
    else:
        q_plot = dfq[["date", "Q_rr"]].copy()

    fig_q = px.line(q_plot, x="date", y="Q_rr", markers=True, color_discrete_sequence=["#2e8b57"],
                    title=f"Débit prévu estimé — {station_q}")
    st.plotly_chart(fig_q, use_container_width=True)

    st.markdown("### ⚙️ Estimation hydraulique — Manning/Strickler")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        use_n = st.toggle("Saisir n (Manning) ?", value=True)
    with c2:
        n_manning = st.number_input("n (Manning)", 0.010, 0.150, 0.035, 0.001)
    with c3:
        k_strick = st.number_input("k (Strickler, m^(1/3)/s)", 5.0, 150.0, 28.6, 0.1)
    with c4:
        use_k = st.toggle("Saisir k (Strickler) ?", value=False)

    c5,c6,c7 = st.columns(3)
    with c5:
        A_m2 = st.number_input("Aire mouillée A (m²)", 0.0, 1e5, 50.0, 1.0)
    with c6:
        R_m  = st.number_input("Rayon hydraulique R (m)", 0.0, 1000.0, 1.0, 0.01)
    with c7:
        S    = st.number_input("Pente énergétique S (m/m)", 0.0, 1.0, 0.001, 0.0001, format="%.4f")

    k_use = (1.0 / n_manning) if (use_n and not use_k) else (k_strick if use_k else None)
    Q_man = (k_use * A_m2 * (R_m ** (2/3.0)) * (S ** 0.5)) if k_use and A_m2 and R_m and S else np.nan
    st.metric("Débit (Manning/Strickler) estimé", f"{(Q_man if Q_man==Q_man else 0):.3f} m³/s")
