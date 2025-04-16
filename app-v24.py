import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter

# -------------------------
# 1) CUSTOM CSS
# -------------------------
def set_custom_style():
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #FFFFFF !important; }
    [data-testid="stMarkdownContainer"] * { color: #333333 !important; }
    </style>
    """, unsafe_allow_html=True)

# -------------------------
# 2) ABI‑FUNKTSIOONID
# -------------------------
def unique_preserve_order(tags):
    seen = []
    for t in tags:
        if t not in seen:
            seen.append(t)
    return seen

def distribution_for_subtags(dff, chosen_tags):
    total = len(dff)
    counts = []
    for tg in chosen_tags:
        cnt = dff['ManualTagsList'].apply(lambda lst: tg in lst).sum()
        counts.append((tg, cnt))
    leftover = total - sum(c for _, c in counts)
    if leftover > 0:
        counts.append(("Muu", leftover))
    return pd.DataFrame(counts, columns=["Tag","Count"])

# -------------------------
# 3) GRAAFIKU‑FUNKTSIOONID
# -------------------------
def show_chart_single(df_counts, chart_title):
    measure = st.radio("Kuva andmed:", ["Arvudes","Protsentides"], key=chart_title+"_measure")
    color_choice = st.selectbox("Vali värvus",
        ["Sinine","Punane","Roheline","Lilla","Värviline"],
        key=chart_title+"_color"
    )

    palettes = {
      "Sinine": px.colors.sequential.Blues,
      "Punane": px.colors.sequential.Reds,
      "Roheline": px.colors.sequential.Greens,
      "Lilla": px.colors.sequential.Purples,
      "Värviline": px.colors.qualitative.Set2
    }
    pal = palettes[color_choice]

    dfp = df_counts.copy()
    cat, val = dfp.columns[0], dfp.columns[1]
    total = dfp[val].sum()
    if measure=="Protsentides" and total>0:
        dfp[val] = dfp[val] / total * 100

    diag = st.selectbox("Vali diagrammi tüüp:",
        ["Tulpdiagramm","Sektordiagramm","Mõlemad"], key=chart_title+"_diag"
    )

    common_layout = dict(
      template="plotly_white",
      paper_bgcolor='white', plot_bgcolor='white',
      title_font_color="#333333", font_color="#333333",
      xaxis=dict(
        title_font_color="#333333", tickfont_color="#333333",
        gridcolor="lightgray"
      ),
      yaxis=dict(
        title_font_color="#333333", tickfont_color="#333333",
        gridcolor="lightgray"
      ),
      legend=dict(font_color="#333333")
    )
    config = {
      "displayModeBar": True,
      "modeBarButtonsToAdd": ["toggleFullscreen"],
      "toImageButtonOptions": {"format":"png","scale":3}
    }

    # Tulpdiagramm
    if diag in ["Tulpdiagramm","Mõlemad"]:
        df_bar = dfp[dfp[cat]!="Muu"]
        if color_choice=="Värviline":
            palette_bar = pal
        else:
            # tumedam pool sequential paletist
            palette_bar = pal[len(pal)//2:][::-1]

        ylab = "Protsentides" if measure=="Protsentides" else "Arvudes"
        fig = px.bar(
            df_bar, x=cat, y=val,
            color=(cat if color_choice=="Värviline" else None),
            color_discrete_sequence=palette_bar,
            labels={cat:cat, val:ylab},
            title=f"{chart_title} – Tulpdiagramm"
        )
        if measure=="Protsentides":
            fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside", cliponaxis=False)
        else:
            fig.update_traces(texttemplate="%{y:d}", textposition="outside", cliponaxis=False)

        fig.update_layout(**common_layout)
        st.plotly_chart(fig, use_container_width=True, config=config)

    # Sektordiagramm
    if diag in ["Sektordiagramm","Mõlemad"]:
        df_sec = dfp[dfp["Tag"]!="Muu"]
        if color_choice=="Värviline":
            palette_sec = pal
        else:
            # tumedam pool sequential paletist
            palette_sec = pal[len(pal)//2:]
        ylab = "Protsentides" if measure=="Protsentides" else "Arvudes"
        fig = px.pie(
            df_sec, names=cat, values=val,
            color=(cat if color_choice=="Värviline" else None),
            color_discrete_sequence=palette_sec,
            labels={val:ylab},
            title=f"{chart_title} – Sektordiagramm"
        )
        fig.update_layout(**common_layout)
        st.plotly_chart(fig, use_container_width=True, config=config)

def show_time_chart(df_time, chart_title):
    measure = st.radio("Kuva andmed:", ["Arvudes","Protsentides"], key=chart_title+"_time_meas")
    color_choice = st.selectbox("Vali värvus",
        ["Sinine","Punane","Roheline","Lilla","Värviline"],
        key=chart_title+"_time_color"
    )

    palettes = {
      "Sinine": px.colors.sequential.Blues,
      "Punane": px.colors.sequential.Reds,
      "Roheline": px.colors.sequential.Greens,
      "Lilla": px.colors.sequential.Purples,
      "Värviline": px.colors.qualitative.Set2
    }
    pal = palettes[color_choice]

    dft = df_time.copy()
    if measure=="Protsentides" and dft['Count'].sum()>0:
        dft['Count'] = dft['Count'] / dft['Count'].sum() * 100

    diag = st.selectbox("Vali diagrammi tüüp:",
        ["Tulpdiagramm","Sektordiagramm","Mõlemad"],
        key=chart_title+"_time_diag"
    )

    common_layout = dict(
      template="plotly_white",
      paper_bgcolor='white', plot_bgcolor='white',
      title_font_color="#333333", font_color="#333333",
      xaxis=dict(
        title_font_color="#333333", tickfont_color="#333333",
        gridcolor="lightgray"
      ),
      yaxis=dict(
        title_font_color="#333333", tickfont_color="#333333",
        gridcolor="lightgray"
      ),
      legend=dict(font_color="#333333")
    )
    config = {
      "displayModeBar": True,
      "modeBarButtonsToAdd": ["toggleFullscreen"],
      "toImageButtonOptions": {"format":"png","scale":3}
    }

    # Tulpdiagramm
    if diag in ["Tulpdiagramm","Mõlemad"]:
        if color_choice=="Värviline":
            palette_bar = pal
        else:
            palette_bar = pal[len(pal)//2:][::-1]
        ylab = "Protsentides" if measure=="Protsentides" else "Arvudes"
        fig = px.bar(
            dft, x='Time', y='Count',
            color=('Time' if color_choice=="Värviline" else None),
            color_discrete_sequence=palette_bar,
            labels={'Time':'Aeg','Count':ylab},
            title=f"{chart_title} – Tulpdiagramm"
        )
        if measure=="Protsentides":
            fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside", cliponaxis=False)
        else:
            fig.update_traces(texttemplate="%{y:d}", textposition="outside", cliponaxis=False)
        fig.update_layout(**common_layout)
        st.plotly_chart(fig, use_container_width=True, config=config)

    # Sektordiagramm
    if diag in ["Sektordiagramm","Mõlemad"]:
        df_sec = dft.copy()
        if color_choice=="Värviline":
            palette_sec = pal
        else:
            palette_sec = pal[len(pal)//2:]
        ylab = "Protsentides" if measure=="Protsentides" else "Arvudes"
        fig = px.pie(
            df_sec, names='Time', values='Count',
            color=('Time' if color_choice=="Värviline" else None),
            color_discrete_sequence=palette_sec,
            labels={'Count':ylab},
            title=f"{chart_title} – Sektordiagramm"
        )
        fig.update_layout(**common_layout)
        st.plotly_chart(fig, use_container_width=True, config=config)

# -------------------------
# 4) PEAFUNKTSIOON
# -------------------------
def main():
    set_custom_style()

    st.title("AUTOMAKSU KAJASTUSTE TONAALSUSE JA KÕNEISIKUTE ANALÜÜS EPLi JA ERRi VEEBIUUDISTE NÄITEL")
    st.markdown("""
    Violeta Osula  
    BFM MA Nüüdismeedia meediauuringud  
    15.04.2025  

    Andmed pärinevad ERRi ja EPLi uudiste veebiportaalist, mis on avalikustatud ajavahemikus 19.07.23 – 16.08.24
    """)

    uploaded = st.file_uploader("Lae üles CSV-fail", type=["csv"])
    if not uploaded:
        st.warning("Palun lae CSV-fail, et jätkata.")
        return

    df_orig = pd.read_csv(uploaded, encoding="utf-8")
    keep = ["Item Type","Publication Year","Author","Title","Publication Title",
            "Url","Abstract Note","Date","Manual Tags","Editor"]
    cols = [c for c in keep if c in df_orig.columns]

    global df
    df = df_orig[cols].copy()
    st.write(f"Kokku on sisestatud {len(df)} väljaannet.")
    st.dataframe(df.head(20))

    df['Manual Tags'] = df['Manual Tags'].fillna('')
    df['ManualTagsList'] = df['Manual Tags'].apply(
        lambda x: unique_preserve_order(t.strip() for t in x.split(';') if t.strip())
    )

    # 1) Otsi sõnu
    st.subheader("Otsi sõnu (mitme sõna puhul kasuta koma)")
    txt = st.text_input("Sisesta otsitavad sõnad (nt 'kaja, kallas, jürgen'):")
    if txt:
        terms = [s.strip().lower() for s in txt.split(',') if s.strip()]
        cols_txt = df.select_dtypes(include=[object]).columns
        res = [(t, df.apply(lambda r: t in " ".join(str(r[c]).lower() for c in cols_txt), axis=1).sum()) for t in terms]
        df_cnt = pd.DataFrame(res, columns=["Term","Count"])
        show_chart_single(df_cnt, f"Otsi sõnu: {terms}")
    else:
        st.info("Sisesta sõnad, et kuvada iga termini diagramm.")

    # 2) Ajapõhine otsing
    st.subheader("Ajapõhine otsing")
    df['Date_parsed'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year']  = df['Date_parsed'].dt.year
    df['Month'] = df['Date_parsed'].dt.month
    ys = sorted(df['Year'].dropna().astype(int).unique())
    ms = list(range(1,13))

    yc = st.selectbox("Vali aasta", ["Kõik"]+[str(y) for y in ys])
    mc = st.selectbox("Vali kuu",  ["Kõik"]+[str(m) for m in ms])
    tw = st.text_input("Vali sõna (valikuline):")

    dft = df.copy()
    if tw:
        terms = [s.strip().lower() for s in tw.split(',') if s.strip()]
        cols_txt = dft.select_dtypes(include=[object]).columns
        dft = dft[dft.apply(lambda r: any(t in " ".join(str(r[c]).lower() for c in cols_txt) for t in terms), axis=1)]

    if yc=="Kõik" and mc=="Kõik":
        dg = dft.dropna(subset=['Year','Month'])
        dg['YM'] = dg['Year'].astype(int).astype(str)+"-"+dg['Month'].astype(int).astype(str)
        gc = dg.groupby("YM")['Title'].count().reset_index().rename(columns={'YM':'Time','Title':'Count'})
        show_time_chart(gc, "Kõik aastad ja kuud")
    elif yc!="Kõik" and mc!="Kõik":
        y,m = int(yc), int(mc)
        subset = dft[(dft['Year']==y)&(dft['Month']==m)]
        one = pd.DataFrame([[f"{y}-{m}", len(subset)]], columns=["Time","Count"])
        show_time_chart(one, f"Aasta {y}, kuu {m}")
    elif yc!="Kõik":
        y = int(yc)
        subset = dft[dft['Year']==y].copy()
        subset['Mo'] = subset['Month'].astype(int)
        gc = subset.groupby('Mo')['Title'].count().reset_index()
        gc['Time'] = gc['Mo'].astype(str).apply(lambda x: f"{y}-{x}")
        gc = gc[['Time','Title']].rename(columns={'Title':'Count'})
        show_time_chart(gc, f"Aasta {y}, kõik kuud")
    else:
        m = int(mc)
        subset = dft[dft['Month']==m]
        gc = subset.groupby('Year')['Title'].count().reset_index().rename(columns={'Year':'Time','Title':'Count'})
        show_time_chart(gc, f"Kuu {m}, kõik aastad")

    # 3) Peamine ja lisa märksõnad
    st.subheader("Vali peamine märksõna")
    tags = sorted({t for lst in df['ManualTagsList'] for t in lst})
    pt = st.selectbox("Märksõnad", ["(Vali)"]+tags)
    if pt!="(Vali)":
        dp = df[df['ManualTagsList'].apply(lambda lst: pt in lst)]
        st.write(f"Valitud märksõna: **{pt}** ({len(dp)} kirjet)")
        st.dataframe(dp.head(20))

        if 'Author' in dp.columns:
            ca = dp['Author'].value_counts().reset_index().rename(columns={'index':'Tag','Author':'Count'})
            show_chart_single(ca, pt)

        rel = sorted({t for lst in dp['ManualTagsList'] for t in lst if t!=pt})
        st.subheader("Lisa märksõnad")
        ms = st.multiselect("Märksõnad", options=rel)
        if ms:
            df_sub = distribution_for_subtags(dp, ms)
            show_chart_single(df_sub, f"{pt} + {ms}")
            for mtag in ms:
                subdf = dp[dp['ManualTagsList'].apply(lambda lst: mtag in lst)]
                st.write(f"{mtag}: {len(subdf)} kirjet")
                st.dataframe(subdf.head(15))
                if 'Author' in subdf.columns:
                    c2 = subdf['Author'].value_counts().reset_index().rename(columns={'index':'Tag','Author':'Count'})
                    show_chart_single(c2, f"{pt}, {mtag}")
        else:
            st.info("Vali lisa märksõnad, et näha jaotusi.")
    else:
        st.info("Palun vali peamine märksõna.")

if __name__=="__main__":
    main()
