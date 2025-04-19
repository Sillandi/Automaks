import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter

# ----------------------
# 1) STYLES
# ----------------------
def set_custom_style():
    st.markdown("""
    <style>
      [data-testid="stAppViewContainer"] { background-color: #FFFFFF !important; }
      [data-testid="stMarkdownContainer"] * { color: #333333 !important; }
      .pill { display:inline-block; padding:4px 8px; margin:2px; border-radius:4px; font-size:90%; }
      .pill-tag { background-color:#FFCCCC; color:#000000; }
      .pill-text { background-color:#DDDDDD; color:#000000; }
    </style>
    """, unsafe_allow_html=True)

# ----------------------
# 2) HELPERS
# ----------------------
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

# ----------------------
# 3) CHART FUNCTIONS
# ----------------------
def show_chart_single(df_counts, chart_title):
    cat, val = df_counts.columns[0], df_counts.columns[1]
    measure = st.radio("Kuva andmed:", ["Arvudes","Protsentides"], key=chart_title+"_measure")
    color_choice = st.selectbox("Vali värvus",
        ["Sinine","Punane","Roheline","Lilla","Värviline"], key=chart_title+"_color")
    palettes = {
      "Sinine": px.colors.sequential.Blues,
      "Punane": px.colors.sequential.Reds,
      "Roheline": px.colors.sequential.Greens,
      "Lilla": px.colors.sequential.Purples,
      "Värviline": px.colors.qualitative.Set2
    }
    pal = palettes[color_choice]

    dfp = df_counts.copy()
    if measure=="Protsentides" and dfp[val].sum()>0:
        dfp[val] = dfp[val]/dfp[val].sum()*100

    diag = st.selectbox("Vali diagrammi tüüp:",
        ["Tulpdiagramm","Sektordiagramm","Mõlemad"], key=chart_title+"_diag")

    layout_common = dict(
      template="plotly_white",
      paper_bgcolor='white',
      plot_bgcolor='white',
      title_font_color="#333333",
      font_color="#333333",
      xaxis=dict(
        title_font_color="#333333",
        tickfont_color="#333333",
        gridcolor="lightgray"
      ),
      yaxis=dict(
        title_font_color="#333333",
        tickfont_color="#333333",
        gridcolor="lightgray"
      ),
      legend=dict(font_color="#333333")
    )
    config = {"displayModeBar": True, "modeBarButtonsToAdd": ["toggleFullscreen"],
              "toImageButtonOptions": {"format":"png","scale":3}}

    # Tulpdiagramm
    if diag in ["Tulpdiagramm","Mõlemad"]:
        df_bar = dfp[dfp[cat]!="Muu"]
        palette_bar = pal if color_choice=="Värviline" else pal[len(pal)//2:][::-1]
        ylab = "Protsentides" if measure=="Protsentides" else "Arvudes"
        fig = px.bar(
          df_bar, x=cat, y=val,
          color=(cat if color_choice=="Värviline" else None),
          color_discrete_sequence=palette_bar,
          labels={cat:cat, val:ylab},
          title=f"{chart_title} – Tulpdiagramm"
        )
        fmt = "%{y:.1f}%" if measure=="Protsentides" else "%{y:d}"
        fig.update_traces(
            texttemplate=fmt,
            textposition="outside",
            cliponaxis=False,
            textfont_color="#333333"
        )
        fig.update_layout(**layout_common)
        st.plotly_chart(fig, use_container_width=True, config=config)

    # Sektordiagramm
    if diag in ["Sektordiagramm","Mõlemad"]:
        df_sec = dfp[dfp[cat]!="Muu"]
        palette_sec = pal if color_choice=="Värviline" else pal[len(pal)//2:]
        val_lab = "Protsentides" if measure=="Protsentides" else "Arvudes"
        fig = px.pie(
          df_sec, names=cat, values=val,
          color=(cat if color_choice=="Värviline" else None),
          color_discrete_sequence=palette_sec,
          labels={val:val_lab},
          title=f"{chart_title} – Sektordiagramm"
        )
        fig.update_layout(**layout_common)
        st.plotly_chart(fig, use_container_width=True, config=config)

def show_time_chart(df_time, chart_title):
    measure = st.radio("Kuva andmed:", ["Arvudes","Protsentides"], key=chart_title+"_time_meas")
    color_choice = st.selectbox("Vali värvus",
        ["Sinine","Punane","Roheline","Lilla","Värviline"], key=chart_title+"_time_color")
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
        dft['Count'] = dft['Count']/dft['Count'].sum()*100

    diag = st.selectbox("Vali diagrammi tüüp:",
        ["Tulpdiagramm","Sektordiagramm","Mõlemad"], key=chart_title+"_time_diag")

    layout_common = dict(
      template="plotly_white",
      paper_bgcolor='white',
      plot_bgcolor='white',
      title_font_color="#333333",
      font_color="#333333",
      xaxis=dict(
        title_font_color="#333333",
        tickfont_color="#333333",
        gridcolor="lightgray"
      ),
      yaxis=dict(
        title_font_color="#333333",
        tickfont_color="#333333",
        gridcolor="lightgray"
      ),
      legend=dict(font_color="#333333")
    )
    config = {"displayModeBar": True, "modeBarButtonsToAdd": ["toggleFullscreen"],
              "toImageButtonOptions": {"format":"png","scale":3}}

    # Tulpdiagramm
    if diag in ["Tulpdiagramm","Mõlemad"]:
        palette_bar = pal if color_choice=="Värviline" else pal[len(pal)//2:][::-1]
        ylab = "Protsentides" if measure=="Protsentides" else "Arvudes"
        fig = px.bar(
          dft, x='Time', y='Count',
          color=('Time' if color_choice=="Värviline" else None),
          color_discrete_sequence=palette_bar,
          labels={'Time':'Aeg','Count':ylab},
          title=f"{chart_title} – Tulpdiagramm"
        )
        fmt = "%{y:.1f}%" if measure=="Protsentides" else "%{y:d}"
        fig.update_traces(
            texttemplate=fmt,
            textposition="outside",
            cliponaxis=False,
            textfont_color="#333333"
        )
        fig.update_layout(**layout_common)
        st.plotly_chart(fig, use_container_width=True, config=config)

    # Sektordiagramm
    if diag in ["Sektordiagramm","Mõlemad"]:
        df_sec = dft.copy()
        palette_sec = pal if color_choice=="Värviline" else pal[len(pal)//2:]
        lab = "Protsentides" if measure=="Protsentides" else "Arvudes"
        fig = px.pie(
          df_sec, names='Time', values='Count',
          color=('Time' if color_choice=="Värviline" else None),
          color_discrete_sequence=palette_sec,
          labels={'Count':lab},
          title=f"{chart_title} – Sektordiagramm"
        )
        fig.update_layout(**layout_common)
        st.plotly_chart(fig, use_container_width=True, config=config)

# ----------------------
# 4) MAIN
# ----------------------
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

    global df, df_search_base
    df = df_orig[cols].copy()
    df['Manual Tags'] = df['Manual Tags'].fillna('')
    df['ManualTagsList'] = df['Manual Tags'].apply(
      lambda x: unique_preserve_order(t.strip() for t in x.split(';') if t.strip())
    )

    # --- 1) Otsi sõnu + väljanne ---
    st.subheader("Otsi sõnu (mitme sõna puhul kasuta koma)")
    pub1 = st.radio("Väljaanne:", ["Kõik","EPL","ERR"], key="pub1")
    df1 = df if pub1=="Kõik" else df[df["Publication Title"]==pub1]

    txt = st.text_input("Sisesta otsitavad sõnad ja/või märksõnad:", key="searchtxt")
    if txt:
        terms = [s.strip() for s in txt.split(',') if s.strip()]
        alltags = {t.lower() for lst in df1['ManualTagsList'] for t in lst}
        tag_terms = [t for t in terms if t.lower() in alltags]
        text_terms = [t for t in terms if t.lower() not in alltags]

        df_search_base = df1[
          df1['ManualTagsList'].apply(
            lambda lst: any(t.lower()==x.lower() for x in lst for t in tag_terms)
          )
          |
          df1.apply(lambda r: any(
              t.lower() in str(r["Title"]).lower() or
              t.lower() in str(r["Abstract Note"]).lower()
              for t in text_terms), axis=1)
        ]

        pills = []
        for t in terms:
            cls = "pill-tag" if t.lower() in tag_terms else "pill-text"
            pills.append(f"<span class='pill {cls}'>{t}</span>")
        st.markdown(" ".join(pills), unsafe_allow_html=True)

        # Eraldi märksõna vs tekstisõna arvud
        text_count = df_search_base.apply(
            lambda r: any(t.lower() in str(r["Title"]).lower() or t.lower() in str(r["Abstract Note"]).lower()
                          for t in text_terms), axis=1
        ).sum()
        tag_count = len(df_search_base) - text_count
        st.write(f"Otsingu tulemusi – **{len(df_search_base)}** vastet "
                 f"({tag_count} Märksõna, {text_count} Tekstisõna)")

        st.dataframe(df_search_base.head(20))

        if pub1=="Kõik":
            rec=[]
            for t in terms:
                for src in ["ERR","EPL"]:
                    sub = df_search_base[df_search_base["Publication Title"]==src]
                    if t.lower() in tag_terms:
                        ct = sub['ManualTagsList'].apply(
                            lambda lst: any(x.lower()==t.lower() for x in lst)
                        ).sum()
                    else:
                        ct = sub.apply(
                            lambda r: t.lower() in str(r["Title"]).lower() or
                                      t.lower() in str(r["Abstract Note"]).lower(),
                            axis=1
                        ).sum()
                    rec.append((t, src, int(ct)))
            df_tp = pd.DataFrame(rec,columns=["Tag","Publication Title","Count"])
            fig = px.bar(
              df_tp, x="Tag", y="Count", color="Publication Title",
              color_discrete_map={'ERR':'#003366','EPL':'#4a90e2'},
              labels={'Count':'Arvudes'},
              title=f'Otsi sõnu – {" & ".join(terms)} jaotus (ERR vs EPL)'
            )
            fmt = "%{y:d}"
            fig.update_traces(texttemplate=fmt, textposition="outside", textfont_color="#333333")
            fig.update_layout(
              template="plotly_white", paper_bgcolor='white', plot_bgcolor='white',
              xaxis=dict(gridcolor="lightgray", title_font_color="#333333", tickfont_color="#333333"),
              yaxis=dict(gridcolor="lightgray", title_font_color="#333333", tickfont_color="#333333"),
              legend=dict(font_color="#333333")
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":True})
        else:
            ca = Counter()
            for t in terms:
                if t.lower() in tag_terms:
                    ca[t] = df_search_base['ManualTagsList'].apply(
                        lambda lst: any(x.lower()==t.lower() for x in lst)
                    ).sum()
                else:
                    ca[t] = int(df_search_base.apply(
                        lambda r: t.lower() in str(r["Title"]).lower() or
                                  t.lower() in str(r["Abstract Note"]).lower(),
                        axis=1
                    ).sum())
            df_terms = pd.DataFrame(ca.items(), columns=["Tag","Count"])
            show_chart_single(df_terms, "Otsi sõnu – termide jaotus")
    else:
        df_search_base = df1
        st.info("Sisesta sõnad või märksõnad, et kuvada tulemusi.")

    # --- 2) Ajapõhine otsing ---
    st.subheader("Ajapõhine otsing")
    df_base = df_search_base.copy()
    df_base['Date_parsed'] = pd.to_datetime(df_base['Date'], errors='coerce')
    df_base['Year'] = df_base['Date_parsed'].dt.year
    df_base['Month'] = df_base['Date_parsed'].dt.month
    df_base['YM'] = (df_base['Year'].astype('Int64').astype(str)
                     + "-" + df_base['Month'].astype('Int64').astype(str))

    ys = sorted(df_base['Year'].dropna().astype(int).unique())
    yc = st.selectbox("Vali aasta", ["Kõik"]+[str(y) for y in ys], key="year2")

    month_opts = (list(range(1,13)) if yc=="Kõik"
                  else sorted(df_base[df_base['Year']==int(yc)]['Month']
                               .dropna().astype(int).unique()))
    mc = st.multiselect("Vali kuu", ["Kõik"]+[str(m) for m in month_opts],
                        default=["Kõik"], key="month2")

    dft = df_base.copy()
    if yc!="Kõik":
        dft = dft[dft['Year']==int(yc)]
    if "Kõik" not in mc:
        sel = [int(m) for m in mc]
        dft = dft[dft['Month'].isin(sel)]

    st.write(f"Otsingu tulemusi – **{len(dft)}** vastet")
    dg = dft.dropna(subset=['YM']).copy()
    gc = dg.groupby("YM")['Title'].count().reset_index().rename(columns={'YM':'Time','Title':'Count'})

    if pub1=="Kõik":
        rec=[]
        for time,count in gc.values:
            sub = dft[dft['YM']==time]
            rec.append((time,"ERR", sub[sub['Publication Title']=="ERR"].shape[0]))
            rec.append((time,"EPL", sub[sub['Publication Title']=="EPL"].shape[0]))
        df_tp = pd.DataFrame(rec,columns=["Time","Publication Title","Count"])
        fig = px.bar(
          df_tp,x="Time",y="Count",color="Publication Title",
          color_discrete_map={'ERR':'#003366','EPL':'#4a90e2'},
          labels={'Time':'Aeg','Count':'Arvudes'},
          title="Aja‑jaotus (ERR vs EPL)"
        )
        fmt="%{y:d}"
        fig.update_traces(texttemplate=fmt, textposition="outside", textfont_color="#333333")
        fig.update_layout(
          template="plotly_white", paper_bgcolor='white', plot_bgcolor='white',
          xaxis=dict(gridcolor="lightgray", title_font_color="#333333", tickfont_color="#333333"),
          yaxis=dict(gridcolor="lightgray", title_font_color="#333333", tickfont_color="#333333"),
          legend=dict(font_color="#333333")
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":True})
    else:
        show_time_chart(gc, "Aja‑jaotus")

    # --- 3) Vali peamine märksõna + Lisa märksõnad ---
    st.subheader("Vali peamine märksõna")
    pub3 = st.radio("Väljaanne:", ["Kõik","EPL","ERR"], key="pub3")
    df3 = df if pub3=="Kõik" else df[df["Publication Title"]==pub3]

    all_tags = sorted({t for lst in df['ManualTagsList'] for t in lst})
    tags_pub = sorted({t for lst in df3['ManualTagsList'] for t in lst})
    tags_list = tags_pub.copy()
    if 'pt' in st.session_state and st.session_state.pt not in tags_list and st.session_state.pt in all_tags:
        tags_list.insert(0, st.session_state.pt)

    options = ["(Vali)"] + tags_list
    default_idx = 0
    if 'pt' in st.session_state and st.session_state.pt in options:
        default_idx = options.index(st.session_state.pt)
    pt = st.selectbox("Märksõnad", options, index=default_idx, key="pt")

    if pt!="(Vali)":
        dp = df3[df3['ManualTagsList'].apply(lambda lst: pt in lst)]
        st.write(f"Valitud märksõna: **{pt}** ({len(dp)} kirjet)")
        st.dataframe(dp.head(20))

        # üksteise peal ERR vs EPL
        if pub3=="Kõik":
            rec2 = []
            for src in ["ERR","EPL"]:
                rec2.append((src, dp[dp["Publication Title"]==src].shape[0]))
            df_pub = pd.DataFrame(rec2, columns=["Publication Title","Count"])
            fig2 = px.bar(
              df_pub, x="Publication Title", y="Count",
              color="Publication Title",
              color_discrete_map={'ERR':'#003366','EPL':'#4a90e2'},
              labels={'Count':'Arvudes'},
              title=f"{pt} (ERR vs EPL)"
            )
            fmt="%{y:d}"
            fig2.update_traces(texttemplate=fmt, textposition="outside", textfont_color="#333333")
            fig2.update_layout(
              template="plotly_white", paper_bgcolor='white', plot_bgcolor='white',
              xaxis=dict(gridcolor="lightgray", title_font_color="#333333", tickfont_color="#333333"),
              yaxis=dict(gridcolor="lightgray", title_font_color="#333333", tickfont_color="#333333"),
              legend=dict(font_color="#333333")
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":True})
        else:
            ca = dp['Author'].value_counts().reset_index().rename(columns={'index':'Tag','Author':'Count'})
            show_chart_single(ca, pt)

        st.subheader("Lisa märksõnad")
        rel = sorted({t for lst in dp['ManualTagsList'] for t in lst if t!=pt})
        ms = st.multiselect("Märksõnad", options=rel, default=st.session_state.get('ms', []), key="ms")

        if ms:
            df_sub = distribution_for_subtags(dp, ms)
            show_chart_single(df_sub, f"{pt} + {ms}")

            # iga alam‑märksõna autorite jaotus
            for mtag in ms:
                st.write(f"### Autorite jaotus märksõna “{mtag}”")
                subdf = dp[dp['ManualTagsList'].apply(lambda lst: mtag in lst)]
                st.write(f"{mtag}: **{len(subdf)}** kirjet")
                st.dataframe(subdf.head(15))

                if pub3=="Kõik":
                    rec3 = []
                    for auth in subdf['Author'].dropna().unique():
                        for src in ["ERR","EPL"]:
                            cnt = subdf[
                                (subdf["Author"]==auth) &
                                (subdf["Publication Title"]==src)
                            ].shape[0]
                            rec3.append((auth, src, cnt))
                    df3_pub = pd.DataFrame(rec3, columns=["Author","Publication Title","Count"])
                    fig3 = px.bar(
                      df3_pub, x="Author", y="Count", color="Publication Title",
                      color_discrete_map={'ERR':'#003366','EPL':'#4a90e2'},
                      labels={'Count':'Arvudes'},
                      title=f"{pt}, {mtag} (ERR vs EPL)"
                    )
                    fmt="%{y:d}"
                    fig3.update_traces(texttemplate=fmt, textposition="outside", textfont_color="#333333")
                    fig3.update_layout(
                      template="plotly_white", paper_bgcolor='white', plot_bgcolor='white',
                      xaxis=dict(gridcolor="lightgray", title_font_color="#333333", tickfont_color="#333333"),
                      yaxis=dict(gridcolor="lightgray", title_font_color="#333333", tickfont_color="#333333"),
                      legend=dict(font_color="#333333")
                    )
                    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":True})
                else:
                    c2 = subdf['Author'].value_counts().reset_index().rename(columns={'index':'Tag','Author':'Count'})
                    show_chart_single(c2, f"{pt}, {mtag}")
        else:
            st.info("Vali lisa märksõnad, et näha jaotusi.")
    else:
        st.info("Palun vali peamine märksõna.")

if __name__ == "__main__":
    main()
