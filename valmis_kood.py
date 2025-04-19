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

def add_totals_on_bar(fig, df, xcol, ycol, colorcol):
    # Compute sums per x group
    sums = df.groupby(xcol)[ycol].sum().to_dict()
    for xi, total in sums.items():
        # place annotation above the bar
        fig.add_annotation(
            x=xi, y=total,
            text=str(total),
            showarrow=False,
            yshift=8,
            font=dict(color="#333333")
        )

# ----------------------
# 3) CHART FUNCTIONS
# ----------------------
def stacked_bar_with_totals(df, x, y, color, colors_map, title, xlab, ylab):
    fig = px.bar(
        df,
        x=x, y=y,
        color=color,
        color_discrete_map=colors_map,
        text=y,
        labels={x: xlab, y: ylab},
        title=title,
    )
    # make the bar‐segment text white, positioned inside
    fig.update_traces(textposition="inside", textfont_color="white", cliponaxis=False)
    # add the total on top
    add_totals_on_bar(fig, df, x, y, color)
    # common layout tweaks
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
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
    return fig

# A convenience for your three blocks:
def show_word_search_chart(df, terms, pub_all):
    rec = []
    for t in terms:
        for src in ["ERR","EPL"]:
            sub = df[df["Publication Title"]==src]
            cnt = sub["ManualTagsList"].apply(lambda lst: t.lower() in [x.lower() for x in lst]).sum() \
                  if t.lower() in sum([l for l in df["ManualTagsList"]], []) else \
                  sub.apply(lambda r: t.lower() in (r["Title"] or "").lower() or t.lower() in (r["Abstract Note"] or "").lower(), axis=1).sum()
            rec.append((t, src, int(cnt)))
    df_tp = pd.DataFrame(rec, columns=["Tag","Publication Title","Count"])
    fig = stacked_bar_with_totals(
        df_tp, x="Tag", y="Count", color="Publication Title",
        colors_map={"ERR":"#003366","EPL":"#4a90e2"},
        title=f'Otsi sõnu – {" & ".join(terms)} jaotus (ERR vs EPL)',
        xlab="", ylab="Arvudes"
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":True})

def show_time_chart_all(df):
    rec = []
    for t,grp in df.groupby("Time"):
        rec.append((t,"ERR", grp[grp["Publication Title"]=="ERR"].shape[0]))
        rec.append((t,"EPL", grp[grp["Publication Title"]=="EPL"].shape[0]))
    df_tp = pd.DataFrame(rec, columns=["Time","Publication Title","Count"])
    fig = stacked_bar_with_totals(
        df_tp, x="Time", y="Count", color="Publication Title",
        colors_map={"ERR":"#003366","EPL":"#4a90e2"},
        title="Aja‑jaotus (ERR vs EPL)",
        xlab="Aeg", ylab="Arvudes"
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":True})

def show_keyword_chart(dp, pt, pub_all):
    rec = [(src, dp[dp["Publication Title"]==src].shape[0]) for src in ["ERR","EPL"]]
    df_pub = pd.DataFrame(rec, columns=["Publication Title","Count"])
    fig = stacked_bar_with_totals(
        df_pub, x="Publication Title", y="Count", color="Publication Title",
        colors_map={"ERR":"#003366","EPL":"#4a90e2"},
        title=f"{pt} (ERR vs EPL)",
        xlab="", ylab="Arvudes"
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":True})

# ----------------------
# 4) MAIN
# ----------------------
def main():
    set_custom_style()
    st.title("AUTOMAKSU KAJASTUSTE … EPLi JA ERRi VEEBIUUDISTE NÄITEL")
    st.markdown("""
    _Violeta Osula • BFM MA Nüüdismeedia_  
    **15.04.2025**

    Andmed ERRi ja EPLi uudiste veebiportaalist (19.07.23 – 16.08.24)
    """)

    uploaded = st.file_uploader("Lae üles CSV‑fail", type=["csv"])
    if not uploaded:
        st.warning("Lae CSV ‑fail")
        return

    df0 = pd.read_csv(uploaded, encoding="utf-8")
    keep = ["Item Type","Publication Year","Author","Title","Publication Title",
            "Url","Abstract Note","Date","Manual Tags","Editor"]
    df = df0[[c for c in keep if c in df0.columns]].copy()
    df["Manual Tags"] = df["Manual Tags"].fillna("")
    df["ManualTagsList"] = df["Manual Tags"].apply(
        lambda x: unique_preserve_order(t.strip() for t in x.split(";") if t.strip())
    )

    # -- Block 1: Otsi sõnu --
    st.subheader("Otsi sõnu (mitme sõna puhul kasuta koma)")
    pub1 = st.radio("Väljaanne:", ["Kõik","EPL","ERR"], key="pub1")
    df1 = df if pub1=="Kõik" else df[df["Publication Title"]==pub1]

    txt = st.text_input("Sisesta sõnad ja/või märksõnad:", key="searchtxt")
    if txt:
        terms = [t.strip() for t in txt.split(",") if t.strip()]
        alltags = {t.lower() for lst in df1["ManualTagsList"] for t in lst}
        tag_terms = [t for t in terms if t.lower() in alltags]
        text_terms = [t for t in terms if t.lower() not in alltags]

        df_search = df1[
            df1["ManualTagsList"].apply(lambda L: any(T.lower()==x.lower() for x in L for T in tag_terms))
            |
            df1.apply(lambda r: any(T.lower() in (r["Title"] or "").lower() or T.lower() in (r["Abstract Note"] or "").lower() for T in text_terms), axis=1)
        ]

        # Count how many matches came from tags vs text
        tag_cnt = df_search["ManualTagsList"].apply(lambda L: any(t.lower() in [x.lower() for x in L] for t in tag_terms)).sum()
        text_cnt = df_search.shape[0] - tag_cnt
        st.markdown(" ".join(
            f"<span class='pill pill-tag'>{t}</span>" if t in tag_terms else f"<span class='pill pill-text'>{t}</span>"
            for t in terms
        ), unsafe_allow_html=True)

        st.write(f"Otsingu tulemusi – **{df_search.shape[0]}** vastet ({tag_cnt} Märksõna, {text_cnt} Tekstisõna)")
        st.dataframe(df_search.head(20), use_container_width=True)

        if pub1=="Kõik":
            show_word_search_chart(df_search, terms, pub1)
        else:
            # single‐series
            cnts = [(t,
                     df_search["ManualTagsList"].apply(lambda L: t.lower() in [x.lower() for x in L]).sum()
                     if t in tag_terms else
                     df_search.apply(lambda r: t.lower() in (r["Title"] or "").lower() or t.lower() in (r["Abstract Note"] or "").lower(), axis=1).sum()
                    )
                    for t in terms]
            df_terms = pd.DataFrame(cnts, columns=["Tag","Count"])
            fig = px.bar(
                df_terms, x="Tag", y="Count",
                labels={"Tag":"","Count":"Arvudes"},
                title="Otsi sõnu – termide jaotus"
            )
            fig.update_traces(texttemplate="%{y:d}", textposition="outside", textfont_color="#333333")
            fig.update_layout(
                template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                xaxis=dict(gridcolor="lightgray", title_font_color="#333333", tickfont_color="#333333"),
                yaxis=dict(gridcolor="lightgray", title_font_color="#333333", tickfont_color="#333333"),
                legend=dict(font_color="#333333")
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":True})
    else:
        st.info("Sisesta sõnad või märksõnad")
        df_search = df1

    # -- Block 2: Ajapõhine otsing --
    st.subheader("Ajapõhine otsing")
    df_time = df_search.copy()
    df_time["Date_parsed"] = pd.to_datetime(df_time["Date"], errors="coerce")
    df_time["Year"]  = df_time["Date_parsed"].dt.year
    df_time["Month"] = df_time["Date_parsed"].dt.month
    df_time["Time"]  = df_time["Year"].astype("Int64").astype(str) + "-" + df_time["Month"].astype("Int64").astype(str)

    years = sorted(df_time["Year"].dropna().astype(int).unique())
    sel_y = st.selectbox("Vali aasta", ["Kõik"]+list(map(str,years)), key="year2")
    df_t2 = df_time.copy()
    if sel_y!="Kõik":
        df_t2 = df_t2[df_t2["Year"]==int(sel_y)]
    months = sorted(df_t2["Month"].dropna().astype(int).unique())
    sel_m = st.multiselect("Vali kuu", ["Kõik"]+list(map(str,months)), default=["Kõik"], key="month2")
    if "Kõik" not in sel_m:
        df_t2 = df_t2[df_t2["Month"].isin(map(int,sel_m))]

    st.write(f"Otsingu tulemusi – **{df_t2.shape[0]}** vastet")
    if pub1=="Kõik":
        show_time_chart_all(df_t2)
    else:
        df_gc = (df_t2.dropna(subset=["Time"])
                 .groupby("Time")["Title"].count()
                 .reset_index(name="Count"))
        fig = px.bar(df_gc, x="Time", y="Count",
                     labels={"Time":"Aeg","Count":"Arvudes"},
                     title="Aja‑jaotus")
        fmt="%{y:d}"
        fig.update_traces(texttemplate=fmt, textposition="outside", textfont_color="#333333")
        fig.update_layout(
            template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(gridcolor="lightgray", title_font_color="#333333", tickfont_color="#333333"),
            yaxis=dict(gridcolor="lightgray", title_font_color="#333333", tickfont_color="#333333"),
            legend=dict(font_color="#333333")
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":True})

    # -- Block 3: Peamine märksõna --
    st.subheader("Vali peamine märksõna")
    pub3 = st.radio("Väljaanne:", ["Kõik","EPL","ERR"], key="pub3")
    df3 = df if pub3=="Kõik" else df[df["Publication Title"]==pub3]

    # build the tag selector
    all_tags = sorted({t for lst in df3["ManualTagsList"] for t in lst})
    prev = st.session_state.get("pt","(Vali)")
    opts = ["(Vali)"] + [t for t in all_tags]
    if prev in opts:
        idx = opts.index(prev)
    else:
        idx = 0
    pt = st.selectbox("Märksõnad", opts, index=idx, key="pt")

    if pt!="(Vali)":
        dp = df3[df3["ManualTagsList"].apply(lambda L: pt in L)]
        st.write(f"Valitud märksõna: **{pt}** ({dp.shape[0]} kirjet)")
        st.dataframe(dp.head(20), use_container_width=True)

        if pub3=="Kõik":
            show_keyword_chart(dp, pt, pub3)
        else:
            ca = dp["Author"].value_counts().reset_index().rename(columns={"index":"Tag","Author":"Count"})
            fig = px.bar(ca, x="Tag", y="Count",
                         labels={"Tag":"","Count":"Arvudes"},
                         title=pt)
            fig.update_traces(texttemplate="%{y:d}", textposition="outside", textfont_color="#333333")
            fig.update_layout(
                template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                xaxis=dict(gridcolor="lightgray", title_font_color="#333333", tickfont_color="#333333"),
                yaxis=dict(gridcolor="lightgray", title_font_color="#333333", tickfont_color="#333333"),
                legend=dict(font_color="#333333")
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":True})

        # Block 4: Lisa märksõnad
        st.subheader("Lisa märksõnad")
        ms_opts = sorted({t for lst in dp["ManualTagsList"] for t in lst if t!=pt})
        prev_ms = st.session_state.get("ms",[])
        ms = st.multiselect("Märksõnad", options=ms_opts, default=prev_ms, key="ms")
        if ms:
            subdf = dp[dp["ManualTagsList"].apply(lambda L: any(m in L for m in ms))]
            # combined stacked for those subtags
            rec = []
            for tg in ms:
                for src in ["ERR","EPL"]:
                    rec.append((
                        tg,
                        src,
                        subdf[(subdf["Publication Title"]==src) & subdf["ManualTagsList"].apply(lambda L: tg in L)].shape[0]
                    ))
            df_sm = pd.DataFrame(rec, columns=["Tag","Publication Title","Count"])
            fig = stacked_bar_with_totals(
                df_sm, x="Tag", y="Count", color="Publication Title",
                colors_map={"ERR":"#003366","EPL":"#4a90e2"},
                title=f"{pt} + {ms} (ERR vs EPL)",
                xlab="", ylab="Arvudes"
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":True})

            # AND per‑subtag author distributions, like v6
            for mtag in ms:
                st.write(f"### Autorite jaotus märksõna “{mtag}”")
                sub2 = dp[dp["ManualTagsList"].apply(lambda L: mtag in L)]
                st.write(f"{mtag}: **{sub2.shape[0]}** kirjet")
                st.dataframe(sub2.head(15), use_container_width=True)

                if pub3=="Kõik":
                    rec2=[]
                    for auth in sub2["Author"].dropna().unique():
                        for src in ["ERR","EPL"]:
                            cnt = sub2[(sub2["Publication Title"]==src)&(sub2["Author"]==auth)].shape[0]
                            rec2.append((auth, src, cnt))
                    df_auth = pd.DataFrame(rec2, columns=["Author","Publication Title","Count"])
                    fig2 = stacked_bar_with_totals(
                        df_auth, x="Author", y="Count", color="Publication Title",
                        colors_map={"ERR":"#003366","EPL":"#4a90e2"},
                        title=f"{pt}, {mtag} (ERR vs EPL)",
                        xlab="", ylab="Arvudes"
                    )
                    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":True})
                else:
                    ca2 = sub2["Author"].value_counts().reset_index().rename(columns={"index":"Tag","Author":"Count"})
                    fig2 = px.bar(ca2, x="Tag", y="Count",
                                  labels={"Tag":"","Count":"Arvudes"},
                                  title=f"{pt}, {mtag}")
                    fig2.update_traces(texttemplate="%{y:d}", textposition="outside", textfont_color="#333333")
                    fig2.update_layout(
                        template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                        xaxis=dict(gridcolor="lightgray", title_font_color="#333333", tickfont_color="#333333"),
                        yaxis=dict(gridcolor="lightgray", title_font_color="#333333", tickfont_color="#333333"),
                        legend=dict(font_color="#333333")
                    )
                    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":True})
        else:
            st.info("Vali lisa märksõnad")
    else:
        st.info("Palun vali peamine märksõna")

if __name__=="__main__":
    main()
