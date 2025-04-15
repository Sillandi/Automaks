import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter

###############################################################################
# 1) ABI-FUNKTSIOONID
###############################################################################

def unique_preserve_order(tags):
    """
    Eemaldab duplikaadid, säilitades esimese esinemise järjekorra.
    """
    seen = []
    for t in tags:
        if t not in seen:
            seen.append(t)
    return seen

def distribution_for_subtags(dff, chosen_tags):
    """
    Koonddiagramm: liigitame iga rea (väljaande) esimese leitud lisa märksõna alla,
    muidu 'Muu'. Tagastab DataFrame ['Tag', 'Count'].
    """
    from collections import Counter
    labels = []
    for _, row in dff.iterrows():
        row_tags = row['ManualTagsList']
        label = "Muu"
        for ctag in chosen_tags:
            if ctag in row_tags:
                label = ctag
                break
        labels.append(label)
    cnt = Counter(labels)
    df_out = pd.DataFrame(list(cnt.items()), columns=["Tag", "Count"])
    return df_out

def show_chart_single(df_counts, chart_title):
    """
    Kuvab diagrame (tulp, sektor, mõlemad).
    Kasutaja valib 'Arv' või 'Protsent'.
    Tulpdiagrammil peidame 'Muu' kategooria, sektordiagrammil näitame ka 'Muu'.
    """
    measure_option = st.radio("Kuva andmed:", ["Arv", "Protsent"], key=chart_title + "_measure")
    df_plot = df_counts.copy()

    total = df_plot['Count'].sum()
    if measure_option == "Protsent" and total > 0:
        df_plot['Count'] = df_plot['Count'] / total * 100

    diag_option = st.selectbox(
        "Vali diagrammi tüüp:",
        ["Tulpdiagramm", "Sektordiagramm", "Mõlemad"],
        key=chart_title + "_diag"
    )

    config = {"displayModeBar": True, "modeBarButtonsToAdd": ["toggleFullscreen"]}
    color_sequence = ["#5b5b5b"]  # ühevärviline hall toon

    # Tulpdiagramm (peidame 'Muu')
    if diag_option in ["Tulpdiagramm", "Mõlemad"]:
        df_bar = df_plot[df_plot['Tag'] != "Muu"].copy()
        y_label = "Protsent" if measure_option == "Protsent" else "Arv"
        fig_bar = px.bar(
            df_bar,
            x='Tag', y='Count',
            title=f"{chart_title} – Tulpdiagramm",
            labels={'Count': y_label},
            color_discrete_sequence=color_sequence
        )
        st.plotly_chart(fig_bar, use_container_width=True, config=config)

    # Sektordiagramm (s.h. 'Muu')
    if diag_option in ["Sektordiagramm", "Mõlemad"]:
        y_label = "Protsent" if measure_option == "Protsent" else "Arv"
        fig_pie = px.pie(
            df_plot,
            names='Tag', values='Count',
            title=f"{chart_title} – Sektordiagramm",
            labels={'Count': y_label},
            color_discrete_sequence=color_sequence
        )
        st.plotly_chart(fig_pie, use_container_width=True, config=config)

def show_time_chart(df_count, chart_title):
    """
    Kuvab ajajaotuse (nt Year-Month) diagrammi (tulp/sektor/mõlemad).
    Kasutaja valib Arv vs Protsent.
    df_count peab sisaldama veerge ['Time','Count'].
    """
    measure_option = st.radio("Kuva andmed:", ["Arv", "Protsent"], key=chart_title + "_time_meas")
    df_plot = df_count.copy()

    total = df_plot['Count'].sum()
    if measure_option == "Protsent" and total > 0:
        df_plot['Count'] = df_plot['Count'] / total * 100

    diag_option = st.selectbox("Vali diagrammi tüüp:",
                               ["Tulpdiagramm", "Sektordiagramm", "Mõlemad"],
                               key=chart_title + "_time_diag")

    config = {"displayModeBar": True, "modeBarButtonsToAdd": ["toggleFullscreen"]}
    color_seq = ["#5b5b5b"]

    if diag_option in ["Tulpdiagramm", "Mõlemad"]:
        y_label = "Protsent" if measure_option == "Protsent" else "Arv"
        fig_bar = px.bar(
            df_plot,
            x='Time', y='Count',
            title=f"{chart_title} – Tulpdiagramm",
            labels={'Count': y_label, 'Time': 'Aeg'},
            color_discrete_sequence=color_seq
        )
        st.plotly_chart(fig_bar, use_container_width=True, config=config)

    if diag_option in ["Sektordiagramm", "Mõlemad"]:
        y_label = "Protsent" if measure_option == "Protsent" else "Arv"
        fig_pie = px.pie(
            df_plot,
            names='Time', values='Count',
            title=f"{chart_title} – Sektordiagramm",
            labels={'Count': y_label},
            color_discrete_sequence=color_seq
        )
        st.plotly_chart(fig_pie, use_container_width=True, config=config)

###############################################################################
# 2) RAKENDUSE ALGUS: PEALKIRI, SISSEJUHATUS, FILE UPLOADER
###############################################################################

def main():
    st.title("AUTOMAKSU KAJASTUSTE TONAALSUSE JA KÕNEISIKUTE ANALÜÜS EESTI PÄEVALEHE JA EESTI RAHVUSRINGHÄÄLINGU VEEBIUUDISTE PORTAALI NÄITEL")

    st.markdown("""
    Violeta Osula  
    BFM MA Nüüdismeedia meediauuringud  
    15.04.2025  

    Andmed pärinevad ERRi ja EPLi uudiste veebiportaalist, mis on avalikustatud ajavahemikus 19.07.23 – 16.08.24
    """)

    # Kasutaja laeb CSV-faili
    uploaded_file = st.file_uploader("Lae üles CSV-fail", type=["csv"])
    if uploaded_file is None:
        st.write("Palun lae CSV-fail, et jätkata...")
        return

    # Kui fail on laetud, loeme andmed
    df_original = pd.read_csv(uploaded_file, encoding="utf-8")

    # Valime veerud, kui tahad jätta ainult teatud veerud
    wanted_cols = [
        "Item Type", "Publication Year", "Author", "Title",
        "Publication Title", "Url", "Abstract Note", "Date",
        "Manual Tags", "Editor"
    ]
    cols_in_csv = [c for c in wanted_cols if c in df_original.columns]
    df = df_original[cols_in_csv].copy()

    st.write(f"Kokku on sisestatud {len(df)} väljaannet.")
    st.dataframe(df.head(50))

    # Lähme "ManualTagsList" teele
    df['Manual Tags'] = df['Manual Tags'].fillna('')
    df['ManualTagsList'] = df['Manual Tags'].apply(
        lambda x: unique_preserve_order(tag.strip() for tag in x.split(';') if tag.strip())
    )

    # Otsi sõna
    st.subheader("Otsi sõna")
    search_text = st.text_input("Sisesta otsitav sõna:")
    if search_text:
        search_cols = df.select_dtypes(include=[object]).columns
        search_text_l = search_text.lower()

        def row_matches(row):
            for c in search_cols:
                val = str(row[c]).lower()
                if search_text_l in val:
                    return True
            return False

        dff_search = df[df.apply(row_matches, axis=1)]
        st.write(f"Leidsime {len(dff_search)} väljaannet, kus sõna **{search_text}** esines.")
        st.dataframe(dff_search.head(30))

        # Diagramm "Leitud" vs "Muu"
        match_count = len(dff_search)
        no_match = len(df) - match_count
        data_search = pd.DataFrame([
            ("Leitud", match_count),
            ("Muu", no_match)
        ], columns=["Tag", "Count"])
        show_chart_single(data_search, f"Otsi sõna: '{search_text}'")
    else:
        st.write("Sisesta ülal tekst, et otsida väljaannetest. Kuvatakse tulemused ja diagramm.")

    # Kuupõhine otsing
    st.subheader("Kuupõhine otsing")
    # Püüame kuupäevaks parssida
    df['Date_parsed'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date_parsed'].dt.year
    df['Month'] = df['Date_parsed'].dt.month

    years_sorted = sorted(set(df['Year'].dropna().astype(int)))
    months_sorted = [i for i in range(1, 13)]

    year_choice = st.selectbox("Vali aasta", options=["Vali kõik"] + [str(y) for y in years_sorted])
    month_choice = st.selectbox("Vali kuu", options=["Vali kõik"] + [str(m) for m in months_sorted])

    if year_choice == "Vali kõik" and month_choice == "Vali kõik":
        # (Year, Month) group
        dff_group = df.dropna(subset=['Year','Month']).copy()
        dff_group['YearMonth'] = dff_group['Year'].astype(int).astype(str) + "-" + dff_group['Month'].astype(int).astype(str)
        group_count = dff_group.groupby("YearMonth")['Title'].count().reset_index()
        group_count.columns = ["Time","Count"]
        st.write("Kokku: ", len(dff_group), " väljaannet, group by Year-Month")
        show_time_chart(group_count, "Kõik aastad ja kuud")

    elif year_choice != "Vali kõik" and month_choice != "Vali kõik":
        # konkreetne aasta, konkreetne kuu
        sel_year = int(year_choice)
        sel_month = int(month_choice)
        dff_y_m = df[(df['Year'] == sel_year) & (df['Month'] == sel_month)]
        st.write(f"Aasta = {year_choice}, kuu = {month_choice}, leidsime {len(dff_y_m)} väljaannet")
        time_label = f"{sel_year}-{sel_month}"
        data_ym = pd.DataFrame([[time_label, len(dff_y_m)]], columns=["Time","Count"])
        show_time_chart(data_ym, f"Aasta {year_choice}, kuu {month_choice}")

    elif year_choice != "Vali kõik" and month_choice == "Vali kõik":
        # konkreetne aasta, kõik kuud
        sel_year = int(year_choice)
        dff_y = df[df['Year'] == sel_year].copy()
        dff_y['Month'] = dff_y['Month'].fillna(0).astype(int)
        group_count = dff_y.groupby("Month")['Title'].count().reset_index()
        group_count['Time'] = group_count['Month'].astype(str).apply(lambda m: f"{sel_year}-{m}")
        group_count = group_count[['Time','Title']].rename(columns={'Title':'Count'})
        st.write(f"Aasta = {year_choice}, kõik kuud – {group_count['Count'].sum()} väljaannet")
        show_time_chart(group_count, f"Aasta {year_choice}, kõik kuud")

    elif year_choice == "Vali kõik" and month_choice != "Vali kõik":
        # kõik aastad, konkreetne kuu
        sel_month = int(month_choice)
        dff_m = df[df['Month'] == sel_month]
        group_count = dff_m.groupby("Year")['Title'].count().reset_index()
        group_count.columns = ["Time","Count"]
        st.write(f"Kuu = {month_choice}, kõik aastad – {group_count['Count'].sum()} väljaannet")
        show_time_chart(group_count, f"Kuu {month_choice}, kõik aastad")

    # ManualTagsList analüüs: vali peamine märksõna + lisa
    st.subheader("Vali peamine märksõna")

    all_tags_set = set()
    for rowtags in df['ManualTagsList']:
        all_tags_set.update(rowtags)
    all_tags_sorted = sorted(all_tags_set)

    primary_tag = st.selectbox("Märksõnad", ["(Vali)"] + all_tags_sorted)

    if primary_tag != "(Vali)":
        dff_primary = df[df['ManualTagsList'].apply(lambda row: primary_tag in row)]
        st.write(f"Valitud märksõna: **{primary_tag}** – leidsime {len(dff_primary)} väljaannet.")
        st.dataframe(dff_primary.head(30))

        # Näidisdiagramm (nt Autorite loendus)
        if not dff_primary.empty and 'Author' in dff_primary.columns:
            c_auth = dff_primary['Author'].value_counts().reset_index()
            c_auth.columns = ['Tag','Count']
            show_chart_single(c_auth, f"Valitud märksõna: '{primary_tag}'")

        rel_tags_set = set()
        for row_tags in dff_primary['ManualTagsList']:
            rel_tags_set.update(row_tags)
        if primary_tag in rel_tags_set:
            rel_tags_set.remove(primary_tag)
        related_tags = sorted(rel_tags_set)

        st.subheader("Lisa märksõnad")
        chosen_subtags = st.multiselect("Märksõnad", options=related_tags)
        if chosen_subtags:
            # kombineeritud jaotus
            df_dist = distribution_for_subtags(dff_primary, chosen_subtags)
            show_chart_single(df_dist, f"Valitud märksõna: '{primary_tag}' + lisa: {chosen_subtags}")

            # iga subtag eraldi
            for stg in chosen_subtags:
                dff_stg = dff_primary[dff_primary['ManualTagsList'].apply(lambda row: stg in row)]
                st.write(f"**Lisa märksõna**: {stg}, leidsime {len(dff_stg)} väljaannet.")
                st.dataframe(dff_stg.head(20))

                if not dff_stg.empty and 'Author' in dff_stg.columns:
                    c2 = dff_stg['Author'].value_counts().reset_index()
                    c2.columns = ['Tag','Count']
                    show_chart_single(c2, f"Valitud märksõna: '{primary_tag}', lisa: '{stg}'")

        else:
            st.write("Pole lisa märksõnu valitud. Kuvame ainult peamise märksõna väljaanded.")

    else:
        st.write("Vali peamine märksõna ülalt, et näha tulemusi.")


###############################################################################
# 3) KÄIVITUS
###############################################################################
if __name__ == "__main__":
    main()
