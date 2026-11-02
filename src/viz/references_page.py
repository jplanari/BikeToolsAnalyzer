"""The References page: the literature behind the app's physics and physiology."""
import streamlit as st

from src import references as refs
from src.i18n import t, tr


def render_references():
    st.header(t("📚 References"))
    st.markdown(t("BikeTools Analyzer is built on published physics and physiology. Each model below is listed with "
                  "the work it comes from and what in the app relies on it."))
    st.caption(t("The models are simplified and the results are estimates: they depend on the quality of your sensors, "
                 "of the weather data and of the values you enter (weight, FTP, CdA, rolling resistance). Where the app "
                 "uses a rule of thumb of its own, the entry says so."))
    for topic, items in refs.by_topic():
        st.subheader(tr(topic))
        for r in items:
            st.markdown(f"- {refs.citation(r)}")
            st.caption(tr(r['used']))
