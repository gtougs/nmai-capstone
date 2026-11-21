"""TRIBE v2 playground: upload an ad (video / audio / copy) and explore predicted brain activity.

Run:  streamlit run app.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import streamlit as st

from nmai.brain import ENCODERS, BrainPlotter, BrainResponse, CorticalAtlas, Modality, Stimulus, TribeEncoder

UPLOADS = Path("./cache/uploads")
st.set_page_config(page_title="TRIBE v2 Playground", page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner="Loading model (first run downloads several GB)…")
def get_encoder(name: str, text_model: str):
    return ENCODERS[name](text_model=text_model)


@st.cache_resource
def get_atlas() -> CorticalAtlas:
    return CorticalAtlas()


@st.cache_data(show_spinner=False, max_entries=20)
def predict(encoder_name: str, text_model: str, path: str, modality: str) -> BrainResponse:
    return get_encoder(encoder_name, text_model).predict(Stimulus(Path(path), Modality(modality)))


# ----------------------------------------------------------------- sidebar: configuration
with st.sidebar:
    st.title("🧠 TRIBE v2 Playground")
    has_tribe = importlib.util.find_spec("tribev2") is not None
    encoder_name = st.selectbox("Encoder", list(ENCODERS), index=0 if has_tribe else 1,
                                help=None if has_tribe else "tribev2 is not installed here, so only the mock runs.")
    text_model = TribeEncoder.TEXT_MODELS[st.selectbox(
        "Text feature model", list(TribeEncoder.TEXT_MODELS), disabled=encoder_name != "TRIBE v2",
        help="Used for every input type (speech is transcribed). The mirror has the same Llama 3.2 3B "
             "weights without Meta's access gate; the Llama 3.2 Community License still applies.")]
    modality = Modality(st.radio("Stimulus type", [m.value for m in Modality], horizontal=True))
    threshold = st.slider("Display threshold (fraction of max)", 0.0, 0.9, 0.3, 0.05)
    view = st.selectbox("Camera", list(BrainPlotter.CAMERAS))
    st.caption("Predictions are for an *average* subject on the fsaverage5 cortex, "
               "already shifted ~5 s for hemodynamic lag. TRIBE v2 is CC-BY-NC-4.0 (non-commercial).")

# ----------------------------------------------------------------- input
if modality is Modality.TEXT:
    text = st.text_area("Ad copy / script", height=150, placeholder="Paste ad copy here…")
    stimulus = Stimulus.from_text(text, UPLOADS) if text.strip() else None
else:
    upload = st.file_uploader(f"Upload {modality.value}", type=modality.suffixes)
    stimulus = (Stimulus.from_bytes(upload.getvalue(), Path(upload.name).suffix, modality, UPLOADS)
                if upload else None)

if st.button("Predict brain response", type="primary", disabled=stimulus is None):
    with st.spinner("Extracting features and predicting… (video can take minutes on first run)"):
        try:
            st.session_state.result = (stimulus, predict(encoder_name, text_model, str(stimulus.path), modality.value))
        except Exception as exc:  # surface model/setup errors in the UI instead of a blank page
            st.exception(exc)

if "result" not in st.session_state:
    st.info("Pick a stimulus type, provide an input, then press **Predict brain response**.")
    st.stop()

# ----------------------------------------------------------------- results
stim, response = st.session_state.result
atlas = get_atlas()
vmax = float(np.percentile(np.abs(response.preds), 99)) or 1.0
st.success(f"{len(response)} timesteps × {response.preds.shape[1]:,} cortical vertices")

brain_tab, timeline_tab, regions_tab = st.tabs(["🧠 Brain", "📈 Timeline", "🗺️ Regions"])
with brain_tab:
    idx = st.slider("Timestep", 0, len(response) - 1, 0, format="%d",
                    help="Each step is one fMRI TR. Seconds shown below.")
    t = float(response.times[idx])
    left, right = st.columns([2, 1])
    left.plotly_chart(BrainPlotter().figure(response.preds[idx], vmax, threshold, view), use_container_width=True)
    with right:
        st.metric("Time", f"{t:.1f} s")
        st.metric("Whole-cortex mean", f"{response.global_signal.iloc[idx]:+.3f}")
        if stim.modality is Modality.VIDEO:
            st.video(str(stim.path), start_time=int(t))
        elif stim.modality is Modality.AUDIO:
            st.audio(str(stim.path), start_time=int(t))
        st.dataframe(atlas.summarize(response).iloc[idx].nlargest(8).rename("activation"))

with timeline_tab:
    regions = atlas.summarize(response)
    picks = st.multiselect("Regions", list(regions.columns), default=list(atlas.top_regions(response, 4).index))
    st.line_chart(regions[picks].join(response.global_signal))
    st.caption("x-axis: seconds. Peaks suggest moments of strong predicted engagement in that region.")

with regions_tab:
    top = atlas.top_regions(response, 15)
    st.bar_chart(top, horizontal=True)
    c1, c2 = st.columns(2)
    c1.download_button("Region timecourses (CSV)", regions.to_csv().encode(), "regions.csv", "text/csv")
    c2.download_button("Raw predictions (NPZ)", response.to_npz(), "predictions.npz")
