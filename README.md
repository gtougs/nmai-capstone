# nmai-capstone

Neuromarketing capstone. **Stage 1:** a GUI playground for Meta's [TRIBE v2](https://github.com/facebookresearch/tribev2) brain-encoding model. You upload an ad (video, audio or copy) and explore the *predicted* fMRI response of an average brain over time.

## Run it

| Where | How |
|---|---|
| **Google Colab (GPU)** | Open `colab_launcher.ipynb` in Colab, set the GPU runtime and the `HF_TOKEN` secret, then run all cells and open the printed link. |
| **Your own GPU machine** | Use Python 3.11 or newer. Run `pip install -r requirements.txt uv "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2"`, then `huggingface-cli login`, then `streamlit run app.py` |
| **Any laptop (UI only)** | Run `pip install -r requirements.txt` and then `streamlit run app.py`. Pick the **Mock (no GPU)** encoder, which gives synthetic data for UI work. |

Every input type uses Llama 3.2 3B text features, because speech in video and audio is transcribed. Until your access to the gated [`meta-llama/Llama-3.2-3B`](https://huggingface.co/meta-llama/Llama-3.2-3B) is approved, choose **Text feature model → Ungated mirror (unsloth)** in the sidebar. It is the same model architecture, but the weights have not been verified byte-identical, and the Llama 3.2 Community License still applies.

## Layout

```
app.py              Streamlit UI (thin: widgets + layout only)
nmai/brain.py       Domain layer
  Stimulus            content-addressed uploaded asset (video/audio/text)
  BrainEncoder        abstract strategy -> TribeEncoder (real) | MockEncoder (no GPU)
  BrainResponse       (timesteps x 20,484 fsaverage5 vertices) predictions + times
  CorticalAtlas       Destrieux parcellation -> per-region timecourses
  BrainPlotter        interactive Plotly 3D inflated cortex
colab_launcher.ipynb  GPU + public URL via Cloudflare quick tunnel
```

## Caveats

- TRIBE v2 code and weights are **CC-BY-NC-4.0** (non-commercial only).
- Predictions are for an *average subject*, not a real measurement. They come out shifted about 5 s to account for hemodynamic lag.
