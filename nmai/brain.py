"""Domain layer: stimuli, brain encoders, predicted brain responses, and their visualisation.

Everything model-specific lives behind ``BrainEncoder`` so the UI (and future features)
never import TRIBE v2 directly. ``MockEncoder`` lets the GUI run on machines without a GPU.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

HEMI_VERTICES = 10_242  # fsaverage5 vertices per hemisphere; TRIBE v2 predicts both (20,484)


class Modality(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"

    @property
    def suffixes(self) -> list[str]:  # mirrors tribev2.demo_utils.VALID_SUFFIXES
        return {"video": ["mp4", "avi", "mkv", "mov", "webm"],
                "audio": ["wav", "mp3", "flac", "ogg"],
                "text": ["txt"]}[self.value]


@dataclass(frozen=True)
class Stimulus:
    """A marketing asset on disk. Content-addressed so identical uploads share caches."""
    path: Path
    modality: Modality

    @classmethod
    def from_bytes(cls, data: bytes, suffix: str, modality: Modality, workdir: str | Path) -> Stimulus:
        path = Path(workdir) / f"{hashlib.sha256(data).hexdigest()[:16]}.{suffix.lstrip('.').lower()}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return cls(path, modality)

    @classmethod
    def from_text(cls, text: str, workdir: str | Path) -> Stimulus:
        return cls.from_bytes(text.encode("utf-8"), "txt", Modality.TEXT, workdir)


@dataclass(frozen=True)
class BrainResponse:
    """Predicted cortical activity: ``preds`` is (timesteps, vertices); ``times`` is seconds per row."""
    preds: np.ndarray
    times: np.ndarray

    def __post_init__(self):
        if self.preds.ndim != 2 or self.preds.shape[1] != 2 * HEMI_VERTICES:
            raise ValueError(f"Expected (T, {2 * HEMI_VERTICES}) predictions, got {self.preds.shape}")
        if len(self.times) != len(self.preds):
            raise ValueError("times and preds must have the same length")

    def __len__(self) -> int:
        return len(self.preds)

    @property
    def global_signal(self) -> pd.Series:
        """Mean predicted activation across the cortex at each timestep."""
        return pd.Series(self.preds.mean(axis=1), index=self.times, name="Whole cortex")

    def to_npz(self) -> bytes:
        import io
        buf = io.BytesIO()
        np.savez_compressed(buf, preds=self.preds, times=self.times)
        return buf.getvalue()


# --------------------------------------------------------------------------- encoders
class BrainEncoder(ABC):
    """Strategy interface: turn a stimulus into a predicted brain response."""

    @abstractmethod
    def predict(self, stimulus: Stimulus) -> BrainResponse: ...


class TribeEncoder(BrainEncoder):
    """Meta's TRIBE v2 (CC-BY-NC-4.0). Needs a GPU; text input needs gated meta-llama/Llama-3.2-3B access."""

    def __init__(self, repo_id: str = "facebook/tribev2", cache_folder: str = "./cache", device: str = "auto"):
        from tribev2 import TribeModel  # heavy import, deferred so the mock path stays light
        self._model = TribeModel.from_pretrained(repo_id, cache_folder=cache_folder, device=device)

    def predict(self, stimulus: Stimulus) -> BrainResponse:
        events = self._model.get_events_dataframe(**{f"{stimulus.modality.value}_path": str(stimulus.path)})
        preds, segments = self._model.predict(events=events, verbose=False)
        tr = float(self._model.data.TR)
        times = np.array([getattr(s, "start", i * tr) for i, s in enumerate(segments)], dtype=float)
        return BrainResponse(np.asarray(preds, dtype=np.float32), times)


class MockEncoder(BrainEncoder):
    """Deterministic, spatially smooth fake responses for GPU-free UI development. Not real science."""

    def __init__(self, n_timesteps: int = 30, tr: float = 1.0):
        self.n_timesteps, self.tr = n_timesteps, tr

    def predict(self, stimulus: Stimulus) -> BrainResponse:
        rng = np.random.default_rng(int(hashlib.sha256(stimulus.path.stem.encode()).hexdigest()[:8], 16))
        coords = BrainPlotter.mesh()[0]
        centers = coords[rng.choice(len(coords), 6, replace=False)]
        blobs = np.exp(-((coords[:, None] - centers[None]) ** 2).sum(-1) / (2 * 15.0 ** 2))  # (V, 6)
        t = np.arange(self.n_timesteps)
        drive = np.sin(t[:, None] / rng.uniform(2, 6, 6) + rng.uniform(0, 6, 6))  # (T, 6)
        preds = drive @ blobs.T + rng.normal(0, 0.05, (self.n_timesteps, len(coords)))
        return BrainResponse(preds.astype(np.float32), t * self.tr)


ENCODERS: dict[str, type[BrainEncoder]] = {"TRIBE v2": TribeEncoder, "Mock (no GPU)": MockEncoder}


# --------------------------------------------------------------------------- analysis & rendering
class CorticalAtlas:
    """Destrieux anatomical parcellation on fsaverage5, used to summarise vertices into named regions."""

    def __init__(self):
        from nilearn import datasets
        atlas = datasets.fetch_atlas_surf_destrieux()
        names = np.array([str(n.decode() if isinstance(n, bytes) else n) for n in atlas.labels], dtype=object)
        self.labels = np.r_["L " + names[atlas.map_left], "R " + names[atlas.map_right]]
        self._keep = np.array([not n.endswith(("Unknown", "Medial_wall")) for n in self.labels])

    def summarize(self, response: BrainResponse) -> pd.DataFrame:
        """(timesteps x regions) mean activation per region."""
        df = pd.DataFrame(response.preds[:, self._keep].T, index=self.labels[self._keep])
        out = df.groupby(level=0).mean().T
        out.index = response.times
        return out

    def top_regions(self, response: BrainResponse, n: int = 15) -> pd.Series:
        return self.summarize(response).mean().nlargest(n)


class BrainPlotter:
    """Interactive 3D inflated cortex (Plotly). Works headless, unlike TRIBE's PyVista backend."""
    _MESH: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    CAMERAS = {"Lateral": dict(x=-2, y=0, z=0.4), "Medial": dict(x=0.01, y=0, z=2.2),
               "Front": dict(x=0, y=2, z=0.3), "Back": dict(x=0, y=-2, z=0.3)}

    @classmethod
    def mesh(cls, gap: float = 8.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(coords, faces, sulcal depth) for both inflated hemispheres, loaded once per process."""
        if cls._MESH is None:
            from nilearn import datasets, surface
            fs = datasets.fetch_surf_fsaverage("fsaverage5")
            coords, faces, sulc = [], [], []
            for hemi in ("left", "right"):
                m = surface.load_surf_mesh(fs[f"infl_{hemi}"])
                c = np.array(m.coordinates, dtype=float)
                c[:, 0] += -c[:, 0].max() - gap / 2 if hemi == "left" else -c[:, 0].min() + gap / 2
                faces.append(np.asarray(m.faces) + sum(len(x) for x in coords))
                coords.append(c)
                sulc.append(surface.load_surf_data(fs[f"sulc_{hemi}"]))
            cls._MESH = np.vstack(coords), np.vstack(faces), np.concatenate(sulc)
        return cls._MESH

    def figure(self, values: np.ndarray, vmax: float, threshold: float = 0.0, view: str = "Lateral"):
        import plotly.graph_objects as go
        from matplotlib import colormaps
        coords, faces, sulc = self.mesh()
        rgb = colormaps["RdBu_r"](np.clip((values / vmax + 1) / 2, 0, 1))[:, :3]
        grey = np.where(sulc > 0, 0.55, 0.8)[:, None].repeat(3, 1)  # sulcal background shading
        rgb = np.where((np.abs(values) >= threshold * vmax)[:, None], rgb, grey)
        colors = [f"rgb({r},{g},{b})" for r, g, b in (rgb * 255).astype(int)]
        fig = go.Figure(go.Mesh3d(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                                  i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], vertexcolor=colors,
                                  lighting=dict(ambient=0.6, diffuse=0.6, specular=0.1), hoverinfo="skip"))
        axis = dict(visible=False)
        fig.update_layout(scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data",
                                     camera=dict(eye=self.CAMERAS[view])),
                          margin=dict(l=0, r=0, t=0, b=0), height=480, uirevision="brain")
        return fig
