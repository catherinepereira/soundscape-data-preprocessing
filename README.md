# Embeddings Data Preprocessing

Scripts to build the `viz.json` file used by the [embeddings-playground](https://github.com/catherinepereira/embeddings-playground) visualization.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline

### 1. `scripts/build_parquet.py`

Parses the raw FMA metadata CSVs into `tracks.parquet`

### 2. `scripts/build_embeddings.py`

Runs each track through the CLAP model (`laion/larger_clap_music_and_speech`) and saves embeddings in 512D. Selects two 20 second clips per track and averages the embeddings to get a more representative vector for the full audio track

Outputs:

- `data/embeddings.npz`: full-precision embeddings + ids
- `data/vectors.bin`: flat row-major vector blob for the web frontend, float16 by default (`--vector-format float32` for full precision)
- `data/vectors.meta.json`: `{dim, dtype, count, ids}` so the frontend maps track id to row

### 3. `scripts/build_viz_data.py`

Reduces CLAP embeddings to 2D and 3D with UMAP, maps genres to colors, writes `data/viz.json`, and saves the UMAP models to `data/`

### 4. Copy to the frontend

The frontend serves these from its `public/` directory:

```bash
cp data/viz.json data/vectors.bin data/vectors.meta.json ../soundscape-frontend/public/
```
