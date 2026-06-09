import argparse
import ast
import json
import pickle
import re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import umap


def fma_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_")

_repo   = Path(__file__).resolve().parents[1]
RAW_DIR = _repo.parent / "fma-data"
OUT_DIR = _repo / "data"

PALETTE = [
    "#8B5CF6", "#7A9E7E", "#C45D3E", "#D48B5A", "#4A90D9",
    "#F0C040", "#E87D9A", "#5B8DB8", "#A0785A", "#D45555",
    "#6BBFBF", "#D47AB0", "#8B7355", "#70A870", "#9E9E7A",
]
OTHER_COLOR = "#B8A08A"

def load_genre_map(meta_dir: Path) -> tuple[dict, dict]:
    df = pd.read_csv(meta_dir / "genres.csv")
    titles  = dict(zip(df["genre_id"].astype(int), df["title"]))
    parents = dict(zip(df["genre_id"].astype(int), df["parent"].astype(int)))
    return titles, parents


def top_level_genre(gid: int, titles: dict, parents: dict) -> str:
    seen = set()
    while gid and gid not in seen:
        seen.add(gid)
        if parents.get(gid, 0) == 0:
            return titles.get(gid, "Other")
        gid = parents[gid]
    return "Other"


def resolve_genre(raw: str | None, titles: dict, parents: dict) -> str:
    if not raw:
        return "Other"
    try:
        ids = ast.literal_eval(str(raw))
        if isinstance(ids, list) and ids:
            top = top_level_genre(int(ids[0]), titles, parents)
            return top
    except (ValueError, SyntaxError):
        pass
    return "Other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings",   default=str(OUT_DIR / "embeddings.npz"))
    parser.add_argument("--tracks",       default=str(OUT_DIR / "tracks.parquet"))
    parser.add_argument("--fma-meta",     default=str(RAW_DIR / "fma_metadata"))
    parser.add_argument("--out",          default=str(OUT_DIR / "viz.json"))
    parser.add_argument("--n-neighbors",  type=int,   default=15)
    parser.add_argument("--min-dist",     type=float, default=0.1)
    parser.add_argument("--random-state", type=int,   default=42)
    args = parser.parse_args()

    print("Loading embeddings...")
    data = np.load(args.embeddings)
    ids: np.ndarray        = data["ids"]
    embeddings: np.ndarray = data["embeddings"].astype(np.float32)
    print(f"  {embeddings.shape[0]} tracks, dim={embeddings.shape[1]}")

    print("Loading metadata...")
    tracks = pd.read_parquet(args.tracks)
    genre_titles, genre_parents = load_genre_map(Path(args.fma_meta))

    umap_kwargs = dict(n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                       metric="cosine", random_state=args.random_state, verbose=True)

    print("Running UMAP 2D...")
    reducer2d = umap.UMAP(n_components=2, **umap_kwargs)
    coords2d  = reducer2d.fit_transform(embeddings)

    print("Running UMAP 3D...")
    reducer3d = umap.UMAP(n_components=3, **umap_kwargs)
    coords3d  = reducer3d.fit_transform(embeddings)

    print("Resolving genres...")
    genres = []
    for tid in ids.tolist():
        row = tracks.loc[int(tid)] if int(tid) in tracks.index else None
        raw = str(row["track_genres"]) if row is not None and pd.notna(row.get("track_genres")) else None
        genres.append(resolve_genre(raw, genre_titles, genre_parents))

    counts = dict(Counter(genres))
    palette = list(PALETTE)
    genre_colors = {"Other": OTHER_COLOR}
    for g, _ in sorted(counts.items(), key=lambda x: -x[1]):
        if g != "Other":
            genre_colors[g] = palette.pop(0) if palette else OTHER_COLOR

    print("  Genres:")
    for g, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {g:25s} {n}")

    print("Building output...")
    points = []
    for i, raw_tid in enumerate(ids.tolist()):
        tid = int(raw_tid)
        row = tracks.loc[tid] if tid in tracks.index else None

        audio_url = str(row["audio_url"]) if row is not None and pd.notna(row.get("audio_url")) else None

        track_url = None
        if audio_url and row is not None and pd.notna(row.get("artist_name")) and pd.notna(row.get("album_title")):
            parts = audio_url.split("/track/")
            if len(parts) == 2:
                slug = parts[1].split("/")[0]
                track_url = "https://freemusicarchive.org/music/{}/{}/{}/".format(
                    fma_slug(str(row["artist_name"])),
                    fma_slug(str(row["album_title"])),
                    slug,
                )

        artist_url = "https://freemusicarchive.org/music/{}/".format(
            fma_slug(str(row["artist_name"]))
        ) if row is not None and pd.notna(row.get("artist_name")) else None

        points.append({
            "id":        tid,
            "x":         round(float(coords3d[i, 0]), 4),
            "y":         round(float(coords3d[i, 1]), 4),
            "z":         round(float(coords3d[i, 2]), 4),
            "x2":        round(float(coords2d[i, 0]), 4),
            "y2":        round(float(coords2d[i, 1]), 4),
            "title":     str(row["track_title"]) if row is not None and pd.notna(row.get("track_title")) else None,
            "artist":    str(row["artist_name"]) if row is not None and pd.notna(row.get("artist_name")) else None,
            "genre":     genres[i],
            "color":     genre_colors.get(genres[i], OTHER_COLOR),
            "audio_url": audio_url,
            "track_url": track_url,
            "artist_url": artist_url,
        })

    out = {
        "points": points,
        "genre_colors": genre_colors,
        "meta": {
            "n_tracks": len(points),
            "umap_n_neighbors": args.n_neighbors,
            "umap_min_dist": args.min_dist,
            "embedding_dim": int(embeddings.shape[1]),
            "model": "laion/larger_clap_music_and_speech",
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"Saved {len(points)} points -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    print("Building genre averages...")
    genre_arr = np.array(genres)
    genre_vectors = {}
    for g in sorted(set(genres)):
        mean = embeddings[genre_arr == g].mean(axis=0)
        mean = mean / np.linalg.norm(mean)
        genre_vectors[g] = [round(float(v), 4) for v in mean]
    gv_path = out_path.parent / "genre_vectors.json"
    gv_path.write_text(json.dumps(
        {"dim": int(embeddings.shape[1]), "genres": genre_vectors},
        separators=(",", ":"),
    ))
    print(f"Saved {len(genre_vectors)} genre averages -> {gv_path}")

    for name, model in [("umap_model.pkl", reducer3d), ("umap_model_2d.pkl", reducer2d)]:
        path = out_path.parent / name
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved UMAP model to {path}")


if __name__ == "__main__":
    main()
