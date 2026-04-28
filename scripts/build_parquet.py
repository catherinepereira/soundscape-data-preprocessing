import argparse
from pathlib import Path
import pandas as pd

_repo   = Path(__file__).resolve().parents[1]
RAW_DIR = _repo.parent / "fma-data"
OUT_DIR = _repo / "data"


def load_tracks(meta_dir: Path) -> pd.DataFrame:
    tracks = pd.read_csv(meta_dir / "tracks.csv", index_col=0, header=[0, 1])
    tracks.columns = ["_".join(c).strip() for c in tracks.columns]
    keep = [c for c in [
        "track_title", "track_genres", "track_duration", "track_listens",
        "artist_name", "album_title", "set_subset", "track_date_created",
    ] if c in tracks.columns]
    df = tracks[keep].copy()
    df.index.name = "track_id"

    raw_path = meta_dir / "raw_tracks.csv"
    if raw_path.exists():
        raw = pd.read_csv(raw_path, usecols=["track_id", "track_url"])
        raw["stream_slug"] = raw["track_url"].str.rstrip("/").str.split("/").str[-1]
        raw = raw.set_index("track_id")[["stream_slug"]]
        df = df.join(raw, how="left")
        df["audio_url"] = df["stream_slug"].apply(
            lambda s: f"https://freemusicarchive.org/track/{s}/stream/" if pd.notna(s) else None
        )
        df.drop(columns=["stream_slug"], inplace=True)
        print(f"  {df['audio_url'].notna().sum()} tracks have stream URLs")
    else:
        print("  raw_tracks.csv not found, skipping stream URLs")
        df["audio_url"] = None

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fma-meta", default=str(RAW_DIR / "fma_metadata"))
    parser.add_argument("--out",      default=str(OUT_DIR / "tracks.parquet"))
    args = parser.parse_args()

    meta_dir = Path(args.fma_meta)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Loading tracks...")
    tracks = load_tracks(meta_dir)

    if "set_subset" in tracks.columns:
        tracks = tracks[tracks["set_subset"].isin(["small", "medium"])]
        print(f"  {len(tracks)} tracks in medium subset")

    tracks.to_parquet(out)
    print(f"Saved {len(tracks)} tracks to {out}")


if __name__ == "__main__":
    main()
