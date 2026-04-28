import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

_repo   = Path(__file__).resolve().parents[1]
RAW_DIR = _repo.parent / "fma-data"
OUT_DIR = _repo / "data"

CLAP_MODEL = "laion/larger_clap_music_and_speech"
SR = 48_000


def get_audio_path(audio_dir: Path, track_id: int) -> Path | None:
    tid = f"{track_id:06d}"
    p = audio_dir / tid[:3] / f"{tid}.mp3"
    return p if p.exists() else None


def load_audio(path: Path) -> list[np.ndarray] | None:
    try:
        import librosa
        duration = librosa.get_duration(path=str(path))
        clip_len = 20.0
        offset1 = min(5.0, max(0.0, duration - clip_len))
        clip1, _ = librosa.load(path, sr=SR, mono=True, offset=offset1, duration=clip_len)
        offset2 = max(offset1, duration / 2 - clip_len / 2)
        clip2, _ = librosa.load(path, sr=SR, mono=True, offset=offset2, duration=clip_len)
        return [clip1, clip2]
    except Exception:
        return None


def embed_batch(
    model: ClapModel,
    processor: ClapProcessor,
    clips: list[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    inputs = processor(audios=clips, sampling_rate=SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        embs = model.get_audio_features(**inputs)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    return embs.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", default=str(RAW_DIR / "fma_medium"))
    parser.add_argument("--tracks",    default=str(OUT_DIR / "tracks.parquet"))
    parser.add_argument("--out",       default=str(OUT_DIR / "embeddings.npz"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Embed only the first N tracks")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading CLAP model: {CLAP_MODEL}")
    processor = ClapProcessor.from_pretrained(CLAP_MODEL)
    model = ClapModel.from_pretrained(CLAP_MODEL).to(device)
    model.eval()

    tracks = pd.read_parquet(args.tracks)
    track_ids = tracks.index.tolist()
    if args.limit:
        track_ids = track_ids[:args.limit]
    print(f"Tracks to embed: {len(track_ids)}")

    all_ids: list[int] = []
    all_embeddings: list[np.ndarray] = []
    failed: list[int] = []

    batch_ids: list[int] = []
    batch_clips: list[np.ndarray] = []

    def flush() -> None:
        if not batch_clips:
            return
        embs = embed_batch(model, processor, batch_clips, device)
        clip1_embs = embs[0::2]
        clip2_embs = embs[1::2]
        avg = (clip1_embs + clip2_embs) / 2
        avg = avg / np.linalg.norm(avg, axis=-1, keepdims=True)
        all_ids.extend(batch_ids)
        all_embeddings.append(avg)
        batch_ids.clear()
        batch_clips.clear()

    for tid in tqdm(track_ids, desc="embedding"):
        path = get_audio_path(audio_dir, tid)
        if path is None:
            failed.append(tid)
            continue
        clips = load_audio(path)
        if clips is None:
            failed.append(tid)
            continue
        batch_ids.append(tid)
        batch_clips.extend(clips)
        if len(batch_ids) >= args.batch_size:
            flush()

    flush()

    if not all_embeddings:
        raise RuntimeError("No embeddings produced. Check audio paths")

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    ids = np.array(all_ids, dtype=np.int32)

    np.savez(out, ids=ids, embeddings=embeddings)
    print(f"Saved {len(ids)} embeddings to {out} with shape={embeddings.shape}")

    if failed:
        fail_path = out.with_suffix(".failed.json")
        fail_path.write_text(json.dumps(failed))
        print(f"  {len(failed)} tracks failed at {fail_path}")


if __name__ == "__main__":
    main()
