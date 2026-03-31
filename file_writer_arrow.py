import os
from pathlib import Path
import time
import pyarrow as pa
import pyarrow.ipc as ipc
from logger import logger
import torch

from pathlib import Path

def getFilePath(video_path, root_dir, feature_type):
    """
    Generate the output file path for a given video and feature type.

    New rule:
    - If the path contains 'BWVs', use the *entire* subpath after 'BWVs'
      e.g.
      /mnt/.../BWVs/first_10_videos/session1/a.mp4
      -> root_dir/first_10_videos/session1/a.<feature_type>.arrow

    - Otherwise:
      root_dir/<file_name>.<feature_type>.arrow
    """
    video_path = Path(video_path).resolve()
    root_dir = Path(root_dir).resolve()
    parts = video_path.parts

    if "BWVs" in parts:
        bwv_idx = parts.index("BWVs")
        if bwv_idx + 1 >= len(parts):
            raise ValueError(f"Path '{video_path}' has 'BWVs' but no subdirectory after it.")
        
        rel_path = Path(*parts[bwv_idx + 1:])   # e.g. first_10_videos/session1/a.mp4
    else:
        rel_path = Path(video_path.name)

    out_path = root_dir / rel_path.with_suffix(f".{feature_type}.arrow")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path

def ArrowWriterProcess(root_dir, data_queue, stop_event, batch_size=10):
    """
    Writer process:
    - Each video × feature gets its own Arrow IPC file.
    - Accumulates up to `batch_size` frames before writing.
    - Flushes remaining frames on 'video_done' or shutdown.
    """
    os.makedirs(root_dir, exist_ok=True)

    writers = {}     # (video_path, feature_type) -> (sink, writer)
    buffers = {}     # (video_path, feature_type) -> list of Arrow arrays

    while True:
        item = data_queue.get()
        if item is None:  # shutdown signal
            break

        vpath = item["video_path"]
        feature_type = item["type"]  # e.g. "video_embedding"
        key = (vpath, feature_type) # Unique key per video+feature

        if item["type"] == "speech_mels_features":
            mels = item["mels"]
            audio_shape = item["audio_shape"]

            # Convert to Arrow arrays
            mels_arr = pa.array([mels.tolist()])
            audio_shape_arr = pa.array([list(audio_shape)])

            # If first time writing this (video, feature) → open file
            if key not in writers:
                out_path = getFilePath(vpath, root_dir, feature_type)
                sink = pa.OSFile(str(out_path), "wb")
                schema = pa.schema(
                    [
                        pa.field("mels", mels_arr.type),
                        pa.field("audio_shape", audio_shape_arr.type),
                    ]
                )
                writer = ipc.new_file(sink, schema)
                writers[key] = (sink, writer)
                logger.info(f"✍️ Opened writer for {vpath}/{feature_type} → {out_path}")

            # Build a table with one row
            table = pa.table(
                {
                    "mels": mels_arr,
                    "audio_shape": audio_shape_arr,
                }
            )

            # Write immediately — no batching
            writers[key][1].write_table(table)

            logger.info(
                f"📝 Wrote speech mel spectrogram for {vpath}/{feature_type} "
                f"(shape={mels.shape}, audio_shape={audio_shape})"
            )
            
        elif item["type"] == "open_smile_features":

            # Convert to Arrow arrays
            feature_arrays = {}
            for k, v in item.items():
                if k == "audio_shape":
                    feature_arrays[k] = pa.array([list(item[k])])
                elif k == "type" or k == "video_path":
                    continue
                else:
                    logger.info(f"Prepared OpenSmile feature '{k}' for {vpath}")
                    feature_arrays[k] = pa.array([v.tolist()])  # wrap in list for single row
            # If first time writing this (video, feature) → open file
            if key not in writers:
                out_path = getFilePath(vpath, root_dir, feature_type)
                sink = pa.OSFile(str(out_path), "wb")
                
                schema = pa.schema([pa.field(k, arr.type) for k, arr in feature_arrays.items()])
                writer = ipc.new_file(sink, schema)
                writers[key] = (sink, writer)
                logger.info(f"✍️ Opened writer for {vpath}/{feature_type} → {out_path}")

            # Build a table with one row
            table = pa.table(feature_arrays)

            # Write immediately — no batching
            writers[key][1].write_table(table)

            logger.info(
                f"📝 Wrote OpenSmile features  "
            )
            for k in feature_arrays.keys():
                logger.info(
                    f"📝 Wrote {k} for {vpath}/{feature_type} "
                )

        if item["type"].endswith("_embedding"):
            embs = item["embeddings"]    # shape [num_patches, hidden_dim]
            num_patches, hidden_dim = embs.shape

            # Ensure numpy/float16
            if isinstance(embs, torch.Tensor):
                embs = embs.cpu().to(torch.float16).numpy()

            # Convert to Arrow array
            arr = pa.FixedSizeListArray.from_arrays(
                pa.array(embs.reshape(-1), type=pa.float16()),
                hidden_dim
            )

            # If first time writing this (video, feature) → open file
            if key not in writers:
                out_path = getFilePath(vpath, root_dir, feature_type)
                sink = pa.OSFile(str(out_path), "wb")
                schema = pa.schema([
                    (feature_type, pa.list_(pa.float16(), hidden_dim))
                ])
                writer = ipc.new_file(sink, schema)
                writers[key] = (sink, writer)
                buffers[key] = []
                logger.info(f"✍️ Opened writer for {vpath}/{feature_type} → {out_path}")

            # Add to buffer
            buffers[key].append(arr)

            # Flush if batch full
            if len(buffers[key]) >= batch_size:
                t0 = time.time()
                table = pa.table({feature_type: pa.concat_arrays(buffers[key])})
                writers[key][1].write_table(table)
                t1 = time.time()
                logger.info(
                    f"📝 Flushed {len(buffers[key])} batches of frame for {vpath}/{feature_type} in {t1 - t0:.3f} sec"
                )
                buffers[key] = []

        elif item["type"] == "video_done":
            vpath = item["video_path"]
            # flush and close all features for this video
            for (vid, feat), arrs in list(buffers.items()):
                if vid == vpath and arrs:
                    t0 = time.time()
                    table = pa.table({feat: pa.concat_arrays(arrs)})
                    writers[(vid, feat)][1].write_table(table)
                    t1 = time.time()
                    logger.info(
                        f"📝 Final flush {len(arrs)} batches of frame for {vpath}/{feat} in {t1 - t0:.3f} sec"
                    )
                    buffers[(vid, feat)] = []

            for (vid, feat), (sink, writer) in list(writers.items()):
                if vid == vpath:
                    # Close writer
                    writer.close()
                    sink.close()
                    del writers[(vid, feat)]
                    logger.info(f"✅ Closed writer for {vpath}/{feat}")

    # Cleanup on crash/interrupt
    for (vid, feat), (sink, writer) in list(writers.items()):
        try:
            if buffers.get((vid, feat)):
                table = pa.table({feat: pa.concat_arrays(buffers[(vid, feat)])})
                writer.write_table(table)
                logger.warning(
                    f"⚠️ Force-flushed {len(buffers[(vid, feat)])} frames for {vid}/{feat}"
                )
            writer.close()
            sink.close()
            logger.warning(f"⚠️ Force-closed unfinished writer for {vid}/{feat}")
        except Exception:
            pass
