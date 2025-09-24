import os
import time
import pyarrow as pa
import pyarrow.ipc as ipc
from logger import logger
import torch

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
    hidden_dims = {} # (video_path, feature_type) -> hidden_dim

    while True:
        item = data_queue.get()
        if item is None:  # shutdown signal
            break

        if item["type"].endswith("_embedding"):
            vpath = item["video_path"]
            feature_type = item["type"]  # e.g. "video_embedding"
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

            # Unique key per video+feature
            key = (vpath, feature_type)

            # If first time writing this (video, feature) → open file
            if key not in writers:
                fname = os.path.basename(vpath)
                out_path = os.path.join(root_dir, f"{fname}.{feature_type}.arrow")
                sink = pa.OSFile(out_path, "wb")
                schema = pa.schema([
                    (feature_type, pa.list_(pa.float16(), hidden_dim))
                ])
                writer = ipc.new_file(sink, schema)
                writers[key] = (sink, writer)
                buffers[key] = []
                hidden_dims[key] = hidden_dim
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
                    f"📝 Flushed {len(buffers[key])} frames for {vpath}/{feature_type} "
                    f"({num_patches}×{hidden_dim}) in {t1 - t0:.3f} sec"
                )
                buffers[key] = []

        elif item["type"] == "video_done":
            vpath = item["video_path"]
            # flush and close all features for this video
            for (vid, feat), arrs in list(buffers.items()):
                if vid == vpath:
                    if arrs:
                        hidden_dim = hidden_dims[(vid, feat)]
                        t0 = time.time()
                        table = pa.table({feat: pa.concat_arrays(arrs)})
                        writers[(vid, feat)][1].write_table(table)
                        t1 = time.time()
                        logger.info(
                            f"📝 Final flush {len(arrs)} frames for {vpath}/{feat} in {t1 - t0:.3f} sec"
                        )
                        buffers[(vid, feat)] = []

                    # Close writer
                    sink, writer = writers[(vid, feat)]
                    writer.close()
                    sink.close()
                    del writers[(vid, feat)]
                    del buffers[(vid, feat)]
                    del hidden_dims[(vid, feat)]
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
