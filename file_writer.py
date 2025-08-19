import os
import queue
import threading
import time
from typing import Dict, List, Any
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SENTINEL = object()

class ParquetShardWriter:
    """
    每个 writer 一个有界队列 + 后台线程：
    - 接收若干帧记录(list[dict])；
    - 聚到 shard_rows 或超时就 flush 成一个 parquet 分片；
    - Schema（每行=一帧）：
        video_path: string
        feature_name: string
        frame_id: int64
        grid_rows: int16
        grid_cols: int16
        emb_dim: int32
        num_patches: int32
        embedding: list<list<float32/16>>   # shape=(num_patches, emb_dim)
    """
    def __init__(self,
                 root_dir: str,
                 group_id: int,
                 gpu_id: int,
                 worker_id: int,
                 shard_rows: int = 20_000,
                 max_queue: int = 5_000,      # 有界 -> backpressure
                 compression: str = "zstd",
                 embedding_dtype: str = "float32",
                 flush_interval_s: float = 60.0):
        self.root_dir = root_dir
        self.group_id = group_id
        self.gpu_id = gpu_id
        self.worker_id = worker_id
        self.shard_rows = shard_rows
        self.max_queue = max_queue
        self.compression = compression
        self.embedding_dtype = embedding_dtype
        self.flush_interval_s = flush_interval_s

        self.group_dir = self.root_dir
        os.makedirs(self.group_dir, exist_ok=True)

        self.q: "queue.Queue[Any]" = queue.Queue(maxsize=max_queue)
        self._buf: List[Dict[str, Any]] = []
        self._seq = 0
        self._stop = threading.Event()

        self._thr = threading.Thread(target=self._run, name=f"writer-{gpu_id}-t{worker_id}", daemon=True)
        self._thr.start()

    def enqueue(self, records: List[Dict[str, Any]]):
        self.q.put(records)  # 队列满则阻塞，形成背压

    def close(self):
        self.q.put(SENTINEL)
        self._stop.set()
        self._thr.join()

    # ------- 内部实现 -------

    def _run(self):
        last_flush = time.time()
        while True:
            try:
                item = self.q.get(timeout=1.0)
            except queue.Empty:
                item = None

            if item is SENTINEL:
                if self._buf:
                    self._flush_now()
                break

            if item:
                self._buf.extend(item)

            need_time_flush = self._buf and (time.time() - last_flush) >= self.flush_interval_s
            if len(self._buf) >= self.shard_rows or need_time_flush:
                self._flush_now()
                last_flush = time.time()

        if self._buf:
            self._flush_now()

    def _next_path(self) -> str:
        self._seq += 1
        filename = f"group{self.group_id}_{self._seq:05d}.parquet"
        return os.path.join(self.group_dir, filename)

    def _normalize_embs(self, embs, rows: int, cols: int):
        """
        将 embs 统一为二维 (num_patches, dim) 并返回 (arr2d, dim)。
        支持输入形状：
            (rows*cols, dim) 或 (rows, cols, dim) 或 (dim,)
        """
        arr = np.asarray(embs)
        if arr.ndim == 3:
            r, c, dim = arr.shape
            if r != rows or c != cols:
                raise ValueError(f"embeddings shape {arr.shape} not match (grid_rows, grid_cols)=({rows},{cols})")
            arr2d = arr.reshape(rows * cols, dim)
        elif arr.ndim == 2:
            num, dim = arr.shape
            if num != rows * cols:
                # 也允许没有网格拆分：rows*cols==1 的情况
                if not (rows == 1 and cols == 1):
                    raise ValueError(f"embeddings shape {arr.shape} not match rows*cols={rows*cols}")
            arr2d = arr
        elif arr.ndim == 1:
            dim = arr.shape[0]
            arr2d = arr.reshape(1, dim)
        else:
            raise ValueError(f"Unsupported embeddings ndim={arr.ndim}")

        # dtype 统一
        target_dtype = np.float16 if self.embedding_dtype == "float16" else np.float32
        if arr2d.dtype != target_dtype:
            arr2d = arr2d.astype(target_dtype, copy=False)
        return arr2d, int(arr2d.shape[1])

    def _flush_now(self):
        rows = self._buf
        self._buf = []

        # 构造 Arrow 列
        video_paths, feature_names = [], []
        frame_ids, grid_rows, grid_cols = [], [], []
        emb_dims, num_patches = [], []
        emb_cells: List[List[List[float]]] = []  # list of 2D list

        for r in rows:
            vp = r["video_path"]
            fn = r["feature_name"]
            fid = int(r["frame_id"])
            gr = int(r["grid_rows"])
            gc = int(r["grid_cols"])
            arr2d, dim = self._normalize_embs(r["embeddings"], gr, gc)

            video_paths.append(vp)
            feature_names.append(fn)
            frame_ids.append(fid)
            grid_rows.append(gr)
            grid_cols.append(gc)
            emb_dims.append(dim)
            num_patches.append(int(arr2d.shape[0]))
            emb_cells.append(arr2d.tolist())  # 2D -> list(list(float))

        elem_ty = pa.float16() if self.embedding_dtype == "float16" else pa.float32()

        table = pa.Table.from_arrays(
            [
                pa.array(video_paths),                         # string
                pa.array(feature_names),                       # string
                pa.array(frame_ids, type=pa.int64()),
                pa.array(grid_rows, type=pa.int16()),
                pa.array(grid_cols, type=pa.int16()),
                pa.array(emb_dims, type=pa.int32()),
                pa.array(num_patches, type=pa.int32()),
                pa.array(emb_cells, type=pa.list_(pa.list_(elem_ty))),
            ],
            names=[
                "video_path", "feature_name", "frame_id",
                "grid_rows", "grid_cols", "emb_dim", "num_patches",
                "embedding"
            ],
        )

        tmp = self._next_path() + ".tmp"
        pq.write_table(
            table,
            tmp,
            compression=self.compression,
            data_page_size=1 << 16,  # 64KB
            write_statistics=True
        )
        os.replace(tmp, tmp[:-4])  # 原子替换成 .parquet
