import time
import numpy as np
from .. import database

# Example DB config
db_config = {
    "type": "milvus",
    "name": "testdb",
    "host": "127.0.0.1",
    "port": "19530",
    "batch_size": 1000,
    "collections": [
        {
            "name": "test_inserting_speed_collection",
            "fields": [
                {"name": "video_path", "dtype": "VARCHAR", "max_length": 200},
                {"name": "frame_id", "dtype": "INT32"},
                {"name": "row_idx", "dtype": "INT32"},
                {"name": "col_idx", "dtype": "INT32"},
                {"name": "embeddings", "dtype": "FLOAT_VECTOR", "dim": 1024}
            ]
        }
    ]
}

# Initialize DB
db = database.init_db(db_config)

def generate_fake_feature(frame_id, rows, cols, dim=1024):
    """Generate synthetic embedding features for testing"""
    embeddings = [np.random.rand(dim).astype("float32").tolist() for _ in range(rows * cols)]
    return {
        "frame_id": frame_id,
        "grid_rows": rows,
        "grid_cols": cols,
        "embeddings": embeddings
    }

def benchmark_inserts(num_frames=1000, rows=51, cols=91, dim=1024):
    start_time = time.time()
    
    for fid in range(num_frames):
        feature_value = generate_fake_feature(fid, rows, cols, dim)
        db.insert_video_embedding("video_embedding_collection", f"video_{fid}.mp4", feature_value)
        elapsed = time.time() - start_time
        print(f"Inserted {fid+1} frames, elapsed {elapsed:.2f}s, avg { (fid+1)/elapsed:.2f} frames/sec")

    total_time = time.time() - start_time
    print(f"\n✅ Done. Inserted {num_frames} frames in {total_time:.2f}s "
          f"({num_frames/total_time:.2f} frames/sec).")

if __name__ == "__main__":
    benchmark_inserts(num_frames=1000, rows=4, cols=4, dim=1024)
    db.close()
