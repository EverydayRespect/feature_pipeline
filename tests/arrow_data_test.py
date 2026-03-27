import os
from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as ipc

ROOT_DIRS = [
    Path("/mnt/14t_drive/VideoLLaMa3_embeddings_240_420_fps1"),
]

MAGIC = b"ARROW1"
MAGIC_LEN = len(MAGIC)


def has_valid_arrow_magic(path: Path):
    """
    Fast check using magic bytes.
    Returns:
        True  -> looks like a complete Arrow IPC file
        False -> corrupted / truncated / not arrow
    """
    try:
        size = path.stat().st_size
        if size < 2 * MAGIC_LEN:
            return False

        with open(path, "rb") as f:
            head = f.read(MAGIC_LEN)
            f.seek(-MAGIC_LEN, os.SEEK_END)
            tail = f.read(MAGIC_LEN)

        return head == MAGIC and tail == MAGIC

    except Exception:
        return False


def count_embeddings(path: Path):
    """
    Slow path: real Arrow parsing
    """
    total = 0
    with pa.memory_map(os.fspath(path), "r") as source:
        reader = ipc.open_file(source)
        for i in range(reader.num_record_batches):
            total += reader.get_batch(i).num_rows
    return total


def main():
    output_file = Path("magic_ok_files.txt")
    with open(output_file, "w") as f:
        for root_dir in ROOT_DIRS:
            arrow_files = list(root_dir.rglob("*.arrow")) + list(root_dir.rglob("*.pyarrow"))

            total_files = len(arrow_files)
            magic_ok = []
            magic_bad = []

            print(f"🔍 Scanning {total_files} arrow files (magic byte only)...")

            # ===== Stage 1: magic byte =====
            for path in arrow_files:
                if has_valid_arrow_magic(path):
                    magic_ok.append(path)
                else:
                    magic_bad.append(path)

            print(root_dir)
            print("========== MAGIC BYTE RESULT ==========")
            print(f"Total files      : {total_files}")
            print(f"✅ Magic OK       : {len(magic_ok)}")
            print(f"❌ Magic BAD     : {len(magic_bad)}")
            print("======================================\n")

            # Write magic_ok files to output file
            for path in magic_ok:
                f.write(str(path) + "\n")

if __name__ == "__main__":
    main()
