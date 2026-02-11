import pyarrow.ipc as ipc
import os

def analyze_arrow_file(file_path):
    # 1. Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"Analyzing: {file_path}")
    print("-" * 30)

    try:
        # 2. Open the file using memory mapping for efficiency
        with open(file_path, 'rb') as f:
            reader = ipc.open_file(f)
            
            # 3. Get metadata
            num_record_batches = reader.num_record_batches
            schema = reader.schema
            
            # 4. Calculate total rows
            # We iterate through batch metadata to avoid loading the actual data into RAM
            total_rows = sum(reader.get_batch(i).num_rows for i in range(num_record_batches))

            print("Status: VALID Apache Arrow file.")
            print(f"Total Rows (Lines): {total_rows}")
            print(f"Number of Record Batches: {num_record_batches}")
            print("\nSchema / Columns:")
            print(schema)

    except Exception as e:
        print(f"Status: INVALID or Corrupted Arrow file.")
        print(f"Error details: {e}")

if __name__ == "__main__":
    target_path = '/mnt/14t_drive/VideoLLaMa3_embeddings_240_420_fps1/fifth_45_videos/evidence.com_evidence_package_1_of_3_created_2024-06-06T15_24_24Z/Blocking_Sidewalk.video_embedding.arrow'
    analyze_arrow_file(target_path)