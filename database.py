import os
from abc import abstractmethod
from logger import logger
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import time
import numpy as np

class BaseVectorDB:
    def __init__(self, db_config):
        self.db_config = db_config
        self.db_name = self.db_config.get("name", "default")
        if self.db_config["type"] != "milvus":
            raise ValueError("Currently only 'milvus' type is supported.")

    @abstractmethod
    def insert(self, data):
        raise NotImplementedError("insert() must be implemented by subclasses.")
    
    @abstractmethod
    def read(self, feature_name):
        raise NotImplementedError("read() must be implemented by subclasses.")
    
    @abstractmethod
    def close(self):
        raise NotImplementedError("close() must be implemented by subclasses.")

milvus_dtype_map = {
    "BOOL": DataType.BOOL,
    "INT8": DataType.INT8,
    "INT16": DataType.INT16,
    "INT32": DataType.INT32,
    "INT64": DataType.INT64,
    "FLOAT": DataType.FLOAT,
    "DOUBLE": DataType.DOUBLE,
    "STRING": DataType.STRING,
    "VARCHAR": DataType.VARCHAR,
    "ARRAY": DataType.ARRAY,
    "JSON": DataType.JSON,
    "GEOMETRY": DataType.GEOMETRY,
    "FLOAT_VECTOR": DataType.FLOAT_VECTOR,
    "BINARY_VECTOR": DataType.BINARY_VECTOR,
    "FLOAT16_VECTOR": DataType.FLOAT16_VECTOR,
    "BFLOAT16_VECTOR": DataType.BFLOAT16_VECTOR,
    "SPARSE_FLOAT_VECTOR": DataType.SPARSE_FLOAT_VECTOR
}

class MilvusVectorDB(BaseVectorDB):
    
    def __init__(self, db_config):
        super().__init__(db_config)

        # self.DB_BASE_ROOT = "./db/milvus"

        # self.db_path = os.path.join(self.DB_BASE_ROOT, self.db_name)
        self.host = db_config.get("host", "127.0.0.1")
        self.port = db_config.get("port", "19530")
        self.batch_size = db_config.get("batch_size", 5000)
        self.uri = f"http://{self.host}:{self.port}"
        try:
            self.client = MilvusClient(uri=self.uri)
        except:
            raise ConnectionError(f"Failed to connect to Milvus database at {self.uri}.")

        self.collections = {}  # store collection to field mapping

        collections_cfg = self.db_config.get("collections", [])
        for collection_cfg in collections_cfg:
            collection_name = collection_cfg["name"]
            fields_cfg = collection_cfg["fields"]
            self.collections[collection_name] = [f["name"] for f in fields_cfg]

            fields = []
            for f in fields_cfg:
                dtype = milvus_dtype_map.get(f["dtype"], DataType.UNKNOWN)
                kwargs = {
                    "name": f["name"],
                    "dtype": dtype,
                }
                if dtype in (DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR):
                    kwargs["dim"] = f["dim"]
                if dtype == DataType.VARCHAR:
                    kwargs["max_length"] = f["max_length"]
                if f.get("is_primary", False):
                    kwargs["is_primary"] = True
                if f.get("auto_id", False):
                    kwargs["auto_id"] = True

                fields.append(FieldSchema(**kwargs))

            schema = CollectionSchema(fields)

            if self.client.has_collection(collection_name):
                logger.info(f"Collection '{collection_name}' already exists.")
            else:
                logger.info(f"Creating collection '{collection_name}'.")
                self.client.create_collection(collection_name=collection_name, schema=schema)
                logger.info(f"Collection '{collection_name}' created successfully.")

    def insert(self, feature_name, video_path, feature_value):
        
        collection_name = f'{feature_name}_collection'
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' does not exist in the database.")
        
        # insert video embedding into collection
        if feature_name == "video_embedding":
            self.insert_video_embedding(collection_name, video_path, feature_value)

        if feature_name == "video_embedding_pooling":
            self.insert_video_embedding_pooling(collection_name, video_path, feature_value)

    def insert_video_embedding(self, collection_name, video_path, feature_value):
        # Feature_value is a dictionary with keys "frame_id", "grid_rows", "grid_cols", and "embeddings"
        fid = feature_value["frame_id"]
        rows = feature_value["grid_rows"]
        cols = feature_value["grid_cols"]
        embs = feature_value["embeddings"]

        records = [
            {
                "video_path": video_path,
                "frame_id": fid,
                "row_idx": r,
                "col_idx": c,
                "embeddings": embs[r * cols + c]
            }
            for r in range(rows)
            for c in range(cols)
        ]

        # for i in tqdm(range(0, len(records), self.batch_size), desc="Inserting patches", unit="batch"):
        for i in range(0, len(records), self.batch_size):
            batch = records[i: i + self.batch_size]
            self.client.insert(collection_name, batch)

    def insert_video_embedding_pooling(self, collection_name, video_path, feature_value):
        # Feature_value is a dictionary with keys "frame_id", "grid_rows", "grid_cols", and "embeddings"
        frame_num = len(feature_value["frame_embeddings"]) 
        embs = feature_value["frame_embeddings"]

        records = [
            {
                "video_path": video_path,
                "frame_id": f,
                "frame_embeddings": embs[f]
            }
            for f in range(frame_num)
        ]

        # for i in tqdm(range(0, len(records), self.batch_size), desc="Inserting patches", unit="batch"):
        self.client.insert(collection_name, records)
        
    def read(self, feature_name):

        collection_name = f'{feature_name}_collection'
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' does not exist in the database.")
        
        results = self.client.query(collection_name, limit=10)
        return results

    def close(self):
        logger.info("Closing Milvus connection.")
        self.client.close()


db_map = {
    "milvus": MilvusVectorDB,
}

def init_db(db_config):
    """
    Initialize the database based on the configuration.
    :param db_config: Configuration dictionary for the database.
    :return: An instance of the database class.
    """
    db_type = db_config.get("type")
    if db_type not in db_map:
        raise ValueError(f"Unsupported database type: {db_type}. Supported types are: {list(db_map.keys())}.")
    db_class = db_map[db_type]
    db_instance = db_class(db_config)
    return db_instance

if __name__ == "__main__":
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
                    {"name": "id", "dtype": "INT64", "is_primary": "true", "auto_id": "true"},
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
    db = init_db(db_config)

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
            db.insert_video_embedding("test_inserting_speed_collection", f"video_{fid}.mp4", feature_value)
            elapsed = time.time() - start_time
            print(f"Inserted {fid+1} frames, elapsed {elapsed:.2f}s, avg { (fid+1)/elapsed:.2f} frames/sec")

        total_time = time.time() - start_time
        print(f"\n✅ Done. Inserted {num_frames} frames in {total_time:.2f}s "
            f"({num_frames/total_time:.2f} frames/sec).")

    benchmark_inserts(num_frames=1000)
    db.close()