import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

def main():
    print("==========================================================")
    print("Re-embedding Script for Sigma Web Development Dataset")
    print("==========================================================")
    
    # Check dependencies
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: sentence-transformers is not installed.")
        print("Please install it using: pip install sentence-transformers")
        sys.exit(1)
        
    model_name = os.getenv("MODEL_NAME", "BAAI/bge-small-en-v1.5")
    output_path = os.getenv("INDEX_PATH", "embeddings.joblib")
    
    print(f"Loading sentence-transformer model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    jsons_dir = "jsons"
    if not os.path.exists(jsons_dir):
        print(f"Error: Transcript directory '{jsons_dir}' not found.")
        sys.exit(1)
        
    jsons = [f for f in os.listdir(jsons_dir) if f.endswith(".json")]
    if not jsons:
        print(f"Error: No JSON files found in '{jsons_dir}'.")
        sys.exit(1)
        
    print(f"Found {len(jsons)} transcript JSON files. Starting embedding generation...")
    my_dicts = []
    chunk_id = 0
    
    # Process files
    for idx, json_file in enumerate(jsons, 1):
        file_path = os.path.join(jsons_dir, json_file)
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            
        chunks = content.get("chunks", [])
        if not chunks:
            continue
            
        print(f"[{idx}/{len(jsons)}] Embedding {len(chunks)} chunks from {json_file}...")
        
        # Batch extract texts
        texts = [chunk["text"] for chunk in chunks]
        
        # Compute embeddings
        # convert_to_numpy=True returns numpy array. We normalize for cosine similarity.
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-9)
        
        # Save to records
        for i, chunk in enumerate(chunks):
            chunk["chunk_id"] = chunk_id
            chunk["embedding"] = normalized_embeddings[i].tolist() # save as list for joblib/json compatibility
            chunk_id += 1
            my_dicts.append(chunk)

    print(f"Embedding complete. Stacking dataframe for {len(my_dicts)} total chunks...")
    df = pd.DataFrame.from_records(my_dicts)
    
    # Save the dataframe
    print(f"Saving dataframe to {output_path}...")
    joblib.dump(df, output_path)
    
    print("==========================================================")
    print("SUCCESS!")
    print(f"Index successfully updated and saved to: {output_path}")
    print(f"New embedding dimensions: {len(df['embedding'].iloc[0])}")
    print(f"Total chunks indexed: {len(df)}")
    print("==========================================================")

if __name__ == "__main__":
    main()
