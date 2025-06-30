from modules import pd, faiss, json, os, SentenceTransformer

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

embedder = get_embedder()

folder_path = "faiss/"

files_to_verify = [
    os.path.join(folder_path, "ticket_index.faiss"),
    os.path.join(folder_path, "ticket_history.json"),
    os.path.join(folder_path, "ticket_metadata.json"),
    os.path.join(folder_path, "communication_logs_2024.csv"),
    os.path.join(folder_path, "classified_logs_24.csv"),
]

def verify_files():
    missing_files = list()
    for path in files_to_verify:
        if os.path.exists(path):
            state = "CORRECT"
        else:
            missing_files.append(path)
            state = "WRONG"
        print(f"{path}: {state}")
    print()
    return missing_files

def load_tickets(path, filetype="jsonl"):
    if filetype == "jsonl":
        df = pd.read_json(path, lines=True)
    elif filetype == "csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported filetype")
    return df
    
def build_faiss_index(df, dfc=None):
    grouped = df.groupby("Ticket_ID")
    docs = []
    metadata = []

    if dfc is not None:
        for _, row in dfc.iterrows():
            metadata.append({
                "ID": row["Ticket ID"],
                "category": row["Predicted Category"].strip()
            })

    for _, group in grouped:
        convo = ""
        for _, row in group.iterrows():
            convo += f"{row['Contact_Type'].strip()}: {row['Communication_Log'].strip()}\n"
        full_text = f"{convo.strip()}"
        docs.append(full_text)
    embeddings = embedder.encode(docs, convert_to_numpy=True)

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save Index
    faiss.write_index(index, f"{folder_path}ticket_index.faiss")

    # Save docs for search alignment
    with open(f"{folder_path}ticket_history.json", "w") as f:
        json.dump(docs, f, indent=2)

    # Save metadata
    with open(f"{folder_path}ticket_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

# Query the RAG Index
def load_data_and_index():
    df = load_tickets(f"{folder_path}communication_logs_2024.csv", filetype="csv")
    dfc = load_tickets(f"{folder_path}classified_logs_24.csv", filetype="csv")
    build_faiss_index(df, dfc)

# Load saved data and index
def get_dataset():
  with open(f"{folder_path}ticket_history.json", "r") as f:
      docs = json.load(f)

  with open(f"{folder_path}ticket_metadata.json", "r") as f:
      metadata = json.load(f)

  index = faiss.read_index(f"{folder_path}ticket_index.faiss")

  return docs, metadata, index


if __name__ == "__main__":
    # Verify files
    missing_files = verify_files()
    if missing_files != []:
        # Load data and build index
        load_data_and_index()
    else:
        print("All files are present. No need to rebuild the index.")
        
    