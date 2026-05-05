from sentence_transformers import SentenceTransformer

def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(model, texts):
    return model.encode(texts, show_progress_bar=True)