import ollama
from sklearn.feature_extraction.text import TfidfVectorizer

# Assume you have a text document stored in a variable called 'document'
document = "This is a sample text document that needs to be vectorized."

# Initialize Ollama (assuming it's running and accessible)
# Replace 'your_ollama_model' with the actual model name you want to use for embeddings
# For example, 'nomic-embed-text' or 'all-minilm'
try:
    ollama.embeddings(model='nomic-embed-text', prompt='This is a test message')
except ollama.ResponseError as e:
    print(f"Ollama not running or model not found: {e}")
    print("Please make sure Ollama is running and you have the 'nomic-embed-text' model pulled.")
    print("You can pull the model using: ollama pull nomic-embed-text")
    exit() # Exit if Ollama is not set up

# Generate embeddings using Ollama
try:
    response = ollama.embeddings(
        model='nomic-embed-text',
        prompt=document
    )
    ollama_embedding = response['embedding']
    print("Ollama Embedding:")
    print(ollama_embedding)
except Exception as e:
    print(f"An error occurred during Ollama embedding: {e}")


# Alternatively, you can use TfidfVectorizer from scikit-learn
# This is a traditional method and doesn't require Ollama
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the document
tfidf_vector = tfidf_vectorizer.fit_transform([document])

print("\nTF-IDF Vector:")
print(tfidf_vector.toarray())

# You can now use the 'ollama_embedding' or 'tfidf_vector' for further tasks like similarity calculations or machine learning models.