from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Sample documents
documents = [
    "The cat sat on the mat.",
    "The dog barked at the cat.",
    "The mouse ran away from the cat and dog."
]

# Get NLTK's English stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase and remove punctuation
    import re
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def build_inverted_index(docs, stop_words):
    inverted_index = defaultdict(list)
    for doc_id, text in enumerate(docs):
        tokens = word_tokenize(clean_text(text))
        for token in tokens:
            if token not in stop_words and doc_id not in inverted_index[token]:
                inverted_index[token].append(doc_id)
    return inverted_index

index = build_inverted_index(documents, stop_words)
# Print the inverted index (sorted alphabetically)
for term in sorted(index.keys()):
    print(f"{term}: {index[term]}")