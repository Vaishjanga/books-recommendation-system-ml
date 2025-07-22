import numpy as np
import pandas as pd
from app import app
from models import db, Book
from tqdm import tqdm

# Load embeddings and book metadata
embeddings = np.load("dataset/Book-Crossing/book_embeddings_minilm.npy")
books_df = pd.read_csv("dataset/Book-Crossing/Books.csv", sep=';', encoding='latin-1')

with app.app_context():
    # Load all books into a dictionary for fast lookup
    books = {book.isbn: book for book in Book.query.all()}
    for idx, row in tqdm(books_df.iterrows(), total=len(books_df), desc='Updating plot embeddings'):
        isbn = str(row['ISBN'])
        embedding = embeddings[idx].tolist()
        book = books.get(isbn)
        if book:
            book.plot_embedding = embedding
    db.session.commit()
    print("All book plot embeddings have been stored in the database (batch update).") 