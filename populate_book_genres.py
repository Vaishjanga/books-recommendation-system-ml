import pandas as pd
from app import app
from models import db, BookGenre

# Load the CSV
books_with_genres = pd.read_csv('Books_with_genres.csv', sep=';')

with app.app_context():
    count = 0
    for _, row in books_with_genres.iterrows():
        isbn = str(row['ISBN'])
        genre = row['Genre']
        if genre and genre != 'Unknown':
            bg = BookGenre.query.get(isbn)
            if bg:
                bg.genre = genre
            else:
                db.session.add(BookGenre(isbn=isbn, genre=genre))
            count += 1
    db.session.commit()
    print(f"Inserted/updated {count} book genres.") 