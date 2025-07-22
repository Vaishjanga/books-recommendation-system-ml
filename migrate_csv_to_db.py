import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from models import db, User, Book, Rating, UserFeedback
from flask import Flask
import json
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import logging
from tqdm import tqdm
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://vaish:Vaish_811496@localhost/recommendations_db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    return app

def migrate_data():
    """Migrate Book-Crossing Books.csv, Ratings.csv, and Users.csv to PostgreSQL database."""
    app = create_app()
    with app.app_context():
        db.create_all()
        import pandas as pd
        import numpy as np
        import random
        from models import User, Book, Rating
        from werkzeug.security import generate_password_hash
        from sentence_transformers import SentenceTransformer
        from textblob import TextBlob
        
        # Load CSV files
        print("Loading CSV files...")
        books_df = pd.read_csv("dataset/Book-Crossing/Books.csv", sep=';', encoding='latin-1', on_bad_lines='skip')
        ratings_df = pd.read_csv("dataset/Book-Crossing/Ratings.csv", sep=';', encoding='latin-1', on_bad_lines='skip')
        users_df = pd.read_csv("dataset/Book-Crossing/Users.csv", sep=';', encoding='latin-1', on_bad_lines='skip')
        
        # Clean column names
        books_df.columns = [col.strip().replace('-', '_').lower() for col in books_df.columns]
        ratings_df.columns = [col.strip().replace('-', '_').lower() for col in ratings_df.columns]
        users_df.columns = [col.strip().replace('-', '_').lower() for col in users_df.columns]
        
        # Initialize embedder
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # --- USERS ---
        print("Migrating users...")
        default_password_hash = generate_password_hash("default123", method='pbkdf2:sha256')
        countries = ['USA', 'UK', 'Germany', 'Canada', 'Australia', 'India', 'France', 'Japan']
        for i, row in tqdm(users_df.iterrows(), total=len(users_df)):
            try:
                # Skip if user already exists
                if db.session.get(User, str(row['user_id'])):
                    continue
                location = random.choice(countries)
                try:
                    age_val = int(row['age']) if 'age' in row and not pd.isna(row['age']) and str(row['age']).isdigit() else random.randint(18, 70)
                    if 5 <= age_val <= 90:
                        age = age_val
                    else:
                        age = random.randint(18, 70)
                except (ValueError, TypeError):
                    age = random.randint(18, 70)
                user = User(
                    id=str(row['user_id']),
                    username=f"user_{row['user_id']}",
                    email=f"user_{row['user_id']}@example.com",
                    age=age,
                    location=location,
                    preferences={}
                )
                user.password_hash = default_password_hash
                db.session.add(user)
                if (i+1) % 1000 == 0:
                    db.session.commit()
                    print(f"Inserted {i+1} users...")
            except Exception as e:
                print(f"Error migrating user {row['user_id']}: {e}")
                db.session.rollback()
                continue
        db.session.commit()
        print(f"Users migration completed: {len(users_df)} users")
        
        # --- BOOKS ---
        print("Migrating books...")
        for i, row in tqdm(books_df.iterrows(), total=len(books_df)):
            try:
                # Skip if book already exists
                if Book.query.get(str(row['isbn'])):
                    continue
                title = row['title'] if pd.notna(row['title']) else "Unknown Title"
                author = row['author'] if pd.notna(row['author']) else "Unknown Author"
                text = f"{title} by {author}"
                embedding = embedder.encode(text, convert_to_numpy=True).tolist()
                sentiment = TextBlob(title).sentiment.polarity
                def get_mood(score):
                    if score > 0.5: return "inspirational"
                    elif score < -0.2: return "dark"
                    return "neutral"
                year = int(row['year']) if 'year' in row and not pd.isna(row['year']) and str(row['year']).isdigit() else None
                book = Book(
                    id=str(row['isbn']),
                    isbn=str(row['isbn']),
                    title=title,
                    author=author,
                    year=year,
                    publisher=row.get('publisher'),
                    image_url_s=row.get('image_url_s') if 'image_url_s' in row else None,
                    image_url_m=row.get('image_url_m') if 'image_url_m' in row else None,
                    image_url_l=row.get('image_url_l') if 'image_url_l' in row else None,
                    plot_embedding=embedding,
                    sentiment=sentiment,
                    mood=get_mood(sentiment),
                    freshness_score=1 if year and year >= 2015 else 0
                )
                db.session.add(book)
                if (i+1) % 1000 == 0:
                    db.session.commit()
                    print(f"Inserted {i+1} books...")
            except Exception as e:
                print(f"Error migrating book {row['isbn']}: {e}")
                db.session.rollback()
                continue
        db.session.commit()
        print(f"Books migration completed: {len(books_df)} books")
        
        # --- RATINGS ---
        print("Migrating ratings...")
        for i, row in tqdm(ratings_df.iterrows(), total=len(ratings_df)):
            try:
                # Only insert if both user and book exist
                user_id = str(row['user_id'])
                book_id = str(row['isbn'])
                if not User.query.get(user_id):
                    continue
                if not Book.query.get(book_id):
                    continue
                # Optionally, skip if rating already exists
                if Rating.query.filter_by(user_id=user_id, book_id=book_id).first():
                    continue
                rating = Rating(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    book_id=book_id,
                    rating=row['rating'] if 'rating' in row else None,
                    review_text=None,
                    device=None,
                    platform=None,
                    session_id=None,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                db.session.add(rating)
                if (i + 1) % 100000 == 0:
                    print(f"Inserted {i+1} ratings...")
                    db.session.commit()
            except Exception as e:
                print(f"Error migrating rating: {e}")
                db.session.rollback()
        db.session.commit()
        print(f"Ratings migration completed: {len(ratings_df)} rows processed")

    

    logger.info("Updating book popularity scores...")
    books = Book.query.all()
    for book in books:
        rating_count = Rating.query.filter_by(book_id=book.id).count()
        avg_rating = db.session.query(db.func.avg(Rating.rating)).filter_by(book_id=book.id).scalar() or 0
        book.popularity_score = rating_count
        book.quality_score = (
            0.4 * min(rating_count / 100, 1) +
            0.3 * (book.sentiment or 0) +
            0.3 * (book.freshness_score or 0)
        )
    db.session.commit()
    logger.info("Book popularity/quality scores updated!")

if __name__ == "__main__":
    migrate_data()