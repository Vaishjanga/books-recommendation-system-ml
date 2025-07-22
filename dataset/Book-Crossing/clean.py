import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
from collections import Counter
import time

def fast_genre_classifier():
    """Ultra-fast genre classification for Book-Crossing dataset"""
    
    start_time = time.time()
    
    # Load data
    print("Loading data...")
    books = pd.read_csv('dataset/Book-Crossing/Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
    ratings = pd.read_csv('dataset/Book-Crossing/Ratings.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
    
    print(f"Loaded {len(books)} books and {len(ratings)} ratings")
    
    # Initialize genre column
    books['Genre'] = 'Unknown'
    
    # Method 1: Publisher patterns (VERY FAST)
    publisher_patterns = {
        # Children
        'scholastic': 'Children', 'puffin': 'Children', 'ladybird': 'Children',
        'disney': 'Children', 'nickelodeon': 'Children', 'walker books': 'Children',
        
        # Romance
        'harlequin': 'Romance', 'silhouette': 'Romance', 'avon': 'Romance',
        'berkley romance': 'Romance', 'signet romance': 'Romance',
        
        # Fantasy/Sci-Fi
        'tor': 'Fantasy', 'del rey': 'Fantasy', 'daw books': 'Fantasy',
        'baen': 'Science Fiction', 'ace books': 'Science Fiction',
        
        # Mystery/Thriller
        'mysterious press': 'Mystery', 'minotaur': 'Mystery', 'st. martin\'s dead letter': 'Mystery',
        
        # Academic/Nonfiction
        'oxford university': 'Nonfiction', 'cambridge': 'Nonfiction', 
        'wiley': 'Nonfiction', 'mcgraw': 'Nonfiction', 'penguin reference': 'Nonfiction',
        
        # Comics
        'marvel': 'Comics', 'dc comics': 'Comics', 'dark horse': 'Comics',
        'image comics': 'Comics', 'vertigo': 'Comics'
    }
    
    # Method 2: Title keywords (FAST)
    title_keywords = {
        'Fantasy': ['harry potter', 'lord of the rings', 'dragon', 'magic', 'wizard', 
                   'witch', 'spell', 'kingdom', 'quest', 'fairy', 'enchant'],
        'Mystery': ['murder', 'detective', 'mystery', 'crime', 'investigation', 
                   'clue', 'sherlock', 'agatha christie', 'whodunit'],
        'Romance': ['love', 'romance', 'heart', 'kiss', 'bride', 'wedding', 
                   'affair', 'desire', 'passion'],
        'Science Fiction': ['space', 'alien', 'robot', 'future', 'sci-fi', 
                           'galaxy', 'star wars', 'star trek', 'cyberpunk'],
        'Horror': ['horror', 'scary', 'ghost', 'vampire', 'zombie', 'terror', 
                  'haunted', 'stephen king', 'nightmare'],
        'Children': ['children', 'kids', 'juvenile', 'young readers', 'picture book',
                    'bedtime', 'abc', 'counting'],
        'Biography': ['biography', 'autobiography', 'life of', 'memoir', 'diary',
                     'letters of', 'journals'],
        'Historical': ['history', 'historical', 'war', 'wwii', 'civil war', 
                      'medieval', 'ancient', 'dynasty'],
        'Thriller': ['thriller', 'suspense', 'conspiracy', 'espionage', 'cia', 'fbi'],
        'Nonfiction': ['guide', 'how to', 'manual', 'cookbook', 'textbook', 
                      'reference', 'encyclopedia', 'dictionary'],
        'Comics': ['comic', 'manga', 'graphic novel', 'superhero'],
        'Poetry': ['poetry', 'poems', 'verse', 'haiku', 'anthology']
    }
    
    # Method 3: Author mapping (INSTANT)
    author_genres = {
        'stephen king': 'Horror', 'dean koontz': 'Horror',
        'agatha christie': 'Mystery', 'arthur conan doyle': 'Mystery',
        'j.k. rowling': 'Fantasy', 'j.r.r. tolkien': 'Fantasy',
        'george r.r. martin': 'Fantasy', 'terry pratchett': 'Fantasy',
        'isaac asimov': 'Science Fiction', 'philip k. dick': 'Science Fiction',
        'danielle steel': 'Romance', 'nora roberts': 'Romance',
        'dr. seuss': 'Children', 'roald dahl': 'Children',
        'malcolm gladwell': 'Nonfiction', 'stephen hawking': 'Nonfiction'
    }
    
    def classify_single_book(book_data):
        """Classify a single book"""
        title = str(book_data.get('Book-Title', '')).lower()
        author = str(book_data.get('Book-Author', '')).lower()
        publisher = str(book_data.get('Publisher', '')).lower()
        
        # Check publisher first (fastest)
        for pub_key, genre in publisher_patterns.items():
            if pub_key in publisher:
                return genre
        
        # Check author
        for auth_key, genre in author_genres.items():
            if auth_key in author:
                return genre
        
        # Check title keywords
        for genre, keywords in title_keywords.items():
            for keyword in keywords:
                if keyword in title:
                    return genre
        
        # Year-based hints
        year = book_data.get('Year-Of-Publication', 0)
        if isinstance(year, (int, float)) and year > 0:
            if year < 1950:
                if 'war' in title or 'history' in title:
                    return 'Historical'
                elif 'life' in title or 'story' in title:
                    return 'Biography'
        
        return 'Unknown'
    
    # Process books
    print("\nClassifying books...")
    
    # Convert to dict for faster processing
    books_dict = books.to_dict('records')
    
    # Process in chunks with progress
    chunk_size = 10000
    for i in range(0, len(books_dict), chunk_size):
        chunk = books_dict[i:i+chunk_size]
        for j, book in enumerate(chunk):
            genre = classify_single_book(book)
            books.at[i+j, 'Genre'] = genre
        
        print(f"Processed {min(i+chunk_size, len(books_dict))}/{len(books_dict)} books")
    
    # Add ratings statistics
    print("\nCalculating ratings statistics...")
    ratings_stats = ratings.groupby('ISBN').agg({
        'Rating': ['count', 'mean', 'std', 'min', 'max']
    }).round(2)
    ratings_stats.columns = ['NumRatings', 'AvgRating', 'StdRating', 'MinRating', 'MaxRating']
    
    # Merge ratings
    books = books.merge(ratings_stats, left_on='ISBN', right_index=True, how='left')
    books['NumRatings'] = books['NumRatings'].fillna(0).astype(int)
    books['AvgRating'] = books['AvgRating'].fillna(0)
    
    # Calculate popularity score
    books['PopularityScore'] = (
        books['NumRatings'] * 0.3 + 
        books['AvgRating'] * 0.7
    ).round(2)
    
    # Genre statistics
    print("\nGenre Distribution:")
    genre_dist = books['Genre'].value_counts()
    for genre, count in genre_dist.items():
        print(f"{genre}: {count} ({count/len(books)*100:.1f}%)")
    
    # Save results
    print("\nSaving results...")
    books.to_csv('Books_with_genres.csv', sep=';', index=False)
    
    # Create a summary file
    summary = pd.DataFrame({
        'Genre': genre_dist.index,
        'Count': genre_dist.values,
        'Percentage': (genre_dist.values / len(books) * 100).round(2)
    })
    summary.to_csv('Genre_Summary.csv', index=False)
    
    end_time = time.time()
    print(f"\nCOMPLETED in {(end_time - start_time)/60:.1f} minutes!")
    print(f"Total books processed: {len(books)}")
    print(f"Books with known genres: {len(books[books['Genre'] != 'Unknown'])}")
    print(f"Output saved to: Books_with_genres.csv")
    
    return books

# Run the classifier
if __name__ == "__main__":
    books_with_genres = fast_genre_classifier()