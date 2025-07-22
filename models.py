# models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
from sqlalchemy.dialects.postgresql import JSON
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
from flask_login import UserMixin

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    age = db.Column(db.Integer)
    location = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)
    preferences = db.Column(JSON)  # Store user preferences as JSON
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)
    
    # Relationships
    ratings = db.relationship('Rating', backref='user', lazy='dynamic')
    feedback = db.relationship('UserFeedback', backref='user', lazy='dynamic')
    sessions = db.relationship('UserSession', backref='user', lazy='dynamic')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'age': self.age,
            'location': self.location,
            'preferences': self.preferences,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_active': self.last_active.isoformat() if self.last_active else None
        }
    


class Book(db.Model):
    __tablename__ = 'books'
    
    id = db.Column(db.String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    isbn = db.Column(db.String(20), unique=True)
    title = db.Column(db.String(500), nullable=False)
    author = db.Column(db.String(255))
    year = db.Column(db.Integer)
    publisher = db.Column(db.String(255))
    image_url_s = db.Column(db.String(500))
    image_url_m = db.Column(db.String(500))
    image_url_l = db.Column(db.String(500))
    
    # Content features
    plot_embedding = db.Column(JSON)  # Store embedding as JSON array
    sentiment = db.Column(db.Float)
    mood = db.Column(db.String(50))
    genre = db.Column(db.String(100))
    language = db.Column(db.String(10))
    page_count = db.Column(db.Integer)
    
    # Computed features
    popularity_score = db.Column(db.Float, default=0.0)
    quality_score = db.Column(db.Float, default=0.0)
    freshness_score = db.Column(db.Float, default=0.0)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    ratings = db.relationship('Rating', backref='book', lazy='dynamic')
    recommendations = db.relationship('Recommendation', backref='book', lazy='dynamic')
    
    def to_dict(self):
        return {
            'id': self.id,
            'isbn': self.isbn,
            'title': self.title,
            'author': self.author,
            'year': self.year,
            'publisher': self.publisher,
            'sentiment': self.sentiment,
            'mood': self.mood,
            'genre': self.genre,
            'popularity_score': self.popularity_score,
            'quality_score': self.quality_score,
            'image_url_s': self.image_url_s,
            'image_url_m': self.image_url_m,
            'image_url_l': self.image_url_l
        }

    def __repr__(self):
        return f'<Book {self.title}>'
    
class Rating(db.Model):
    __tablename__ = 'ratings'
    
    id = db.Column(db.String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(50), db.ForeignKey('users.id'), nullable=False)
    book_id = db.Column(db.String(50), db.ForeignKey('books.id'), nullable=False)
    rating = db.Column(db.Integer, nullable=False)  # 1-10 or 1-5 scale
    review_text = db.Column(db.Text)
    
    # Context
    device = db.Column(db.String(50))
    platform = db.Column(db.String(50))
    session_id = db.Column(db.String(50))
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Unique constraint
    __table_args__ = (db.UniqueConstraint('user_id', 'book_id', name='_user_book_rating'),)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'book_id': self.book_id,
            'rating': self.rating,
            'review_text': self.review_text,
            'device': self.device,
            'platform': self.platform,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class UserFeedback(db.Model):
    __tablename__ = 'user_feedback'
    
    id = db.Column(db.String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(50), db.ForeignKey('users.id'), nullable=False)
    book_id = db.Column(db.String(50), db.ForeignKey('books.id'), nullable=False)
    
    # Explicit feedback
    liked = db.Column(db.Boolean)
    not_interested = db.Column(db.Boolean)
    saved_for_later = db.Column(db.Boolean)
    
    # Implicit feedback
    view_duration = db.Column(db.Integer)  # seconds
    completion_rate = db.Column(db.Float)  # 0.0 to 1.0
    click_through = db.Column(db.Boolean)
    skip = db.Column(db.Boolean)
    re_read = db.Column(db.Boolean)
    social_share = db.Column(db.Boolean)
    
    # Contextual info
    rec_source = db.Column(db.String(100))  # homepage, search, trending, etc.
    user_state = db.Column(db.String(50))   # happy, curious, bored, stressed
    session_duration = db.Column(db.Integer)  # total session time
    
    # Computed
    feedback_score = db.Column(db.Float)
    feedback_category = db.Column(db.String(50))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'book_id': self.book_id,
            'liked': self.liked,
            'not_interested': self.not_interested,
            'saved_for_later': self.saved_for_later,
            'view_duration': self.view_duration,
            'completion_rate': self.completion_rate,
            'feedback_score': self.feedback_score,
            'feedback_category': self.feedback_category,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Recommendation(db.Model):
    __tablename__ = 'recommendations'
    
    id = db.Column(db.String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(50), db.ForeignKey('users.id'), nullable=False)
    book_id = db.Column(db.String(50), db.ForeignKey('books.id'), nullable=False)
    
    # Recommendation metadata
    algorithm = db.Column(db.String(100))  # user_cf, item_cf, content, hybrid, etc.
    score = db.Column(db.Float)
    rank = db.Column(db.Integer)
    explanation = db.Column(db.Text)
    
    # Context when recommended
    context = db.Column(JSON)
    recommended_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # User interaction with recommendation
    viewed = db.Column(db.Boolean, default=False)
    clicked = db.Column(db.Boolean, default=False)
    acted_on = db.Column(db.Boolean, default=False)  # rated, saved, etc.
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'book_id': self.book_id,
            'algorithm': self.algorithm,
            'score': self.score,
            'rank': self.rank,
            'explanation': self.explanation,
            'context': self.context,
            'recommended_at': self.recommended_at.isoformat() if self.recommended_at else None,
            'viewed': self.viewed,
            'clicked': self.clicked,
            'acted_on': self.acted_on
        }

class UserSession(db.Model):
    __tablename__ = 'user_sessions'
    
    id = db.Column(db.String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(50), db.ForeignKey('users.id'), nullable=False)
    session_token = db.Column(db.String(255), unique=True, nullable=False)
    
    # Session info
    device = db.Column(db.String(50))
    platform = db.Column(db.String(50))
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(500))
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)

class ABTest(db.Model):
    __tablename__ = 'ab_tests'
    
    id = db.Column(db.String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    
    # Test configuration
    algorithm_a = db.Column(db.String(100))
    algorithm_b = db.Column(db.String(100))
    traffic_split = db.Column(db.Float, default=0.5)  # 0.5 = 50/50 split
    
    # Status
    is_active = db.Column(db.Boolean, default=False)
    start_date = db.Column(db.DateTime)
    end_date = db.Column(db.DateTime)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ABTestAssignment(db.Model):
    __tablename__ = 'ab_test_assignments'
    
    id = db.Column(db.String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(50), db.ForeignKey('users.id'), nullable=False)
    test_id = db.Column(db.String(50), db.ForeignKey('ab_tests.id'), nullable=False)
    variant = db.Column(db.String(10))  # 'A' or 'B'
    assigned_at = db.Column(db.DateTime, default=datetime.utcnow)

class ModelVersion(db.Model):
    __tablename__ = 'model_versions'
    
    id = db.Column(db.String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    version_name = db.Column(db.String(100), nullable=False)
    algorithm_config = db.Column(JSON)
    model_state = db.Column(JSON)  # Store Q-table, weights, etc.
    
    # Performance metrics
    precision_at_5 = db.Column(db.Float)
    recall_at_5 = db.Column(db.Float)
    ndcg_at_5 = db.Column(db.Float)
    coverage = db.Column(db.Float)
    diversity = db.Column(db.Float)
    
    is_active = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class BookGenre(db.Model):
    __tablename__ = 'book_genres'
    isbn = db.Column(db.String(20), primary_key=True)
    genre = db.Column(db.String(100))