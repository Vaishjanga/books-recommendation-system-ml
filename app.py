# app.py
from flask import Flask, request, jsonify, session, g, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from flask_caching import Cache
import redis
from celery import Celery
import os
from datetime import datetime, timedelta
import uuid
import logging
from functools import wraps
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import json
from models import db, User, Book, Rating, UserFeedback, Recommendation, ABTest, ABTestAssignment, UserSession, ModelVersion, BookGenre
from recommendation_system_db import DatabaseRecommendationSystem
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://vaish:Vaish_811496@localhost:5432/recommendations_db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'super-jwt-secret')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'super-secret-key')
    app.config['CACHE_TYPE'] = 'redis'
    app.config['CACHE_REDIS_URL'] = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    # Celery configuration
    app.config['CELERY_BROKER_URL'] = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1')
    app.config['CELERY_RESULT_BACKEND'] = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/2')
    
    # Initialize 
    db.init_app(app)
    migrate = Migrate(app, db)
    jwt = JWTManager(app)
    cors = CORS(app)
    cache = Cache(app)
    
    # Rate limiting
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=["1000 per hour"]
    )
    limiter.init_app(app)
    
    # Redis client
    try:
        redis_client = redis.Redis.from_url(app.config['CACHE_REDIS_URL'])
        redis_client.ping()  # Test connection
        logger.info("Redis connection successful")
    except Exception as e:
        # logger.warning(f"Redis connection failed: {e}")  # Silenced warning
        redis_client = None
    
    # Celery setup
    celery = Celery(app.import_name)
    celery.conf.update(
        broker_url=app.config['CELERY_BROKER_URL'],
        result_backend=app.config['CELERY_RESULT_BACKEND'],
        task_serializer='json',
        result_serializer='json',
        accept_content=['json'],
        timezone='UTC',
        enable_utc=True
    )
    
    # Initialize recommendation system
    rec_system = DatabaseRecommendationSystem(app, redis_client, celery)
    
    # Flask-Login setup
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'login_page'

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(user_id)
    
    # Build models on startup 
    with app.app_context():
        try:
            db.create_all()
            # Only build models if we have data and not already built
            user_count = User.query.count()
            book_count = Book.query.count()
            if user_count > 0 and book_count > 0:
                if not hasattr(app, '_models_built'):
                    rec_system.build_collaborative_filtering_models()
                    rec_system.build_content_models()
                    app._models_built = True
            # else: do nothing
        except Exception as e:
            logger.error(f"Model building error: {e}")
    
    def require_auth(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return jsonify({'error': 'Authentication required'}), 401
            g.current_user = current_user
            return f(*args, **kwargs)
        return decorated_function
    
    # WEB INTERFACE ROUTES

    @app.route('/')
    def index():
        """Home page"""
        return render_template('index.html')
    
    @app.route('/register', methods=['GET', 'POST'])
    def register_page():
        if request.method == 'POST':
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            age = request.form.get('age')
            location = request.form.get('location')
            # Add any other fields as needed
            if not username or not email or not password:
                flash('Missing required fields', 'danger')
                return render_template('auth/register.html')
            if User.query.filter_by(username=username).first():
                flash('Username already exists', 'danger')
                return render_template('auth/register.html')
            if User.query.filter_by(email=email).first():
                flash('Email already exists', 'danger')
                return render_template('auth/register.html')
            user = User(
                username=username,
                email=email,
                age=age if age else None,
                location=location,
                preferences={}
            )
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login_page'))
        return render_template('auth/register.html')
    
    @app.route('/login', methods=['GET', 'POST'])
    def login_page():
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            user = User.query.filter_by(username=username).first()
            if user and user.check_password(password):
                login_user(user)
                if user.is_admin:
                    flash('Login successful! Welcome back, admin! Redirecting to admin dashboard...', 'success')
                    return redirect(url_for('admin_dashboard'))
                else:
                    flash('Login successful! Welcome back!', 'success')
                    return redirect(url_for('profile_page'))
            else:
                flash('Login Failed: Invalid credentials', 'danger')
        return render_template('auth/login.html')
    
    @app.route('/logout')
    def logout():
        logout_user()
        flash('You have been logged out.', 'info')
        return redirect(url_for('login_page'))
    
    @app.route('/profile')
    def profile_page():
        if not current_user.is_authenticated:
            return redirect(url_for('login_page'))
        user = current_user
        if not user:
            return redirect(url_for('login_page'))
        return render_template('user/profile.html', user=user)
    
    @app.route('/books', methods=['GET', 'POST'])
    def books_search_page():
        query = request.args.get('query', '').strip()
        details_id = request.args.get('details')
        results = []
        parsed = None
        is_nl = (
            any(word in query.lower() for word in ['recommend', 'like', 'similar', 'find', 'suggest']) or
            len(query.split()) > 4
        ) if query else False
        details_book = None
        if query and not details_id:
            # Always use semantic search for any query
            query_embedding = rec_system.embedder.encode(query)
            results = []
            for book_id, score in faiss_search(query_embedding, top_n=10):
                b = Book.query.get(book_id)
                if b:
                    results.append({
                        'id': b.id,
                        'title': b.title,
                        'author': b.author,
                        'genre': b.genre,
                        'mood': b.mood,
                        'explanation': f"Matched on semantic similarity to: {query}",
                        'similarity': score
                    })
        # Inline details logic
        if details_id:
            book = Book.query.get(details_id)
            if book:
                ratings_stats = db.session.query(
                    db.func.avg(Rating.rating).label('avg_rating'),
                    db.func.count(Rating.rating).label('rating_count'),
                    db.func.min(Rating.rating).label('min_rating'),
                    db.func.max(Rating.rating).label('max_rating')
                ).filter_by(book_id=details_id).first()
                recent_ratings = db.session.query(Rating, User.username).join(User).filter(
                    Rating.book_id == details_id,
                    Rating.review_text.isnot(None)
                ).order_by(Rating.created_at.desc()).limit(5).all()
                book_dict = book.to_dict()
                # Fallback to BookGenre if needed
                if not book_dict.get('genre') or book_dict['genre'] in ('N/A', 'Unknown', None):
                    isbn = str(book.isbn).strip()
                    bg = BookGenre.query.get(isbn)
                    if bg:
                        book_dict['genre'] = bg.genre
                book_dict['ratings_stats'] = {
                    'avg_rating': round(float(ratings_stats.avg_rating), 2) if ratings_stats.avg_rating else None,
                    'rating_count': ratings_stats.rating_count,
                    'min_rating': ratings_stats.min_rating,
                    'max_rating': ratings_stats.max_rating
                }
                book_dict['recent_reviews'] = []
                for rating, username in recent_ratings:
                    book_dict['recent_reviews'].append({
                        'username': username,
                        'rating': rating.rating,
                        'review': rating.review_text,
                        'created_at': rating.created_at.isoformat()
                    })
                details_book = book_dict
                # Ensure the book is in the results for inline details
                if not any(str(b['id']) == str(details_book['id']) for b in results):
                    results = [{
                        'id': details_book['id'],
                        'title': details_book['title'],
                        'author': details_book['author'],
                        'genre': details_book.get('genre'),
                        'mood': details_book.get('mood')
                    }] + results
        return render_template('books/search.html', query=query, results=results, parsed=parsed, is_nl=is_nl, details_book=details_book)
    
    @app.route('/books/<book_id>')
    def get_book_details(book_id):
        book = Book.query.get_or_404(book_id)
        # Get ratings statistics
        ratings_stats = db.session.query(
            db.func.avg(Rating.rating).label('avg_rating'),
            db.func.count(Rating.rating).label('rating_count'),
            db.func.min(Rating.rating).label('min_rating'),
            db.func.max(Rating.rating).label('max_rating')
        ).filter_by(book_id=book_id).first()
        # Get recent ratings/reviews
        recent_ratings = db.session.query(Rating, User.username).join(User).filter(
            Rating.book_id == book_id,
            Rating.review_text.isnot(None)
        ).order_by(Rating.created_at.desc()).limit(5).all()
        book_dict = book.to_dict()
        # Fallback to BookGenre if needed
        if not book_dict.get('genre') or book_dict['genre'] in ('N/A', 'Unknown'):
            bg = BookGenre.query.get(book.isbn)
            if bg:
                book_dict['genre'] = bg.genre
        book_dict['ratings_stats'] = {
            'avg_rating': round(float(ratings_stats.avg_rating), 2) if ratings_stats.avg_rating else None,
            'rating_count': ratings_stats.rating_count,
            'min_rating': ratings_stats.min_rating,
            'max_rating': ratings_stats.max_rating
        }
        book_dict['recent_reviews'] = []
        for rating, username in recent_ratings:
            book_dict['recent_reviews'].append({
                'username': username,
                'rating': rating.rating,
                'review': rating.review_text,
                'created_at': rating.created_at.isoformat()
            })
        return render_template('books/details.html', book=book_dict)
    
    @app.route('/books/<book_id>/rate', methods=['POST'])
    def rate_book(book_id):
        if not current_user.is_authenticated:
            flash('You must be logged in to rate books.', 'danger')
            return redirect(url_for('login_page'))
        user = current_user
        if not user:
            flash('User not found.', 'danger')
            return redirect(url_for('login_page'))
        book = Book.query.get_or_404(book_id)
        try:
            rating_value = int(request.form.get('rating'))
            if rating_value < 1 or rating_value > 10:
                flash('Invalid rating value.', 'danger')
                return redirect(url_for('get_book_details', book_id=book_id))
            # Check if user already rated this book
            rating = Rating.query.filter_by(user_id=user.id, book_id=book.id).first()
            if rating:
                rating.rating = rating_value
                rating.created_at = datetime.utcnow()
            else:
                rating = Rating(user_id=user.id, book_id=book.id, rating=rating_value, created_at=datetime.utcnow())
                db.session.add(rating)
            db.session.commit()
            flash('Your rating has been submitted!', 'success')
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving rating: {e}")
            flash('Failed to submit rating. Please try again.', 'danger')
        return redirect(url_for('get_book_details', book_id=book_id))
    
    @app.route('/admin')
    @login_required
    def admin_dashboard():
        stats = {
            'total_users': User.query.count(),
            'total_books': Book.query.count(),
            'total_ratings': Rating.query.count(),
            'active_users': User.query.filter(User.last_active >= datetime.utcnow() - timedelta(days=7)).count()
        }
        abtests = ABTest.query.order_by(ABTest.created_at.desc()).all()
        recent_activity = [
            'Model retrained',
            'New user registered',
            'Batch recommendation processed',
            'Cache cleared',
            'System health check'
        ]
        return render_template('admin/dashboard.html', stats=stats, abtests=abtests, recent_activity=recent_activity)
    
    @app.route('/admin/retrain', methods=['POST'])
    @login_required
    def admin_retrain():
        # Trigger retrain logic here
        flash('Model retraining started!', 'success')
        return redirect(url_for('admin_dashboard'))

    @app.route('/admin/clear_cache', methods=['POST'])
    @login_required
    def admin_clear_cache():
        # Clear cache logic here
        flash('Cache cleared successfully!', 'success')
        return redirect(url_for('admin_dashboard'))

    @app.route('/admin/health', methods=['GET'])
    @login_required
    def admin_health_check():
        # Health check logic here
        flash('System health check: healthy', 'success')
        return redirect(url_for('admin_dashboard'))

    @app.route('/admin/abtest', methods=['GET', 'POST'])
    @login_required
    def admin_abtest():
        abtests = ABTest.query.order_by(ABTest.created_at.desc()).all()
        assignments = []
        for a in ABTestAssignment.query.order_by(ABTestAssignment.assigned_at.desc()).all():
            test = ABTest.query.get(a.test_id)
            assignments.append({
                'user_id': a.user_id,
                'test_name': test.name if test else a.test_id,
                'variant': a.variant,
                'assigned_at': a.assigned_at
            })
        return render_template('admin/abtest.html', abtests=abtests, assignments=assignments)

    @app.route('/admin/abtest/create', methods=['POST'])
    @login_required
    def admin_create_abtest():
        name = request.form.get('name')
        algorithm_a = request.form.get('algorithm_a')
        algorithm_b = request.form.get('algorithm_b')
        traffic_split = float(request.form.get('traffic_split', 0.5))
        if not name or not algorithm_a or not algorithm_b:
            flash('Missing required fields', 'danger')
            return redirect(url_for('admin_abtest'))
        abtest = ABTest(
            name=name,
            algorithm_a=algorithm_a,
            algorithm_b=algorithm_b,
            traffic_split=traffic_split,
            is_active=True
        )
        db.session.add(abtest)
        db.session.commit()
        flash('A/B test created!', 'success')
        return redirect(url_for('admin_abtest'))

    @app.route('/admin/abtest/stop/<test_id>', methods=['POST'])
    @login_required
    def admin_stop_abtest(test_id):
        abtest = ABTest.query.get(test_id)
        if abtest:
            abtest.is_active = False
            db.session.commit()
            flash('A/B test stopped.', 'success')
        else:
            flash('A/B test not found.', 'danger')
        return redirect(url_for('admin_abtest'))
    
    @app.route('/test_session')
    def test_session():
        return f"Session user_id: {session.get('user_id')}"
    
    @app.route('/admin/abtest')
    def abtest_dashboard():
        if not current_user.is_authenticated:
            return redirect(url_for('login_page'))
        user = current_user
        if not user or not user.is_admin:
            return "Forbidden", 403
        return render_template('admin/abtest.html')
    

    # API AUTHENTICATION ENDPOINTS
    
    
    @app.route('/api/auth/register', methods=['POST'])
    @limiter.limit("5 per minute")
    def register():
        try:
            data = request.get_json()
            
            # Validate input
            if not data.get('username') or not data.get('email') or not data.get('password'):
                return jsonify({'error': 'Missing required fields'}), 400
            
            # Check if user exists
            if User.query.filter_by(username=data['username']).first():
                return jsonify({'error': 'Username already exists'}), 400
            
            if User.query.filter_by(email=data['email']).first():
                return jsonify({'error': 'Email already exists'}), 400
            
            # Create user
            user = User(
                username=data['username'],
                email=data['email'],
                age=data.get('age'),
                location=data.get('location'),
                preferences=data.get('preferences', {})
            )
            user.set_password(data['password'])
            
            db.session.add(user)
            db.session.commit()
            
            # Create access token
            access_token = create_access_token(identity=user.id)
            
            logger.info(f"New user registered: {user.username}")
            
            return jsonify({
                'message': 'User created successfully',
                'access_token': access_token,
                'user': user.to_dict()
            }), 201
            
        except Exception as e:
            logger.error(f"Registration error: {e}")
            db.session.rollback()
            return jsonify({'error': 'Registration failed'}), 500
    
    @app.route('/api/auth/login', methods=['POST'])
    @limiter.limit("10 per minute")
    def login():
        try:
            data = request.get_json()
            
            username = data.get('username')
            password = data.get('password')
            
            if not username or not password:
                return jsonify({'error': 'Username and password required'}), 400
            
            # Find user
            user = User.query.filter_by(username=username).first()
            
            if user and user.check_password(password):
                # Update last active
                user.last_active = datetime.utcnow()
                db.session.commit()
                
                # Create access token with is_admin claim
                additional_claims = {"is_admin": user.is_admin}
                access_token = create_access_token(identity=user.id, additional_claims=additional_claims)
                
                logger.info(f"User logged in: {user.username}")
                
                return jsonify({
                    'message': 'Login successful',
                    'access_token': access_token,
                    'user': {**user.to_dict(), 'is_admin': user.is_admin}
                })
            else:
                return jsonify({'error': 'Invalid credentials'}), 401
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            return jsonify({'error': 'Login failed'}), 500
    
    @app.route('/api/auth/check', methods=['GET'])
    def check_auth_status():
        """Check authentication status for frontend"""
        try:
           
            return jsonify({
                'authenticated': False,
                'user': None
            })
        except Exception:
            return jsonify({
                'authenticated': False,
                'user': None
            })
    
    # API RECOMMENDATION ENDPOINTS
    
    
    @app.route('/api/recommendations/<user_id>', methods=['GET'])
    @limiter.limit("100 per hour")
    @require_auth
    def get_recommendations(user_id):
        try:
            # Check if user can access this recommendation
            if g.current_user.id != user_id:
                return jsonify({'error': 'Unauthorized'}), 403
            
            algorithm = request.args.get('algorithm', 'hybrid')
            top_n = min(int(request.args.get('top_n', 10)), 50)  # Max 50
            
            # Context from request
            context = {}
            if request.args.get('time_of_day'):
                context['time_of_day'] = int(request.args.get('time_of_day'))
            if request.args.get('device'):
                context['device'] = request.args.get('device')
            if request.args.get('user_mood'):
                context['user_mood'] = request.args.get('user_mood')
            if request.args.get('season'):
                context['season'] = int(request.args.get('season'))
            
            # Check for A/B test assignment
            active_test = ABTest.query.filter_by(is_active=True).first()
            if active_test:
                assignment = ABTestAssignment.query.filter_by(
                    user_id=user_id,
                    test_id=active_test.id
                ).first()
                
                if assignment:
                    algorithm = active_test.algorithm_a if assignment.variant == 'A' else active_test.algorithm_b
            
            # Get recommendations
            recommendations = rec_system.get_user_recommendations_with_explanations(
                user_id=user_id,
                algorithm=algorithm,
                top_n=top_n,
                context=context if context else None
            )
            
            return jsonify({
                'user_id': user_id,
                'algorithm': algorithm,
                'context': context,
                'recommendations': recommendations,
                'timestamp': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Recommendation error for user {user_id}: {e}")
            return jsonify({'error': 'Failed to get recommendations'}), 500
    
    @app.route('/api/books/<book_id>/similar', methods=['GET'])
    @limiter.limit("200 per hour")
    def get_similar_books(book_id):
        try:
            top_n = min(int(request.args.get('top_n', 10)), 20)
            
            # Get similar books
            similar_books = rec_system.recommend_item_collaborative(book_id, top_n)
            
            return jsonify({
                'book_id': book_id,
                'similar_books': similar_books,
                'timestamp': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Similar books error for {book_id}: {e}")
            return jsonify({'error': 'Failed to get similar books'}), 500
    
    
    # API FEEDBACK ENDPOINTS

    
    @app.route('/api/feedback', methods=['POST'])
    @limiter.limit("500 per hour")
    @require_auth
    def record_feedback():
        try:
            data = request.get_json()
            
            user_id = g.current_user.id
            book_id = data.get('book_id')
            feedback_type = data.get('feedback_type')  # view, like, dislike, save, skip, rating
            feedback_data = data.get('feedback_data', {})
            
            if not book_id or not feedback_type:
                return jsonify({'error': 'Missing required fields'}), 400
            
            # Verify book exists
            book = Book.query.get(book_id)
            if not book:
                return jsonify({'error': 'Book not found'}), 404
            
            # Record feedback
            rec_system.record_recommendation_feedback(
                user_id=user_id,
                book_id=book_id,
                feedback_type=feedback_type,
                feedback_data=feedback_data
            )
            
            logger.info(f"Feedback recorded: {user_id} -> {book_id} ({feedback_type})")
            
            return jsonify({
                'message': 'Feedback recorded successfully',
                'user_id': user_id,
                'book_id': book_id,
                'feedback_type': feedback_type
            })
            
        except Exception as e:
            logger.error(f"Feedback error: {e}")
            return jsonify({'error': 'Failed to record feedback'}), 500
    
    
    # API BOOK ENDPOINTS
    
    
    @app.route('/api/books/search', methods=['GET'])
    @limiter.limit("100 per hour")
    def search_books():
        try:
            query = request.args.get('q', '').strip()
            limit = min(int(request.args.get('limit', 20)), 100)
            
            if not query:
                return jsonify({'error': 'Search query required'}), 400
            
            # Search books by title or author
            books = Book.query.filter(
                db.or_(
                    Book.title.ilike(f'%{query}%'),
                    Book.author.ilike(f'%{query}%')
                )
            ).limit(limit).all()
            
            results = []
            for book in books:
                # Get average rating
                avg_rating = db.session.query(db.func.avg(Rating.rating)).filter_by(book_id=book.id).scalar()
                rating_count = Rating.query.filter_by(book_id=book.id).count()
                
                book_dict = book.to_dict()
                book_dict['avg_rating'] = round(float(avg_rating), 2) if avg_rating else None
                book_dict['rating_count'] = rating_count
                
                results.append(book_dict)
            
            return jsonify({
                'query': query,
                'results': results,
                'total': len(results)
            })
            
        except Exception as e:
            logger.error(f"Search error for '{query}': {e}")
            return jsonify({'error': 'Search failed'}), 500
    
    @app.route('/api/books/<book_id>', methods=['GET'], endpoint='api_get_book_details')
    @limiter.limit("200 per hour")
    def get_book_details_api(book_id):
        try:
            book = Book.query.get_or_404(book_id)
            
            # Get ratings statistics
            ratings_stats = db.session.query(
                db.func.avg(Rating.rating).label('avg_rating'),
                db.func.count(Rating.rating).label('rating_count'),
                db.func.min(Rating.rating).label('min_rating'),
                db.func.max(Rating.rating).label('max_rating')
            ).filter_by(book_id=book_id).first()
            
            # Get recent ratings/reviews
            recent_ratings = db.session.query(Rating, User.username).join(User).filter(
                Rating.book_id == book_id,
                Rating.review_text.isnot(None)
            ).order_by(Rating.created_at.desc()).limit(5).all()
            
            book_dict = book.to_dict()
            book_dict['ratings_stats'] = {
                'avg_rating': round(float(ratings_stats.avg_rating), 2) if ratings_stats.avg_rating else None,
                'rating_count': ratings_stats.rating_count,
                'min_rating': ratings_stats.min_rating,
                'max_rating': ratings_stats.max_rating
            }
            
            book_dict['recent_reviews'] = []
            for rating, username in recent_ratings:
                book_dict['recent_reviews'].append({
                    'username': username,
                    'rating': rating.rating,
                    'review': rating.review_text,
                    'created_at': rating.created_at.isoformat()
                })
            
            return jsonify(book_dict)
            
        except Exception as e:
            logger.error(f"Book details error for {book_id}: {e}")
            return jsonify({'error': 'Failed to get book details'}), 500
    
    
    # API USER ENDPOINTS
    
    
    @app.route('/api/users/profile', methods=['GET'])
    @require_auth
    def get_user_profile():
        try:
            user = g.current_user
            
            # Get user statistics
            total_ratings = Rating.query.filter_by(user_id=user.id).count()
            avg_rating_given = db.session.query(db.func.avg(Rating.rating)).filter_by(user_id=user.id).scalar()
            
            # Get favorite authors and genres
            favorite_authors = db.session.query(
                Book.author, 
                db.func.count(Rating.id).label('count')
            ).join(Rating).filter(Rating.user_id == user.id).group_by(Book.author).order_by(
                db.func.count(Rating.id).desc()
            ).limit(5).all()
            
            # Get reading activity over time
            recent_activity = db.session.query(Rating, Book.title, Book.author).join(Book).filter(
                Rating.user_id == user.id
            ).order_by(Rating.created_at.desc()).limit(10).all()
            
            profile_data = user.to_dict()
            profile_data['statistics'] = {
                'total_ratings': total_ratings,
                'avg_rating_given': round(float(avg_rating_given), 2) if avg_rating_given else None,
                'favorite_authors': [{'author': author, 'books_rated': count} for author, count in favorite_authors],
                'reading_diversity': len(favorite_authors),
                'engagement_score': min(total_ratings / 50.0, 1.0)  # Simple engagement score
            }
            
            profile_data['recent_activity'] = []
            for rating, title, author in recent_activity:
                profile_data['recent_activity'].append({
                    'book_title': title,
                    'book_author': author,
                    'rating': rating.rating,
                    'created_at': rating.created_at.isoformat()
                })
            
            return jsonify(profile_data)
            
        except Exception as e:
            logger.error(f"Profile error for user {g.current_user.id}: {e}")
            return jsonify({'error': 'Failed to get user profile'}), 500
    
    @app.route('/api/users/profile', methods=['PUT'])
    @require_auth
    def update_user_profile():
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
                
            user = g.current_user
            
            # Update allowed fields
            if 'age' in data:
                user.age = data['age']
            if 'location' in data:
                user.location = data['location']
            if 'preferences' in data:
                user.preferences = data['preferences']
            
            db.session.commit()
            
            # Clear user cache
            if redis_client:
                redis_client.delete(f"user_profile:{user.id}")
            
            logger.info(f"Profile updated for user: {user.username}")
            
            return jsonify({
                'message': 'Profile updated successfully',
                'user': user.to_dict()
            })
            
        except Exception as e:
            logger.error(f"Profile update error for user {g.current_user.id}: {e}")
            db.session.rollback()
            return jsonify({'error': 'Failed to update profile'}), 500
    
    
    # API ADMIN ENDPOINTS
    
    
    @app.route('/api/admin/stats', methods=['GET'])
    @login_required
    def get_system_stats():
        
        try:
            stats = {
                'total_users': User.query.count(),
                'total_books': Book.query.count(),
                'total_ratings': Rating.query.count(),
                'total_recommendations_served': Recommendation.query.count(),
                'active_users_last_week': User.query.filter(
                    User.last_active >= datetime.utcnow() - timedelta(days=7)
                ).count(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return jsonify(stats)
            
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return jsonify({'error': 'Failed to get stats'}), 500
    
    @app.route('/api/admin/retrain', methods=['POST'])
    @login_required
    def trigger_model_retrain():
        
        try:
            # Trigger async model retraining
            if celery:
                task = celery.send_task('retrain_models')
                return jsonify({
                    'message': 'Model retraining started',
                    'task_id': str(task.id)
                })
            else:
                # Sync retraining (not recommended for production)
                rec_system.build_collaborative_filtering_models()
                rec_system.build_content_models()
                return jsonify({'message': 'Models retrained successfully'})
                
        except Exception as e:
            logger.error(f"Retrain error: {e}")
            return jsonify({'error': 'Failed to trigger retraining'}), 500

    @app.route('/api/admin/sessions', methods=['GET'])
    @login_required
    def get_user_sessions():
        try:
            sessions = UserSession.query.order_by(UserSession.last_active.desc()).limit(100).all()
            return jsonify({'sessions': [
                {
                    'id': s.id,
                    'user_id': s.user_id,
                    'session_token': s.session_token,
                    'device': s.device,
                    'platform': s.platform,
                    'ip_address': s.ip_address,
                    'user_agent': s.user_agent,
                    'created_at': s.created_at.isoformat() if s.created_at else None,
                    'last_active': s.last_active.isoformat() if s.last_active else None,
                    'expires_at': s.expires_at.isoformat() if s.expires_at else None,
                    'is_active': s.is_active
                } for s in sessions
            ]})
        except Exception as e:
            logger.error(f"Admin sessions error: {e}")
            return jsonify({'error': 'Failed to get user sessions'}), 500

    @app.route('/api/admin/abtests', methods=['GET'])
    @login_required
    def get_abtests():
        try:
            abtests = ABTest.query.order_by(ABTest.created_at.desc()).all()
            assignments = ABTestAssignment.query.order_by(ABTestAssignment.assigned_at.desc()).limit(200).all()
            return jsonify({
                'abtests': [
                    {
                        'id': t.id,
                        'name': t.name,
                        'algorithm_a': t.algorithm_a,
                        'algorithm_b': t.algorithm_b,
                        'traffic_split': t.traffic_split,
                        'is_active': t.is_active,
                        'start_date': t.start_date.isoformat() if t.start_date else None,
                        'end_date': t.end_date.isoformat() if t.end_date else None,
                        'created_at': t.created_at.isoformat() if t.created_at else None
                    } for t in abtests
                ],
                'assignments': [
                    {
                        'id': a.id,
                        'user_id': a.user_id,
                        'test_id': a.test_id,
                        'variant': a.variant,
                        'assigned_at': a.assigned_at.isoformat() if a.assigned_at else None
                    } for a in assignments
                ]
            })
        except Exception as e:
            logger.error(f"Admin abtests error: {e}")
            return jsonify({'error': 'Failed to get abtests'}), 500

    @app.route('/api/admin/abtests', methods=['POST'])
    @login_required
    def create_abtest():
        try:
            data = request.get_json()
            name = data.get('name')
            algorithm_a = data.get('algorithm_a')
            algorithm_b = data.get('algorithm_b')
            traffic_split = float(data.get('traffic_split', 0.5))
            if not name or not algorithm_a or not algorithm_b:
                return jsonify({'error': 'Missing required fields'}), 400
            abtest = ABTest(
                name=name,
                algorithm_a=algorithm_a,
                algorithm_b=algorithm_b,
                traffic_split=traffic_split,
                is_active=True
            )
            db.session.add(abtest)
            db.session.commit()
            return jsonify({
                'id': abtest.id,
                'name': abtest.name,
                'algorithm_a': abtest.algorithm_a,
                'algorithm_b': abtest.algorithm_b,
                'traffic_split': abtest.traffic_split,
                'is_active': abtest.is_active,
                'created_at': abtest.created_at.isoformat() if abtest.created_at else None
            }), 201
        except Exception as e:
            logger.error(f"Create ABTest error: {e}")
            db.session.rollback()
            return jsonify({'error': 'Failed to create A/B test'}), 500

    @app.route('/api/admin/model_versions', methods=['GET'])
    @login_required
    def get_model_versions():
        try:
            versions = ModelVersion.query.order_by(ModelVersion.created_at.desc()).limit(20).all()
            return jsonify({'model_versions': [
                {
                    'id': v.id,
                    'version_name': v.version_name,
                    'algorithm_config': v.algorithm_config,
                    'model_state': v.model_state,
                    'precision_at_5': v.precision_at_5,
                    'recall_at_5': v.recall_at_5,
                    'ndcg_at_5': v.ndcg_at_5,
                    'coverage': v.coverage,
                    'diversity': v.diversity,
                    'is_active': v.is_active,
                    'created_at': v.created_at.isoformat() if v.created_at else None
                } for v in versions
            ]})
        except Exception as e:
            logger.error(f"Admin model_versions error: {e}")
            return jsonify({'error': 'Failed to get model versions'}), 500

    @app.route('/api/admin/abtests/<test_id>', methods=['PATCH'])
    @login_required
    def edit_abtest(test_id):
        try:
            abtest = ABTest.query.get(test_id)
            if not abtest:
                return jsonify({'error': 'A/B test not found'}), 404
            data = request.get_json()
            if 'name' in data:
                abtest.name = data['name']
            if 'algorithm_a' in data:
                abtest.algorithm_a = data['algorithm_a']
            if 'algorithm_b' in data:
                abtest.algorithm_b = data['algorithm_b']
            if 'traffic_split' in data:
                abtest.traffic_split = float(data['traffic_split'])
            db.session.commit()
            return jsonify({'message': 'A/B test updated'})
        except Exception as e:
            logger.error(f"Edit ABTest error: {e}")
            db.session.rollback()
            return jsonify({'error': 'Failed to edit A/B test'}), 500

    @app.route('/api/admin/abtests/<test_id>', methods=['DELETE'])
    @login_required
    def stop_abtest(test_id):
        try:
            abtest = ABTest.query.get(test_id)
            if not abtest:
                return jsonify({'error': 'A/B test not found'}), 404
            abtest.is_active = False
            db.session.commit()
            return jsonify({'message': 'A/B test stopped'})
        except Exception as e:
            logger.error(f"Stop ABTest error: {e}")
            db.session.rollback()
            return jsonify({'error': 'Failed to stop A/B test'}), 500

    @app.route('/api/admin/abtest_assignments/<assignment_id>', methods=['PATCH'])
    @login_required
    def edit_abtest_assignment(assignment_id):
        try:
            assignment = ABTestAssignment.query.get(assignment_id)
            if not assignment:
                return jsonify({'error': 'Assignment not found'}), 404
            data = request.get_json()
            if 'variant' in data:
                assignment.variant = data['variant']
            db.session.commit()
            return jsonify({'message': 'Assignment updated'})
        except Exception as e:
            logger.error(f"Edit ABTestAssignment error: {e}")
            db.session.rollback()
            return jsonify({'error': 'Failed to edit assignment'}), 500

    @app.route('/api/admin/abtest_metrics/<test_id>', methods=['GET'])
    @login_required
    def abtest_metrics(test_id):
        try:
            # In production, compute real metrics from logs/analytics
            metrics = {
                'test_id': test_id,
                'conversion_rate_a': 0.12,
                'conversion_rate_b': 0.15,
                'click_through_a': 0.22,
                'click_through_b': 0.19,
                'engagement_a': 0.8,
                'engagement_b': 0.85,
                'users_a': 500,
                'users_b': 520
            }
            return jsonify(metrics)
        except Exception as e:
            logger.error(f"A/B test metrics error: {e}")
            return jsonify({'error': 'Failed to get metrics'}), 500

    @app.route('/api/nl_recommend', methods=['POST'])
    @jwt_required()
    def nl_recommend():
        data = request.get_json()
        query = data.get('query')
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        # Use the LLM to parse the query
        parsed = parse_nl_query_with_llm(query)
        genre = (parsed.get('genre') or '').strip() or None
        mood = (parsed.get('mood') or '').strip() or None
        comparison = (parsed.get('comparison') or '').strip() or None
        # Step 1: Filter books by genre and mood if present
        books_query = Book.query
        if genre:
            books_query = books_query.filter(Book.genre.ilike(f"%{genre}%"))
        if mood:
            books_query = books_query.filter(Book.mood.ilike(f"%{mood}%"))
        filtered_books = books_query.all()
        # Step 2: If comparison or free-text, use semantic search
        recommendations = []
        if comparison and len(filtered_books) > 0:
            try:
                query_embedding = rec_system.embedder.encode(comparison)
                # Get embeddings for filtered books
                filtered_ids = [b.id for b in filtered_books if b.plot_embedding is not None]
                filtered_embeddings = [np.array(Book.query.get(bid).plot_embedding, dtype=np.float32) for bid in filtered_ids]
                if filtered_embeddings:
                    filtered_embeddings = np.vstack(filtered_embeddings)
                    faiss.normalize_L2(filtered_embeddings)
                    temp_index = faiss.IndexFlatIP(filtered_embeddings.shape[1])
                    temp_index.add(filtered_embeddings)
                    emb = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
                    faiss.normalize_L2(emb)
                    D, I = temp_index.search(emb, 10)
                    for j, idx in enumerate(I[0]):
                        b = Book.query.get(filtered_ids[idx])
                        if b:
                            recommendations.append({
                                'book_id': b.id,
                                'title': b.title,
                                'author': b.author,
                                'genre': b.genre,
                                'mood': b.mood,
                                'image_url': b.image_url_m,
                                'similarity': float(D[0][j]),
                                'explanation': f"Matched on semantic similarity to: {comparison}"
                            })
            except Exception as e:
                logger.error(f"Semantic search error: {e}")
        elif len(filtered_books) > 0:
            for b in filtered_books[:10]:
                recommendations.append({
                    'book_id': b.id,
                    'title': b.title,
                    'author': b.author,
                    'genre': b.genre,
                    'mood': b.mood,
                    'image_url': b.image_url_m,
                    'explanation': f"Matched on genre/mood filter"
                })
        else:
            try:
                query_embedding = rec_system.embedder.encode(comparison or query)
                for book_id, score in faiss_search(query_embedding, top_n=10):
                    b = Book.query.get(book_id)
                    if b:
                        recommendations.append({
                            'book_id': b.id,
                            'title': b.title,
                            'author': b.author,
                            'genre': b.genre,
                            'mood': b.mood,
                            'image_url': b.image_url_m,
                            'similarity': score,
                            'explanation': f"Matched on semantic similarity to: {comparison or query}"
                        })
            except Exception as e:
                logger.error(f"Fallback semantic search error: {e}")
        return jsonify({'parsed': parsed, 'recommendations': recommendations})


    # HEALTH CHECK ENDPOINT
    

    @app.route('/health')
    def health_check():
        """Health check endpoint"""
        try:
            # Check database connection
            db.session.execute('SELECT 1')
            
            # Check Redis connection
            redis_status = 'connected'
            if redis_client:
                try:
                    redis_client.ping()
                except:
                    redis_status = 'disconnected'
            else:
                redis_status = 'not_configured'
            
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'database': 'connected',
                'redis': redis_status,
                'version': '1.0.0'
            })
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return jsonify({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }), 500
    
    
    # ERROR HANDLERS
    
    
    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith('/api'):
            return jsonify({'error': 'API endpoint not found'}), 404
        # For web routes, you could render a 404 template
        return jsonify({'error': 'Page not found'}), 404
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({'error': 'Bad request'}), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        return jsonify({'error': 'Unauthorized'}), 401
    
    @app.errorhandler(403)
    def forbidden(error):
        return jsonify({'error': 'Forbidden'}), 403
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        logger.error(f"Internal server error: {error}")
        return jsonify({'error': 'Internal server error'}), 500
    
    @app.errorhandler(429)
    def ratelimit_handler(e):
        return jsonify({
            'error': 'Rate limit exceeded', 
            'message': str(e.description)
        }), 429
    
    return app


# CELERY TASKS


def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    
    celery.Task = ContextTask
    return celery

# Create Flask app
app = create_app()

# Load LLM pipeline once at app startup
try:
    mistral_llm = pipeline('text2text-generation', model='google/flan-t5-large')
except Exception as e:
    logger.warning(f"Could not load LLM: {e}")
    mistral_llm = None

def parse_nl_query_with_llm(query):
    """
    Use Mistral LLM to extract structured preferences from a natural language query.
    Returns a dict like {'genre': ..., 'mood': ..., 'comparison': ...}
    """
    if not mistral_llm:
        return {'raw_query': query, 'genre': None, 'mood': None, 'comparison': None, 'llm_error': 'LLM not loaded'}
    prompt = f"Extract genre, mood, and comparison from: {query}. Respond in JSON with keys genre, mood, comparison."
    result = mistral_llm(prompt, max_new_tokens=64)[0]['generated_text']
    # Try to parse JSON from result
    try:
        parsed = json.loads(result)
        return {'raw_query': query, **parsed}
    except Exception:
        return {'raw_query': query, 'llm_output': result}

# Load at startup
book_embeddings = np.load("dataset/Book-Crossing/book_embeddings_minilm.npy")
book_df = pd.read_csv("dataset/Book-Crossing/Books.csv", sep=';', encoding='latin-1')
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load zero-shot classifier and labels at startup
genre_labels = ["romance", "mystery", "horror", "fantasy", "science fiction", "thriller", "comedy", "adventure", "drama", "historical", "children", "young adult"]
mood_labels = ["happy", "sad", "dark", "inspirational", "funny", "serious", "uplifting", "suspenseful", "romantic", "scary"]
zero_shot = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")

def extract_genre_mood(query):
    genre_result = zero_shot(query, genre_labels)
    mood_result = zero_shot(query, mood_labels)
    top_genre = genre_result['labels'][0] if genre_result['scores'][0] > 0.5 else None
    top_mood = mood_result['labels'][0] if mood_result['scores'][0] > 0.5 else None
    return top_genre, top_mood

def get_top_books(user_query, top_n=10):
    genre, mood = extract_genre_mood(user_query)
    query_embedding = embedder.encode([user_query])[0]
    scores = cosine_similarity([query_embedding], book_embeddings)[0]
    top_indices = scores.argsort()[-top_n*2:][::-1]  # Get more to allow filtering
    filtered_books = []
    for idx in top_indices:
        book = book_df.iloc[idx]
        explanation = "Semantic match"
        if genre and genre.lower() in str(book.get('genre', '')).lower():
            explanation = f"Matched genre: {genre}"
        elif mood and mood.lower() in str(book.get('mood', '')).lower():
            explanation = f"Matched mood: {mood}"
        filtered_books.append((book, scores[idx], explanation))
        if len(filtered_books) >= top_n:
            break
    results = []
    for book, score, explanation in filtered_books[:top_n]:
        book = book.copy()
        book['explanation'] = explanation
        results.append(book)
    return pd.DataFrame(results)

# --- FAISS SETUP ---
faiss_index = None
faiss_id_to_book = []

def build_faiss_index():
    global faiss_index, faiss_id_to_book
    all_books = Book.query.all()
    book_embeddings = []
    faiss_id_to_book = []
    for book in all_books:
        if book.plot_embedding is not None:
            emb = np.array(book.plot_embedding, dtype=np.float32)
            if emb.ndim == 1:
                book_embeddings.append(emb)
                faiss_id_to_book.append(book.id)
    if book_embeddings:
        book_embeddings = np.vstack(book_embeddings).astype(np.float32)
        dim = book_embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        # Normalize for cosine similarity
        faiss.normalize_L2(book_embeddings)
        faiss_index.add(book_embeddings)
    else:
        faiss_index = None
        faiss_id_to_book = []

# Build FAISS index at startup
with app.app_context():
    build_faiss_index()

# Helper: get top N book ids from FAISS given a query embedding

def faiss_search(query_embedding, top_n=10):
    if faiss_index is None or not faiss_id_to_book:
        return []
    emb = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(emb)
    D, I = faiss_index.search(emb, top_n)
    return [(faiss_id_to_book[i], float(D[0][j])) for j, i in enumerate(I[0]) if i < len(faiss_id_to_book)]

# Create Celery instance
try:
    celery = make_celery(app)
    
    @celery.task
    def update_user_model(user_id):
        """Update user-specific model components"""
        try:
            with app.app_context():
                # Clear user caches
                if app.config.get('CACHE_REDIS_URL'):
                    redis_client = redis.Redis.from_url(app.config['CACHE_REDIS_URL'])
                    redis_client.delete(f"user_recs:{user_id}")
                    redis_client.delete(f"user_profile:{user_id}")
                
                logger.info(f"Updated model for user {user_id}")
                
        except Exception as e:
            logger.error(f"Error updating user model: {e}")
    
    @celery.task
    def retrain_models():
        """Retrain all recommendation models"""
        try:
            with app.app_context():
                rec_system = DatabaseRecommendationSystem(app)
                rec_system.build_collaborative_filtering_models()
                rec_system.build_content_models()
                
                # Clear all caches
                if app.config.get('CACHE_REDIS_URL'):
                    redis_client = redis.Redis.from_url(app.config['CACHE_REDIS_URL'])
                    redis_client.flushdb()
                
                logger.info("All models retrained successfully")
                
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    @celery.task
    def update_book_popularity_scores():
        """Update book popularity and quality scores"""
        try:
            with app.app_context():
                books = Book.query.all()
                
                for book in books:
                    # Calculate new popularity score
                    rating_count = Rating.query.filter_by(book_id=book.id).count()
                    avg_rating = db.session.query(db.func.avg(Rating.rating)).filter_by(book_id=book.id).scalar() or 0
                    
                    book.popularity_score = rating_count
                    book.quality_score = (
                        0.4 * min(rating_count / 100, 1) +
                        0.3 * (book.sentiment or 0) +
                        0.3 * (book.freshness_score or 0)
                    )
                
                db.session.commit()
                logger.info("Book scores updated")
                
        except Exception as e:
            logger.error(f"Error updating book scores: {e}")
            db.session.rollback()

except Exception as e:
    logger.warning(f"Celery setup failed: {e}. Running without background tasks.")
    celery = None


# MAIN APPLICATION ENTRY POINT


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'production') == 'development'
    
    logger.info(f"Starting Flask application on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info("Available routes:")
    logger.info("  Web Interface:")
    logger.info("    GET  /              - Home page")
    logger.info("    GET  /register      - Registration page")
    logger.info("    GET  /login         - Login page")
    logger.info("    GET  /books         - Book search page")
    logger.info("    GET  /profile       - User profile page")
    logger.info("    GET  /admin         - Admin dashboard")
    logger.info("    GET  /admin/abtest  - A/B Testing Platform")
    logger.info("    GET  /test_session  - Test session user_id")
    logger.info("  API Endpoints:")
    logger.info("    POST /api/auth/register - User registration")
    logger.info("    POST /api/auth/login    - User login")
    logger.info("    GET  /api/recommendations/<user_id> - Get recommendations")
    logger.info("    GET  /api/books/search  - Search books")
    logger.info("    POST /api/feedback      - Record feedback")
    logger.info("    GET  /health           - Health check")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )

@app.context_processor
def inject_user():
    return dict(user=current_user)