from models import db, User, Book, Rating, UserFeedback, Recommendation, ModelVersion, ABTest, ABTestAssignment
from flask import current_app
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import random
from sqlalchemy import func, text
import redis
from celery import Celery
import faiss
try:
    from app import faiss_search, build_faiss_index
except ImportError:
    faiss_search = None
    build_faiss_index = None

logger = logging.getLogger(__name__)

class DatabaseRecommendationSystem:
    """Database-powered recommendation system"""
    
    def __init__(self, app=None, redis_client=None, celery_app=None):
        self.app = app
        self.redis_client = redis_client
        self.celery = celery_app
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # ML Components
        self.user_similarity_matrix = None
        self.book_similarity_matrix = None
        self.tfidf_vectorizer = None
        self.embedding_nn = None
        self.q_table = {}
        
        # Cache keys
        self.CACHE_USER_RECS = "user_recs:{}"
        self.CACHE_BOOK_SIMILAR = "book_similar:{}"
        self.CACHE_USER_PROFILE = "user_profile:{}"
        
    def get_user_ratings_df(self, user_id: str = None) -> pd.DataFrame:
        """Get ratings data as DataFrame"""
        query = db.session.query(
            Rating.user_id,
            Rating.book_id,
            Rating.rating,
            Book.title,
            Book.author,
            Book.mood,
            Book.sentiment,
            User.age,
            User.location
        ).join(Book).join(User)
        
        if user_id:
            query = query.filter(Rating.user_id == user_id)
        
        df = pd.read_sql(query.statement, db.engine)
        return df
    
    def get_books_df(self) -> pd.DataFrame:
        """Get books data as DataFrame"""
        query = db.session.query(Book)
        df = pd.read_sql(query.statement, db.engine)
        
        # Convert embedding JSON back to numpy array
        if 'plot_embedding' in df.columns:
            df['plot_embedding'] = df['plot_embedding'].apply(
                lambda x: np.array(x) if x else np.zeros(384)
            )
        
        return df
    
    def build_collaborative_filtering_models(self):
        """Build CF models from database data"""
        logger.info("Building collaborative filtering models...")
        
        # Get ratings data
        ratings_df = self.get_user_ratings_df()
        
        if ratings_df.empty:
            logger.warning("No ratings data found")
            return
        
        # Filter for active users and popular books
        user_counts = ratings_df['user_id'].value_counts()
        book_counts = ratings_df['book_id'].value_counts()
        
        active_users = user_counts[user_counts >= 5].index
        popular_books = book_counts[book_counts >= 10].index
        
        filtered_df = ratings_df[
            ratings_df['user_id'].isin(active_users) & 
            ratings_df['book_id'].isin(popular_books)
        ]
        
        # Create user-item matrix
        user_item_matrix = filtered_df.pivot_table(
            index='user_id', 
            columns='book_id', 
            values='rating'
        ).fillna(0)
        
        # Calculate similarities
        self.user_similarity_matrix = cosine_similarity(user_item_matrix)
        
        # Item-item similarity
        item_user_matrix = user_item_matrix.T
        self.book_similarity_matrix = cosine_similarity(item_user_matrix)
        
        # Store in cache if Redis available
        if self.redis_client:
            self.redis_client.setex(
                "cf_models_timestamp", 
                86400,  # 24 hours
                datetime.now().isoformat()
            )
        
        logger.info("CF models built successfully")
    
    def build_content_models(self):
        """Build content-based models"""
        logger.info("Building content-based models...")
        
        books_df = self.get_books_df()
        
        if books_df.empty:
            logger.warning("No books data found")
            return
        
        # TF-IDF model
        texts = books_df['title'].fillna('') + " " + books_df['author'].fillna('')
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Embedding model
        embeddings = np.vstack(books_df['plot_embedding'].values)
        self.embedding_nn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
        self.embedding_nn.fit(embeddings)
        
        logger.info("Content models built successfully")
    
    def recommend_user_collaborative(self, user_id: str, top_n: int = 10) -> List[Dict]:
        """User-based collaborative filtering with database"""
        cache_key = self.CACHE_USER_RECS.format(user_id)
        
        # Check cache first
        if self.redis_client:
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        
        try:
            # Get user's ratings
            user_ratings = db.session.query(Rating).filter_by(user_id=user_id).all()
            
            if not user_ratings:
                return self._get_popular_recommendations(top_n)
            
            # Get similar users based on rating patterns
            similar_users_query = text("""
                WITH user_similarity AS (
                    SELECT 
                        r2.user_id,
                        COUNT(*) as common_books,
                        AVG(ABS(r1.rating - r2.rating)) as avg_diff
                    FROM ratings r1
                    JOIN ratings r2 ON r1.book_id = r2.book_id
                    WHERE r1.user_id = :user_id AND r2.user_id != :user_id
                    GROUP BY r2.user_id
                    HAVING COUNT(*) >= 3
                    ORDER BY common_books DESC, avg_diff ASC
                    LIMIT 10
                )
                SELECT DISTINCT
                    b.id,
                    b.title,
                    b.author,
                    b.image_url_m,
                    AVG(r.rating) as avg_rating,
                    COUNT(r.rating) as rating_count
                FROM user_similarity us
                JOIN ratings r ON us.user_id = r.user_id
                JOIN books b ON r.book_id = b.id
                WHERE r.book_id NOT IN (
                    SELECT book_id FROM ratings WHERE user_id = :user_id
                )
                GROUP BY b.id, b.title, b.author, b.image_url_m
                HAVING AVG(r.rating) >= 7
                ORDER BY AVG(r.rating) DESC, COUNT(r.rating) DESC
                LIMIT :top_n
            """)
            
            result = db.session.execute(
                similar_users_query, 
                {'user_id': user_id, 'top_n': top_n}
            )
            
            recommendations = []
            for row in result:
                rec = {
                    'book_id': row[0],
                    'title': row[1],
                    'author': row[2],
                    'image_url': row[3],
                    'predicted_rating': round(row[4], 2),
                    'rating_count': row[5],
                    'algorithm': 'user_collaborative',
                    'explanation': f"Users with similar taste rated this {row[4]:.1f}/10"
                }
                recommendations.append(rec)
            
            # Cache results
            if self.redis_client:
                self.redis_client.setex(cache_key, 3600, json.dumps(recommendations))  # 1 hour
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in user collaborative filtering: {e}")
            return self._get_popular_recommendations(top_n)
    
    def recommend_item_collaborative(self, book_id: str, top_n: int = 10) -> List[Dict]:
        """Item-based collaborative filtering"""
        cache_key = self.CACHE_BOOK_SIMILAR.format(book_id)
        
        # Check cache
        if self.redis_client:
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        
        try:
            # Find books rated by users who also rated this book
            similar_books_query = text("""
                WITH book_users AS (
                    SELECT user_id
                    FROM ratings
                    WHERE book_id = :book_id
                ),
                similar_books AS (
                    SELECT 
                        r.book_id,
                        COUNT(*) as common_users,
                        AVG(r.rating) as avg_rating
                    FROM ratings r
                    JOIN book_users bu ON r.user_id = bu.user_id
                    WHERE r.book_id != :book_id
                    GROUP BY r.book_id
                    HAVING COUNT(*) >= 3
                    ORDER BY COUNT(*) DESC, AVG(r.rating) DESC
                    LIMIT :top_n
                )
                SELECT 
                    b.id,
                    b.title,
                    b.author,
                    b.image_url_m,
                    sb.avg_rating,
                    sb.common_users
                FROM similar_books sb
                JOIN books b ON sb.book_id = b.id
            """)
            
            result = db.session.execute(
                similar_books_query,
                {'book_id': book_id, 'top_n': top_n}
            )
            
            recommendations = []
            base_book = Book.query.get(book_id)
            
            for row in result:
                rec = {
                    'book_id': row[0],
                    'title': row[1],
                    'author': row[2],
                    'image_url': row[3],
                    'avg_rating': round(row[4], 2),
                    'common_users': row[5],
                    'algorithm': 'item_collaborative',
                    'explanation': f"Users who liked '{base_book.title if base_book else 'this book'}' also enjoyed this"
                }
                recommendations.append(rec)
            
            # Cache results
            if self.redis_client:
                self.redis_client.setex(cache_key, 7200, json.dumps(recommendations))  # 2 hours
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in item collaborative filtering: {e}")
            return []
    
    def recommend_content_based(self, user_id: str = None, book_id: str = None, top_n: int = 10) -> list:
        """Content-based recommendations using FAISS for fast similarity search"""
        try:
            if book_id:
                # Find similar books by content
                book = Book.query.get(book_id)
                if not book or not book.plot_embedding:
                    return []
                # Use FAISS for similarity
                if faiss_search:
                    query_embedding = np.array(book.plot_embedding, dtype=np.float32)
                    results = faiss_search(query_embedding, top_n=top_n+1)  # +1 to skip self
                    recommendations = []
                    for book_id2, score in results:
                        if book_id2 == book.id:
                            continue  # skip self
                        b = Book.query.get(book_id2)
                        if b:
                            rec = {
                                'book_id': b.id,
                                'title': b.title,
                                'author': b.author,
                                'image_url': b.image_url_m,
                                'similarity_score': round(score, 3),
                                'algorithm': 'content_based',
                                'explanation': f"Similar content and style to '{book.title}'"
                            }
                            recommendations.append(rec)
                        if len(recommendations) >= top_n:
                            break
                    return recommendations
                else:
                    # fallback to old method
                    book_embedding = np.array(book.plot_embedding).reshape(1, -1)
                    distances, indices = self.embedding_nn.kneighbors(book_embedding, n_neighbors=top_n+1)
                    all_books = Book.query.all()
                    recommendations = []
                    for i, idx in enumerate(indices[0][1:]):
                        similar_book = all_books[idx]
                        similarity_score = 1 - distances[0][i+1]
                        rec = {
                            'book_id': similar_book.id,
                            'title': similar_book.title,
                            'author': similar_book.author,
                            'image_url': similar_book.image_url_m,
                            'similarity_score': round(similarity_score, 3),
                            'algorithm': 'content_based',
                            'explanation': f"Similar content and style to '{book.title}'"
                        }
                        recommendations.append(rec)
                    return recommendations
            elif user_id:
                # Content-based recommendations for user based on their history
                user_ratings = db.session.query(Rating, Book).join(Book).filter(
                    Rating.user_id == user_id,
                    Rating.rating >= 7
                ).order_by(Rating.created_at.desc()).limit(5).all()
                if not user_ratings:
                    return self._get_popular_recommendations(top_n)
                user_embeddings = []
                for rating, book in user_ratings:
                    if book.plot_embedding:
                        user_embeddings.append(np.array(book.plot_embedding, dtype=np.float32))
                if not user_embeddings:
                    return self._get_popular_recommendations(top_n)
                user_profile_embedding = np.mean(user_embeddings, axis=0).reshape(1, -1).astype(np.float32)
                # Use FAISS for similarity
                if faiss_search:
                    results = faiss_search(user_profile_embedding[0], top_n=top_n*2)
                    rated_book_ids = {rating.book_id for rating, _ in user_ratings}
                    recommendations = []
                    for book_id2, score in results:
                        if book_id2 in rated_book_ids:
                            continue
                        b = Book.query.get(book_id2)
                        if b:
                            rec = {
                                'book_id': b.id,
                                'title': b.title,
                                'author': b.author,
                                'image_url': b.image_url_m,
                                'similarity_score': round(score, 3),
                                'algorithm': 'content_based',
                                'explanation': "Matches your reading preferences based on book content"
                            }
                            recommendations.append(rec)
                        if len(recommendations) >= top_n:
                            break
                    return recommendations
                else:
                    # fallback to old method
                    distances, indices = self.embedding_nn.kneighbors(user_profile_embedding, n_neighbors=top_n*2)
                    rated_book_ids = {rating.book_id for rating, _ in user_ratings}
                    all_books = Book.query.all()
                    recommendations = []
                    for i, idx in enumerate(indices[0]):
                        candidate_book = all_books[idx]
                        if candidate_book.id in rated_book_ids:
                            continue
                        if len(recommendations) >= top_n:
                            break
                        similarity_score = 1 - distances[0][i]
                        rec = {
                            'book_id': candidate_book.id,
                            'title': candidate_book.title,
                            'author': candidate_book.author,
                            'image_url': candidate_book.image_url_m,
                            'similarity_score': round(similarity_score, 3),
                            'algorithm': 'content_based',
                            'explanation': "Matches your reading preferences based on book content"
                        }
                        recommendations.append(rec)
                    return recommendations
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {e}")
            return []
    
    def recommend_hybrid(self, user_id: str, top_n: int = 10, 
                        weights: Dict[str, float] = None) -> List[Dict]:
        """Hybrid recommendations combining multiple approaches"""
        if weights is None:
            weights = {
                'collaborative': 0.4,
                'content': 0.3,
                'popularity': 0.2,
                'contextual': 0.1
            }
        
        try:
            all_recommendations = {}
            
            # Get collaborative recommendations
            if weights.get('collaborative', 0) > 0:
                collab_recs = self.recommend_user_collaborative(user_id, top_n * 2)
                for rec in collab_recs:
                    book_id = rec['book_id']
                    score = weights['collaborative'] * (rec.get('predicted_rating', 5) / 10)
                    
                    if book_id not in all_recommendations:
                        all_recommendations[book_id] = {
                            'book_id': book_id,
                            'title': rec['title'],
                            'author': rec['author'],
                            'image_url': rec['image_url'],
                            'total_score': 0,
                            'algorithms': [],
                            'explanations': []
                        }
                    
                    all_recommendations[book_id]['total_score'] += score
                    all_recommendations[book_id]['algorithms'].append('collaborative')
                    all_recommendations[book_id]['explanations'].append(rec['explanation'])
            
            # Get content-based recommendations
            if weights.get('content', 0) > 0:
                content_recs = self.recommend_content_based(user_id=user_id, top_n=top_n * 2)
                for rec in content_recs:
                    book_id = rec['book_id']
                    score = weights['content'] * rec.get('similarity_score', 0.5)
                    
                    if book_id not in all_recommendations:
                        all_recommendations[book_id] = {
                            'book_id': book_id,
                            'title': rec['title'],
                            'author': rec['author'],
                            'image_url': rec['image_url'],
                            'total_score': 0,
                            'algorithms': [],
                            'explanations': []
                        }
                    
                    all_recommendations[book_id]['total_score'] += score
                    all_recommendations[book_id]['algorithms'].append('content')
                    all_recommendations[book_id]['explanations'].append(rec['explanation'])
            
            # Get popularity-based recommendations
            if weights.get('popularity', 0) > 0:
                popular_recs = self._get_popular_recommendations(top_n)
                for i, rec in enumerate(popular_recs):
                    book_id = rec['book_id']
                    score = weights['popularity'] * (len(popular_recs) - i) / len(popular_recs)
                    
                    if book_id not in all_recommendations:
                        all_recommendations[book_id] = {
                            'book_id': book_id,
                            'title': rec['title'],
                            'author': rec['author'],
                            'image_url': rec['image_url'],
                            'total_score': 0,
                            'algorithms': [],
                            'explanations': []
                        }
                    
                    all_recommendations[book_id]['total_score'] += score
                    all_recommendations[book_id]['algorithms'].append('popularity')
                    all_recommendations[book_id]['explanations'].append("Popular among all users")
            
            # Sort by total score and return top N
            sorted_recommendations = sorted(
                all_recommendations.values(),
                key=lambda x: x['total_score'],
                reverse=True
            )
            
            # Format final recommendations
            final_recommendations = []
            for rec in sorted_recommendations[:top_n]:
                final_rec = {
                    'book_id': rec['book_id'],
                    'title': rec['title'],
                    'author': rec['author'],
                    'image_url': rec['image_url'],
                    'score': round(rec['total_score'], 3),
                    'algorithm': 'hybrid',
                    'algorithms_used': list(set(rec['algorithms'])),
                    'explanation': self._combine_explanations(rec['explanations'])
                }
                final_recommendations.append(final_rec)
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Error in hybrid recommendations: {e}")
            return self._get_popular_recommendations(top_n)
    
    def recommend_contextual(self, user_id: str, context: Dict[str, Any], 
                           top_n: int = 10) -> List[Dict]:
        """Context-aware recommendations"""
        try:
            # Get base recommendations
            base_recs = self.recommend_hybrid(user_id, top_n * 2)
            
            if not base_recs:
                return []
            
            # Apply contextual filters
            book_ids = [rec['book_id'] for rec in base_recs]
            books = Book.query.filter(Book.id.in_(book_ids)).all()
            book_dict = {book.id: book for book in books}
            
            # Filter based on context
            filtered_recs = []
            
            for rec in base_recs:
                book = book_dict.get(rec['book_id'])
                if not book:
                    continue
                
                # Time-based filtering
                if context.get('time_of_day'):
                    hour = context['time_of_day']
                    if 6 <= hour < 12 and book.mood != 'inspirational':
                        continue  # Morning - prefer inspirational
                    elif hour >= 22 and book.mood == 'dark':
                        continue  # Late night - avoid dark content
                
                # Mood-based filtering
                if context.get('user_mood'):
                    user_mood = context['user_mood']
                    if user_mood in ['sad', 'stressed'] and book.mood == 'dark':
                        continue
                    elif user_mood in ['happy', 'excited'] and book.mood == 'dark':
                        continue
                
                # Device-based filtering
                if context.get('device') == 'mobile':
                    # On mobile, prefer shorter or more engaging content
                    # This would require additional book metadata
                    pass
                
                # Add context bonus to score
                context_bonus = 0
                if context.get('user_mood') == 'happy' and book.mood == 'inspirational':
                    context_bonus += 0.1
                
                rec['score'] += context_bonus
                rec['context_applied'] = True
                filtered_recs.append(rec)
                
                if len(filtered_recs) >= top_n:
                    break
            
            return filtered_recs
            
        except Exception as e:
            logger.error(f"Error in contextual recommendations: {e}")
            return self.recommend_hybrid(user_id, top_n)
    
    def recommend_cross_domain(self, user_id: str, source_domain: str, target_domain: str, top_n: int = 10) -> list:
        """
        Recommend items from target_domain (e.g., movies) based on user preferences in source_domain (e.g., books).
        This is a placeholder; implement shared embedding logic if both domains are present.
        """
        # Example: if you have both Book and Movie models with plot_embedding
        # 1. Get user's favorite items in source_domain
        # 2. Compute average embedding
        # 3. Find most similar items in target_domain
        # 4. Return top_n results
        return []  # Not implemented yet
    
    def record_recommendation_feedback(self, user_id: str, book_id: str, 
                                     feedback_type: str, feedback_data: Dict = None):
        """Record user feedback on recommendations"""
        try:
            # Update or create feedback record
            feedback = UserFeedback.query.filter_by(
                user_id=user_id,
                book_id=book_id
            ).first()
            
            if not feedback:
                feedback = UserFeedback(
                    user_id=user_id,
                    book_id=book_id
                )
                db.session.add(feedback)
            
            # Update feedback based on type
            if feedback_type == 'view':
                feedback.view_duration = feedback_data.get('duration', 0)
                feedback.click_through = True
            elif feedback_type == 'like':
                feedback.liked = True
                feedback.not_interested = False
            elif feedback_type == 'dislike':
                feedback.liked = False
                feedback.not_interested = True
            elif feedback_type == 'save':
                feedback.saved_for_later = True
            elif feedback_type == 'skip':
                feedback.skip = True
            elif feedback_type == 'rating':
                rating_value = feedback_data.get('rating')
                # Create or update rating record
                rating = Rating.query.filter_by(
                    user_id=user_id,
                    book_id=book_id
                ).first()
                
                if not rating:
                    rating = Rating(
                        user_id=user_id,
                        book_id=book_id,
                        rating=rating_value,
                        device=feedback_data.get('device'),
                        platform=feedback_data.get('platform')
                    )
                    db.session.add(rating)
                else:
                    rating.rating = rating_value
                    rating.updated_at = datetime.utcnow()
            
            # Recalculate feedback score
            feedback.feedback_score = self._calculate_feedback_score(feedback)
            feedback.feedback_category = self._categorize_feedback_score(feedback.feedback_score)
            
            db.session.commit()
            
            # Clear relevant caches
            if self.redis_client:
                cache_patterns = [
                    self.CACHE_USER_RECS.format(user_id),
                    self.CACHE_USER_PROFILE.format(user_id)
                ]
                for pattern in cache_patterns:
                    self.redis_client.delete(pattern)
            
            # Trigger async model retraining if significant feedback
            if self.celery and feedback_type in ['rating', 'like', 'dislike']:
                self.celery.send_task('update_user_model', args=[user_id])
            
            logger.info(f"Feedback recorded: {user_id} -> {book_id} ({feedback_type})")
            
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            db.session.rollback()
    
    def get_user_recommendations_with_explanations(self, user_id: str, 
                                                  algorithm: str = 'hybrid',
                                                  top_n: int = 10,
                                                  context: Dict = None) -> List[Dict]:
        """Get recommendations with detailed explanations"""
        try:
            # Get recommendations based on algorithm
            if algorithm == 'collaborative':
                recommendations = self.recommend_user_collaborative(user_id, top_n)
            elif algorithm == 'content':
                recommendations = self.recommend_content_based(user_id=user_id, top_n=top_n)
            elif algorithm == 'contextual' and context:
                recommendations = self.recommend_contextual(user_id, context, top_n)
            else:  # default to hybrid
                recommendations = self.recommend_hybrid(user_id, top_n)
            
            # Store recommendations in database for tracking
            for i, rec in enumerate(recommendations):
                db_rec = Recommendation(
                    user_id=user_id,
                    book_id=rec['book_id'],
                    algorithm=algorithm,
                    score=rec.get('score', 0),
                    rank=i + 1,
                    explanation=rec.get('explanation', ''),
                    context=context
                )
                db.session.add(db_rec)
            
            db.session.commit()
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations with explanations: {e}")
            return []
    
    def _get_popular_recommendations(self, top_n: int = 10) -> List[Dict]:
        """Get popular books as fallback"""
        try:
            popular_books = db.session.query(
                Book,
                func.avg(Rating.rating).label('avg_rating'),
                func.count(Rating.rating).label('rating_count')
            ).join(Rating).group_by(Book.id).having(
                func.count(Rating.rating) >= 10
            ).order_by(
                func.avg(Rating.rating).desc(),
                func.count(Rating.rating).desc()
            ).limit(top_n).all()
            
            recommendations = []
            for book, avg_rating, rating_count in popular_books:
                rec = {
                    'book_id': book.id,
                    'title': book.title,
                    'author': book.author,
                    'image_url': book.image_url_m,
                    'avg_rating': round(float(avg_rating), 2),
                    'rating_count': rating_count,
                    'algorithm': 'popularity',
                    'explanation': f"Popular book with {rating_count} ratings averaging {avg_rating:.1f}/10"
                }
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting popular recommendations: {e}")
            return []
    
    def _calculate_feedback_score(self, feedback: UserFeedback) -> float:
        """Calculate composite feedback score"""
        score = 0.0
        
        if feedback.liked:
            score += 0.4
        elif feedback.not_interested:
            score -= 0.3
        
        if feedback.completion_rate:
            score += 0.2 * feedback.completion_rate
        
        if feedback.re_read:
            score += 0.2
        
        if feedback.social_share:
            score += 0.1
        
        if feedback.saved_for_later:
            score += 0.1
        
        if feedback.skip:
            score -= 0.2
        
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
    
    def _categorize_feedback_score(self, score: float) -> str:
        """Categorize feedback score"""
        if score >= 0.8:
            return "strong_like"
        elif score >= 0.5:
            return "moderate_like"
        elif score >= 0.3:
            return "neutral"
        else:
            return "dislike"
    
    def _combine_explanations(self, explanations: List[str]) -> str:
        """Combine multiple explanations into one"""
        if not explanations:
            return "Recommended for you"
        
        if len(explanations) == 1:
            return explanations[0]
        
        # Take the most specific explanation
        return explanations[0]