# Advanced Multi-Domain Recommendation System

A production-ready recommendation system built with Flask, PostgreSQL, Redis, and Celery. Supports collaborative filtering, content-based filtering, hybrid recommendations, NLP-powered search, and a fully server-side admin dashboard with A/B testing.

## Features

- **Multi-Algorithm Support**: Collaborative filtering, content-based, hybrid, and contextual recommendations
- **NLP-Powered Search**: Natural language queries parsed by an open-source LLM and semantic search
- **Real-time Learning**: Adaptive system that learns from user feedback
- **Scalable Architecture**: Redis caching, Celery background tasks, PostgreSQL database
- **A/B Testing**: Built-in, server-side framework for testing different algorithms
- **Admin Dashboard**: System monitoring, A/B test management, and controls (no JS required)
- **REST API**: Complete RESTful API with authentication and rate limiting

## Quick Start

### 1. Clone the repository
```bash
git clone <repository-url>
cd recommendation-system
```

### 2. Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the project root:
```
SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-secret
DATABASE_URL=postgresql://<user>:<password>@localhost:5432/recommendations_db
REDIS_URL=redis://localhost:6379/0
```

### 5. Initialize the database
```bash
flask db upgrade
```

### 6. Run the app locally
```bash
flask run
```

### 7. Access the app
- Open [http://localhost:5000](http://localhost:5000) in your browser.

## Deployment (Heroku Example)
- Make sure you have a `Procfile`, `requirements.txt`, and `runtime.txt`.
- Follow the Heroku deployment steps in the main documentation or ask for a step-by-step guide.

## Requirements
See `requirements.txt` for the full list. Key packages:
- Flask, Flask-SQLAlchemy, Flask-Migrate, Flask-Login
- PostgreSQL, Redis, Celery
- sentence-transformers, scikit-learn, numpy, pandas
- gunicorn (for production)

## Project Structure
```
recommendation-system/
  app.py
  models.py
  requirements.txt
  Procfile
  runtime.txt
  .env.example
  static/
  templates/
  dataset/
  migrations/
  ...
```

## Admin Dashboard
- Accessible at `/admin` after logging in as an admin user.
- All controls (retrain, clear cache, A/B test management) are server-side forms—no JavaScript required.

## NLP-Powered Search
- Enter natural language queries (e.g., "recommend dark books") in the search box.
- The system uses an open-source LLM to parse intent and semantic search to find relevant books.

## A/B Testing
- Create and manage A/B tests from the admin dashboard.
- Assignments and results are displayed server-side.

## License
MIT
