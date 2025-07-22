from models import db, User
from flask import Flask
from werkzeug.security import generate_password_hash

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://vaish:Vaish_811496@localhost/recommendations_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

def create_admin():
    with app.app_context():
        admin_user = User.query.filter_by(username='admin').first()
        if not admin_user:
            admin = User(
                username='admin',
                email='vaishnavijangayadav@gmail.com',
                age=30,
                location='HQ',
                preferences={},
                is_admin=True
            )
            admin.set_password('admin@123')
            db.session.add(admin)
            db.session.commit()
            print('Admin user created: username="admin", password="admin@123"')
        else:
            print('Admin user already exists.')

if __name__ == "__main__":
    create_admin() 