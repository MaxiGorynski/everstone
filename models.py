from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    #Relationships for association of user with specific recordings and transcripts
    recordings = db.relationship('Recording', backref='user', lazy=True)

    def __repr__(self):
        return f"<User {self.username}>"

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Recording(db.Model):
    __tablename__ = 'recordings'
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    recording_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

class Transcript(db.Model):
    __tablename__ = 'transcript'
    id = db.Column(db.Integer, primary_key = True)
    content = db.Column(db.Text, nullable=False)
    transcript_id = db.Column(db.Integer, db.ForeignKey('recordings.id'), nullable=False)
