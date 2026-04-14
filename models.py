from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with assessments
    assessments = db.relationship('Assessment', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<User {self.username}>'

class Assessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    assessment_type = db.Column(db.String(50), nullable=False)  # blood, heart, diabetes, cardio
    input_data = db.Column(db.Text, nullable=False)  # JSON string of form data
    prediction = db.Column(db.String(255), nullable=False)
    insights = db.Column(db.Text)  # JSON string of insights/notes
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def get_input_data(self):
        """Parse JSON input data"""
        try:
            return json.loads(self.input_data)
        except:
            return {}
    
    def set_input_data(self, data):
        """Store input data as JSON"""
        self.input_data = json.dumps(data)
    
    def get_insights(self):
        """Parse JSON insights"""
        try:
            return json.loads(self.insights) if self.insights else []
        except:
            return []
    
    def set_insights(self, insights):
        """Store insights as JSON"""
        self.insights = json.dumps(insights) if insights else None
    
    def __repr__(self):
        return f'<Assessment {self.assessment_type} for user {self.user_id}>'

class UserProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    date_of_birth = db.Column(db.Date)
    gender = db.Column(db.String(10))
    phone = db.Column(db.String(20))
    emergency_contact = db.Column(db.String(100))
    medical_conditions = db.Column(db.Text)  # JSON string for medical history
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship back to user
    user = db.relationship('User', backref=db.backref('profile', uselist=False))
    
    def get_medical_conditions(self):
        """Parse JSON medical conditions"""
        try:
            return json.loads(self.medical_conditions) if self.medical_conditions else []
        except:
            return []
    
    def set_medical_conditions(self, conditions):
        """Store medical conditions as JSON"""
        self.medical_conditions = json.dumps(conditions) if conditions else None
    
    def __repr__(self):
        return f'<UserProfile for user {self.user_id}>'