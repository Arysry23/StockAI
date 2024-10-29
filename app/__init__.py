# app/__init__.py
from flask import Flask

app = Flask(__name__)

# Import routes (app logic) here
from app import app  # This line should import the routes in app/app.py
