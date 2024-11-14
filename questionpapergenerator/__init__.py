from flask import Flask
from flask_pymongo import PyMongo
from dotenv import load_dotenv
import os

load_dotenv()
database_url = os.getenv("MONGO_URI")
secret_key = os.getenv("SECRET_KEY")

app = Flask(__name__)
app.config["MONGO_URI"] = database_url
app.config['SECRET_KEY'] = secret_key
db = PyMongo(app).db
mongo = PyMongo(app)
pdf_collection= mongo.db.pdf
users_collection = mongo.db.users

