# import libraries
from flask import Flask, render_template, request, jsonify
import pandas as pd
import random
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

df = pd.read_csv("data/gre_words_updated.csv")  #load the dataset
# COMBINE TEXT DATA
df["combined"] = df["word"].fillna("") + " " + df["definition"].fillna("") + " " + df["synonyms"].fillna("")

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["combined"])


app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///gre.db"
db = SQLAlchemy(app)
class QuizScore(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    correct = db.Column(db.Integer)
    total = db.Column(db.Integer)
    streak = db.Column(db.Integer)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_word", methods=["POST"])
def get_word():

    word = request.json["word"].lower()

    result = df[df["word"].str.lower() == word]

    if not result.empty:
        row = result.iloc[0]

        return jsonify({
            "word": row["word"],
            "definition": row["definition"],
            "part_of_speech": row["part_of_speech"],
            "example": row["example"],
            "synonyms": row["synonyms"] if pd.notna(row["synonyms"]) else "Not available",
            "antonyms": row["antonyms"] if pd.notna(row["antonyms"]) else "Not available"
        })

    return jsonify({
        "word": word,
        "definition": "Word not found.",
        "part_of_speech": "",
        "example": "",
        "synonyms": "",
        "antonyms": ""
    })

@app.route("/quiz-question", methods=["GET"])
def quiz_question():
    random_row = df.sample().iloc[0]
    return jsonify({
        "definition": random_row["definition"],
        "word_id": random_row["word"].lower()
    })

@app.route("/check-answer", methods=["POST"])
def check_answer():

    data = request.json

    user_answer = data["answer"].lower()

    correct_answer = data["correct"].lower()

    current_score = data.get("score", 0)

    current_total = data.get("total", 0)

    current_streak = data.get("streak", 0)

    if user_answer.strip().lower() == correct_answer.strip().lower():

        result = "correct"

    else:

        result = "wrong"

    # SAVE TO DATABASE

    quiz_score = QuizScore(

        correct=current_score,

        total=current_total,

        streak=current_streak
    )

    db.session.add(quiz_score)

    db.session.commit()

    if result == "correct":

        return jsonify({"result": "correct"})

    else:

        return jsonify({
            "result": "wrong",
            "correct": correct_answer
        })


@app.route("/stats", methods=["GET"])

def stats():

    scores = QuizScore.query.all()

    total_attempts = len(scores)

    if total_attempts == 0:

        return jsonify({
            "total_attempts": 0,
            "total_correct": 0,
            "best_streak": 0,
            "average_score": 0
        })
    total_correct = sum(score.correct for score in scores)

    best_streak = max(score.streak for score in scores)

    average_score = round(total_correct / total_attempts, 2)

    return jsonify({

        "total_attempts": total_attempts,

        "total_correct": total_correct,

        "best_streak": best_streak,

        "average_score": average_score
    })


@app.route("/semantic-search", methods=["POST"])
def semantic_search():

    data = request.json

    query = data["query"]

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, tfidf_matrix)

    best_match_index = similarities.argmax()

    result = df.iloc[best_match_index]

    return jsonify({

        "word": result["word"],

        "definition": result["definition"],

        "example": result["example"],

        "similarity": round(float(similarities[0][best_match_index]), 2)
    })


@app.route("/test-search")

def test_search():

    query = "bravery"

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, tfidf_matrix)

    best_match_index = similarities.argmax()

    result = df.iloc[best_match_index]

    return jsonify({

        "word": result["word"],

        "definition": result["definition"]

    })


@app.route("/debug-search")

def debug_search():

    query = "bravery"

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, tfidf_matrix)

    best_match_index = similarities.argmax()

    return jsonify({

        "best_word": df.iloc[best_match_index]["word"],

        "score": float(similarities[0][best_match_index])

    })




with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)


