from flask import Flask, request, jsonify,session
from flask import render_template
from flask_cors import CORS
import utils as ut
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
model=ut.Load_model()
print("Model loaded")

def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    

app = Flask(__name__,template_folder='../templates')
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = "application/json"

cors = CORS(app, resources={r"/classify-news": {"origins": "http://localhost:8001"}})
CORS(app)

@app.route("/signup")
def signup_page():
    return render_template("signup.html")  # Load signup page

@app.route("/")
def login_page():
    if 'logged_in' in session:
        return render_template("main.html") 
    return render_template("login.html")  # Load login page

@app.route("/login")
def login_form():
    return render_template("login.html")  # Load login page

@app.route("/index")
def index_page():
    if 'logged_in' in session:
        return render_template("main.html") 
    return render_template("login.html")  # Load index page


@app.route('/classify-news', methods=['POST'])
def classify_news():
    print(request.get_json())
    data=request.get_json()
    print("data123",data['news'])
    res=ut.prediction(model,data['news'])
    if res['prediction']=='Fake':
        result = "Fake"
        percentage=res['probability']
    else:
        result = "Real"
        percentage=res['probability']
    return jsonify({"result": result,
                    "percentage": percentage})

# Signup Endpoint
@app.route("/signupform", methods=["POST"])
def signup():
    data = request.get_json()
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email or not password:
        return jsonify({"error": "All fields are required!"}), 400

    hashed_password = generate_password_hash(password)
    try:
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
                       (username, email, hashed_password))
        conn.commit()
        conn.close()
        return jsonify({"message": "User registered successfully!"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email is already registered!"}), 400
@app.route("/loginform", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    print("email, password",email, password)

    if not email or not password:
        return jsonify({"error": "All fields are required!"}), 400

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    print(user)

    if user and check_password_hash(user[0], password):
        session['logged_in'] = True
        return jsonify({"message": "Login successful!"}), 200
    else:
        return jsonify({"error": "Invalid email or password!"}), 401

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=8001)
