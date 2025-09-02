from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import sqlite3
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# Initialize FastAPI app
app = FastAPI()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Create SQLite DB
def init_db():
    conn = sqlite3.connect("chatbot.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chatlogs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  question TEXT,
                  answer TEXT,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Predefined FAQ data
faq_data = {
    "What is your name?": "I am your AI-powered assistant.",
    "How can you help me?": "I can answer your FAQs and provide support.",
    "What is AI?": "AI stands for Artificial Intelligence, the simulation of human intelligence by machines."
}

faq_questions = list(faq_data.keys())
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    with open("static/index.html", "r") as f:
        return HTMLResponse(f.read())

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")

    # Match with FAQ using embeddings
    query_embedding = model.encode(user_message, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, faq_embeddings)[0]
    best_match_idx = int(scores.argmax())
    best_score = float(scores[best_match_idx])

    if best_score > 0.6:
        bot_reply = faq_data[faq_questions[best_match_idx]]
    else:
        bot_reply = "I'm not sure about that. Can you rephrase?"

    # Log interaction
    conn = sqlite3.connect("chatbot.db")
    c = conn.cursor()
    c.execute("INSERT INTO chatlogs (question, answer, timestamp) VALUES (?, ?, ?)",
              (user_message, bot_reply, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    return JSONResponse({"reply": bot_reply})
