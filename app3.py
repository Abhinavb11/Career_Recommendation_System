import pandas as pd
import numpy as np
import pickle
import requests
import json
import google.generativeai as genai
from flask import Flask, render_template, request, redirect, url_for,jsonify

app = Flask(__name__)


genai.configure(api_key="AIzaSyAiXQ1pe3P1UD6PNYqCD9eh9YG6Ok0Hzjc")

# Load model only once
model = genai.GenerativeModel('gemini-1.5-flash-latest')



# --- Gemini API Configuration ---
GEMINI_API_KEY = 'AIzaSyAiXQ1pe3P1UD6PNYqCD9eh9YG6Ok0Hzjc'  # Replace with your actual Gemini API key
GEMINI_API_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/models/'
GEMINI_MODEL = 'gemini-1.5-flash-latest'  # Or 'gemini-1.5-pro-latest'
GEMINI_API_URL = f"{GEMINI_API_BASE_URL}{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
HEADERS = {'Content-Type': 'application/json'} 

GENERATION_CONFIG = {
    "temperature": 0.7,
    "topP": 1.0,
    "topK": 1,
    "maxOutputTokens": 256,
}

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Load the saved model and data from pickle file
try:
    with open('career_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        knn = model_data['knn']
        career_profiles = model_data['career_profiles']
        feature_columns = model_data['columns']
except FileNotFoundError:
    print("Error: career_model.pkl not found. Make sure the file is in the same directory.")
    exit()
except Exception as e:
    print(f"Error loading pickle file: {e}")
    exit()

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/index', methods=['GET'])
def index():
    skill_categories = {
        'Technical Skills': [
            'Coding', 'Electricity Components', 'Mechanic Parts', 'Computer Parts',
            'Physics', 'Chemistry', 'Mathematics', 'Biology', 'Science', 'Engeeniering'
        ],
        'Creative Skills': [
            'Drawing', 'Dancing', 'Singing', 'Photography', 'Makeup', 'Designing',
            'Content writing', 'Crafting', 'Cartooning', 'Knitting', 'Director'
        ],
        'Soft Skills': [
            'Debating', 'Solving Puzzles', 'Listening Music', 'Researching', 'Teaching'
        ],
        'Domain-Specific Skills': [
            'Accounting', 'Economics', 'Sociology', 'Geography', 'Psycology', 'History',
            'Bussiness Education', 'Literature', 'Reading', 'Architecture',
            'Historic Collection', 'Botany', 'Zoology', 'Doctor', 'Pharmisist',
            'Journalism', 'Bussiness'
        ],
        'Hobbies and Interests': [
            'Sports', 'Video Game', 'Acting', 'Travelling', 'Gardening',
            'Exercise', 'Gymnastics', 'Yoga', 'Cycling', 'Asrtology',
            'Hindi', 'English', 'Other Language'
        ]
    }
    all_skills = [skill for category in skill_categories.values() for skill in category]
    missing_from_categories = set(feature_columns) - set(all_skills)
    if missing_from_categories:
        if 'Other Skills' not in skill_categories:
            skill_categories['Other Skills'] = []
        skill_categories['Other Skills'].extend(list(missing_from_categories))
        skill_categories['Other Skills'] = sorted(list(set(skill_categories['Other Skills'])))
    return render_template('index.html', skill_categories=skill_categories)

def get_gemini_insights(career):
    """Fetch job trends, career suggestions, recommended courses, and top Indian colleges from Gemini API"""
    def call_gemini_api(prompt, max_tokens=256):
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {**GENERATION_CONFIG, "maxOutputTokens": max_tokens},
            "safetySettings": SAFETY_SETTINGS
        }
        try:
            response = requests.post(GEMINI_API_URL, headers=HEADERS, json=payload, timeout=60)
            print(f"API Request Payload: {json.dumps(payload, indent=2)}")
            print(f"API Response Status: {response.status_code}")
            print(f"API Response Text: {response.text}")
            response.raise_for_status()
            response_data = response.json()
            if 'candidates' not in response_data or not response_data['candidates']:
                if 'promptFeedback' in response_data and 'blockReason' in response_data['promptFeedback']:
                    block_reason = response_data['promptFeedback']['blockReason']
                    print(f"Gemini API call blocked. Reason: {block_reason}")
                    return f"Content blocked ({block_reason})."
                print(f"Gemini API Warning: No candidates. Response: {response_data}")
                return "No content generated."
            candidate = response_data['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content'] and candidate['content']['parts']:
                return candidate['content']['parts'][0]['text'].strip()
            finish_reason = candidate.get('finishReason', 'UNKNOWN')
            print(f"Gemini API Warning: Content missing. Finish Reason: {finish_reason}. Candidate: {candidate}")
            return f"No content (Finish Reason: {finish_reason})."
        except requests.exceptions.RequestException as e:
            print(f"Error calling Gemini API: {e}")
            error_details = ""
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    error_details = f" Status: {e.response.status_code}, Details: {error_data.get('error', {}).get('message', e.response.text)}"
                except json.JSONDecodeError:
                    error_details = f" Status: {e.response.status_code}, Body: {e.response.text}"
            return f"API Error: {error_details}"
        except Exception as e:
            print(f"Unexpected error in Gemini API call: {e}")
            return "Unexpected error occurred."

    # Define Prompts
    trend_prompt = f"In 2-3 concise sentences, summarize the job market trends and outlook for {career} careers in India for 2025 and its minimum package."
    suggestion_prompt = f"In 2-3 concise sentences, provide career advice and growth opportunities for a {career} career in India in 2025."
    courses_prompt = f"List point wise 5 recommended courses for a {career} career in India in 2025, each on a new line with a brief description don't include bold words(e.g., 'Course Name - Description')."
    colleges_prompt = f"List the only top 5 Indian colleges for studying a {career} career in 2025, each on a new line with a brief reason don't include bold words(e.g., 'College Name - Reason')."

    # Make API Calls
    job_trend = call_gemini_api(trend_prompt, max_tokens=150)
    career_suggestion = call_gemini_api(suggestion_prompt, max_tokens=150)
    courses_text = call_gemini_api(courses_prompt, max_tokens=350)
    colleges_text = call_gemini_api(colleges_prompt, max_tokens=350)

    # Process Results with Fallbacks
    default_trend = f"Demand for {career} is expected to grow in India in 2025."
    default_suggestion = f"Focus on skills and networking for {career} growth in 2025."
    default_courses = [
        "Course 1 - Basic introduction",
        "Course 2 - Intermediate skills",
        "Course 3 - Advanced techniques",
        "Course 4 - Practical applications",
        "Course 5 - Industry trends"
    ]
    default_colleges = [
        "IIT Bombay - Top engineering programs",
        "IIT Delhi - Strong industry ties",
        "IIT Madras - Research excellence",
        "BITS Pilani - Innovative curriculum",
        "NIT Trichy - Practical training"
    ]

    final_job_trend = job_trend if not job_trend.startswith(("Error", "Content blocked", "No content")) else default_trend
    final_career_suggestion = career_suggestion if not career_suggestion.startswith(("Error", "Content blocked", "No content")) else default_suggestion
    recommended_courses = ([line.strip() for line in courses_text.split('\n') if line.strip() and '-' in line]
                           if courses_text and not courses_text.startswith(("Error", "Content blocked", "No content"))
                           else default_courses)
    top_colleges = ([line.strip() for line in colleges_text.split('\n') if line.strip() and '-' in line]
                    if colleges_text and not colleges_text.startswith(("Error", "Content blocked", "No content"))
                    else default_colleges)

    print(f"Final Job Trend: {final_job_trend}")
    print(f"Final Career Suggestion: {final_career_suggestion}")
    print(f"Final Recommended Courses: {recommended_courses}")
    print(f"Final Top Colleges: {top_colleges}")

    return final_job_trend, final_career_suggestion, recommended_courses, top_colleges

@app.route('/predict', methods=['POST'])
def predict():
    new_user = np.zeros(len(feature_columns))
    selected_skills_count = 0
    for i, skill in enumerate(feature_columns):
        if request.form.get(skill) == 'on':
            new_user[i] = 1
            selected_skills_count += 1

    if selected_skills_count == 0:
        message = "You didn't choose any skills, so we can't recommend a course for you. Please select your skills."
        return render_template('result.html', message=message)
    if selected_skills_count == len(feature_columns):
        message = "You selected all skills, which makes it hard to recommend a specific course. Please refine your selection."
        return render_template('result.html', message=message)

    try:
        probas = knn.predict_proba([new_user])[0]
        top_career_idx = probas.argmax()
        top_career = knn.classes_[top_career_idx]
        confidence = probas[top_career_idx] * 100  # Keep as float

        # Debug prints
        print(f"Probabilities: {probas}")
        print(f"Top career: {top_career}")
        print(f"Confidence (numeric): {confidence}")

        if top_career in career_profiles.index:
            career_skills_needed = career_profiles.loc[top_career]
            user_skills_have = pd.Series(new_user, index=feature_columns)
            gaps = career_skills_needed[(career_skills_needed > 0) & (user_skills_have == 0)]
            missing_skills = gaps[gaps > 0.3].index.tolist()
        else:
            print(f"Warning: Predicted career '{top_career}' not found in career_profiles index.")
            missing_skills = ["Skill gap analysis unavailable"]

        top_3_idx = probas.argsort()[-3:][::-1]
        top_3_careers = [(knn.classes_[idx], f"{probas[idx]*100:.2f}%") for idx in top_3_idx]
    except Exception as e:
        print(f"Error during prediction: {e}")
        message = "An error occurred during the prediction process. Please try again."
        return render_template('result.html', message=message)

    if GEMINI_API_KEY == 'YOUR_GEMINI_API_KEY_HERE':
        print("\nWARNING: Gemini API Key not set. Using fallback data.\n")
        job_trend = "API Key Missing"
        career_suggestion = "API Key Missing"
        recommended_courses = ["Set Gemini API Key"]
        top_colleges = ["Set Gemini API Key"]
    else:
        print(f"Fetching Gemini insights for: {top_career}")
        job_trend, career_suggestion, recommended_courses, top_colleges = get_gemini_insights(top_career)
        print("Gemini insights received.")

    skill_gap_courses = {}
    if missing_skills and missing_skills[0] != "Skill gap analysis unavailable":
        for skill in missing_skills:
            skill_gap_courses[skill] = f"Online course for '{skill}'"

    return render_template('result.html',
                          top_career=top_career,
                          confidence=confidence,  # Pass as float, not string
                          missing_skills=missing_skills,
                          top_3_careers=top_3_careers,
                          colleges=top_colleges,
                          future_scope=career_suggestion,
                          job_trend=job_trend,
                          courses=recommended_courses,
                          skill_gap_courses=skill_gap_courses)


# Create chat session globally so it persists across questions
chat_session = None

# Store chat sessions (for basic memory)
chat_sessions = {}

@app.route('/chat', methods=['POST'])
def chat():
    session_id = request.remote_addr  # Simple session based on user IP
    user_message = request.json.get("message", "")

    try:
        # Initialize session if not present
        if session_id not in chat_sessions:
            system_prompt = (
                "You are a friendly and helpful career advisor chatbot "
                "Your job is to guide users about jobs, technologies, skills, and education. "
                "Always give short, clear answers (1-3 lines). "
                "Mention average salary ranges (in INR), popular tools, trending technologies, and top colleges. "
                "NEVER mention the word 'India' or 'Indian' in your replies. Just assume the user is from there. "
                "Only answer career, education, or technology-related questions. Be engaging and helpful."
            )

            # Start chat with system prompt
            chat_sessions[session_id] = model.start_chat(history=[
                {"role": "user", "parts": [system_prompt]}
            ])

        # Continue the chat
        chat = chat_sessions[session_id]
        response = chat.send_message(user_message)
        bot_reply = response.text.strip()

    except Exception as e:
        print("Gemini Error:", e)
        bot_reply = "I'm having trouble right now. Please ask again in a moment."

    return jsonify({"reply": bot_reply})





if __name__ == '__main__':
    if GEMINI_API_KEY == 'YOUR_GEMINI_API_KEY_HERE':
        print("\n" + "="*50)
        print(" WARNING: Gemini API Key is not set! ")
        print(" Please replace 'YOUR_GEMINI_API_KEY_HERE' in the script ")
        print(" or set the GEMINI_API_KEY environment variable. ")
        print(" API calls for insights will fail. ")
        print("="*50 + "\n")
    print("Starting Flask app...")
    app.run(debug=True)