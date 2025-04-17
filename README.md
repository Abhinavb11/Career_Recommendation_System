# Career Recommendation System

## Overview
The Career Recommendation System is a web-based application designed to help users identify suitable career paths based on their skills and interests. It leverages a machine learning model (K-Nearest Neighbors) to predict career options and provides additional insights such as job market trends, career advice, recommended courses, and top colleges using the Gemini API. The system includes a Flask backend, a user-friendly interface, and a chatbot for career-related queries.

## Features
- **Skill-Based Career Prediction**: Users select their skills, and the system predicts the most suitable career using a pre-trained KNN model.
- **Skill Gap Analysis**: Identifies missing skills required for the predicted career and suggests relevant courses.
- **Gemini API Integration**: Provides real-time insights on job trends, career advice, recommended courses, and top colleges.
- **Chatbot**: A conversational AI powered by Gemini to answer career, education, and technology-related questions.
- **User Interface**: Built with HTML templates for login, skill selection, and result display.

## Project Structure
```
Career_Recommendation_System/
├── app3.py                  # Main Flask application
├── career_Recommendation1.ipynb  # Jupyter notebook for data preprocessing and model training
├── career_model.pkl         # Pre-trained KNN model and related data
├── templates/               # HTML templates (login.html, index.html, result.html)
├── static/                  # CSS, JavaScript, and other static files
├── stud.csv                 # Dataset used for training the model
└── README.md                # Project documentation
```

## Prerequisites
- Python 3.8+
- Flask
- Pandas
- NumPy
- Scikit-learn
- Requests
- Google Generative AI SDK
- A valid Gemini API key

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd Career_Recommendation_System
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure you have a `requirements.txt` file with the following:
   ```
   flask
   pandas
   numpy
   scikit-learn
   requests
   google-generative-ai
   ```

3. **Set Up Gemini API Key**:
   - Replace `YOUR_GEMINI_API_KEY_HERE` in `app3.py` with your actual Gemini API key.
   - Alternatively, set it as an environment variable:
     ```bash
     export GEMINI_API_KEY='your-api-key'
     ```

4. **Prepare the Model**:
   - Ensure `career_model.pkl` is in the project directory. This file contains the pre-trained KNN model, career profiles, and feature columns.
   - If you need to retrain the model, run `career_Recommendation1.ipynb` with the `stud.csv` dataset.

5. **Run the Application**:
   ```bash
   python app3.py
   ```
   The app will start in debug mode at `http://127.0.0.1:5000`.

## Usage
1. **Login**: Access the login page at `/` and proceed to the skill selection page.
2. **Select Skills**: On the `/index` page, choose relevant skills from categorized lists (Technical, Creative, Soft, etc.).
3. **View Results**: Submit skills to `/predict` to receive:
   - Top career recommendation with confidence score
   - Top 3 career options
   - Skill gaps and suggested courses
   - Job trends, career advice, recommended courses, and top colleges
4. **Chatbot**: Use the `/chat` endpoint to ask career-related questions via a JSON POST request.

## Dataset
- **Source**: `https://github.com/Zurinlakdawala91/Career-Recommendation-System-using-ML/blob/main/stud.csv`
- **Description**: Contains 3535 rows with 59 skill columns (e.g., Coding, Drawing) and a target column (`Courses`) listing career paths (e.g., B.Tech, MBBS).
- **Preprocessing**: Handled in `career_Recommendation1.ipynb`, including splitting data and training the KNN model.

## Model
- **Algorithm**: K-Nearest Neighbors (KNN) with 5 neighbors.
- **Training**: Performed in `career_Recommendation1.ipynb` using `stud.csv`.
- **Output**: Saved as `career_model.pkl`, containing:
  - Trained KNN model
  - Career profiles (mean skill values per career)
  - Feature columns (skills)

## API Integration
- **Gemini API**: Used for fetching job trends, career advice, courses, and colleges.
- **Configuration**:
  - API Key: Set in `app3.py`
  - Model: `gemini-1.5-flash-latest`
  - Safety Settings: Blocks harmful content
  - Generation Config: Controls response length and quality

## Notes
- Ensure the Gemini API key is valid to avoid fallback data in results.
- The chatbot maintains session history per user (based on IP address).
- The application assumes the dataset and model file are in the correct format and location.
- For production, disable debug mode (`app.run(debug=False)`) and use a WSGI server like Gunicorn.

## Limitations
- The model relies on the quality and diversity of the `stud.csv` dataset.
- Gemini API responses may occasionally be blocked or empty, falling back to default values.
- The chatbot is limited to career, education, and technology queries.

## Future Improvements
- Enhance the dataset with more skills and career paths.
- Implement user authentication for personalized profiles.
- Add support for multiple languages in the chatbot.
- Integrate additional APIs for more comprehensive career insights.

## Contributor
- [Abhinav B]

  
