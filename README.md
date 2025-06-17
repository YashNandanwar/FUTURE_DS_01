# Social Media Sentiment Analysis

## Overview
This project performs sentiment analysis on social media posts using Python. It includes data preprocessing, sentiment scoring, visualization, machine learning model training, evaluation, and a Streamlit dashboard for interactive analytics.

## Tasks & Code Explanation

### 1. Data Loading & Initial Exploration
- **Dataset:** Social Media Sentiments (from Kaggle)
- **Libraries Used:** pandas, numpy, matplotlib, seaborn, wordcloud, textblob
- **Process:**
  - Download and load the dataset.
  - Display the first few rows for inspection.

### 2. Data Preprocessing & Sentiment Scoring
- Remove rows with missing text.
- Use TextBlob to calculate sentiment polarity and subjectivity for each post.
- Classify sentiment as Positive, Negative, or Neutral based on polarity.

### 3. Visualization
- **Sentiment Distribution:** Bar plot of sentiment counts.
- **Word Cloud:** Visualize most frequent words in posts.
- **Polarity vs Subjectivity:** Scatter plot colored by sentiment.

### 4. Text Cleaning
- Remove special characters, numbers, and stopwords using NLTK.
- Add a new column `cleaned_text` with processed text.

### 5. Tokenization & Vectorization
- Convert cleaned text into numerical features using TF-IDF vectorization.

### 6. Model Selection & Training
- **Model Used:** Logistic Regression
- Split data into training and testing sets (75/25 split).
- Train the model on TF-IDF features.

### 7. Model Evaluation
- Evaluate model using accuracy, precision, recall, and F1-score.
- Display classification report and confusion matrix.

### 8. Prediction & Analysis
- Predict sentiment on test data.
- Compare actual vs predicted sentiment.
- Analyze model performance and misclassifications.

### 9. Model Saving
- Save the trained model using `joblib` for future use.

### 10. Interactive Dashboard
- **dashboard.py:**
  - Built with Streamlit and Plotly.
  - Loads the dataset and displays raw data.
  - Shows metrics (total posts, average likes/retweets).
  - Visualizes sentiment distribution, posts by platform, and posts over time.
  - Sidebar filters for country and platform.
  - Displays filtered data interactively.

## How to Run
1. **Install dependencies:**
   - Ensure you have Python 3.8+ and pip installed.
   - Install required packages:
     ```powershell
     pip install pandas numpy matplotlib seaborn wordcloud textblob scikit-learn nltk streamlit plotly kagglehub joblib
     ```
2. **Run the Jupyter Notebook:**
   - Open `TASK.ipynb` to view and execute the analysis step-by-step.
3. **Launch the Dashboard:**
   - In your terminal, run:
     ```powershell
     streamlit run dashboard.py
     ```
   - The dashboard will open in your browser for interactive exploration.

## Results & Insights
- The model achieved moderate accuracy, with best performance on neutral sentiment.
- Negative sentiment detection can be improved by addressing class imbalance or using advanced models.
- The dashboard provides an interactive way to explore sentiment trends and platform/country breakdowns.

## File Structure
- `TASK.ipynb` — Main analysis notebook (data loading, cleaning, modeling, evaluation)
- `dashboard.py` — Streamlit dashboard for interactive analytics
- `sentimentdataset.csv` — Social media sentiment dataset
- `logistic_regression_sentiment_model.pkl` — Saved trained model

## Next Steps
- Explore advanced NLP models (e.g., BERT, LSTM)
- Address class imbalance in the dataset
- Add more interactive features to the dashboard

---
*For any questions or suggestions, please contact the project maintainer.*
