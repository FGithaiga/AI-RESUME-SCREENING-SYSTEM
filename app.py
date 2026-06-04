from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

def clean_text(text):
    import re, string
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    resume = request.form['resume']
    
    cleaned = clean_text(resume)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)
    
    category = le.inverse_transform(prediction)[0]
    
    return render_template('index.html', prediction=category)

if __name__ == "__main__":
    app.run(debug=True)