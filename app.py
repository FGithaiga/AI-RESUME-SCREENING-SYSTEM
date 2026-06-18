from flask import Flask, request, render_template
import pickle
import PyPDF2

app = Flask(__name__)

# ----------------------------
# LOAD MODELS
# ----------------------------
model = pickle.load(open("resume_classifier.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))


# ----------------------------
# PDF TEXT EXTRACTION
# ----------------------------
def extract_text_from_pdf(file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except Exception:
        return ""
    return text


# ----------------------------
# HOME
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ----------------------------
# PREDICT
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        resume_text = ""

        # PDF input
        file = request.files.get("resume")
        if file and file.filename != "":
            resume_text = extract_text_from_pdf(file)
        else:
            resume_text = request.form.get("resume_text")

        # validation
        if not resume_text or resume_text.strip() == "":
            return render_template(
                "index.html",
                prediction="No resume provided",
                confidence=0,
                top3=[]
            )

        # vectorize
        vector = tfidf.transform([resume_text])

        # prediction
        prediction = model.predict(vector)
        category = encoder.inverse_transform(prediction)[0]

        # probabilities
        probs = model.predict_proba(vector)[0]

        confidence = round(max(probs) * 100, 2)

        # TOP 3
        top3_idx = probs.argsort()[::-1][:3]
        top3 = [
            (encoder.classes_[i], round(probs[i] * 100, 2))
            for i in top3_idx
        ]

        return render_template(
            "index.html",
            prediction=category,
            confidence=confidence,
            top3=top3
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}",
            confidence=0,
            top3=[]
        )


# ----------------------------
# RUN APP
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)