from flask import Flask, request, render_template
import pickle
import PyPDF2
from docx import Document
import textract
import os

app = Flask(__name__)


# LOAD MODELS

model = pickle.load(open("resume_classifier.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))



# TEXT EXTRACTION FUNCTIONS


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


def extract_text_from_docx(file):
    try:
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception:
        return ""


def extract_text_from_doc(file, filename):
    try:
        temp_path = "temp.doc"
        file.save(temp_path)

        text = textract.process(temp_path)
        os.remove(temp_path)  # clean up temp file

        return text.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def extract_text_from_txt(file):
    try:
        return file.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""



# HOME

@app.route("/")
def home():
    return render_template("index.html")



# PREDICT

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("resume")
        resume_text = ""

        
        # FILE-BASED INPUT
        
        if file and file.filename != "":
            filename = file.filename.lower()

            if filename.endswith(".pdf"):
                resume_text = extract_text_from_pdf(file)

            elif filename.endswith(".docx"):
                resume_text = extract_text_from_docx(file)

            elif filename.endswith(".doc"):
                resume_text = extract_text_from_doc(file, filename)

            elif filename.endswith(".txt"):
                resume_text = extract_text_from_txt(file)

      
        # TEXT INPUT FALLBACK
        
        else:
            resume_text = request.form.get("resume_text")

       
        # VALIDATION
        
        if not resume_text or resume_text.strip() == "":
            return render_template(
                "index.html",
                prediction="No resume provided",
                confidence=0,
                top3=[]
            )

        
        # DEBUG (VERY IMPORTANT)
        
        print("EXTRACTED TEXT PREVIEW:\n", resume_text[:500])

        
        # VECTORIZE
      
        vector = tfidf.transform([resume_text])

      
        # PREDICT
       
        prediction = model.predict(vector)
        category = encoder.inverse_transform(prediction)[0]

        
        # PROBABILITIES
        
        probs = model.predict_proba(vector)[0]
        confidence = round(max(probs) * 100, 2)

     
        # TOP 3 PREDICTIONS
        
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



# RUN APP

if __name__ == "__main__":
    app.run(debug=True)