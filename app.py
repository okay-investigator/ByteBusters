from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load preventive measures and doctor location CSVs
preventive_df = pd.read_csv("preventive_measures.csv")
doc_df = pd.read_csv("doc_loc.csv")

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Predict disease
@app.route("/predict", methods=["POST"])
def predict():
    symptoms_input = request.form["symptoms"]
    if not symptoms_input.strip():
        return render_template("index.html", prediction="❌ Please enter symptoms.")

    symptoms_vectorized = vectorizer.transform([symptoms_input])
    prediction = model.predict(symptoms_vectorized)[0]

    return render_template("index.html",
                           prediction=f"{prediction}",
                           show_button=True,
                           predicted_disease=prediction)

# Show preventive measures
@app.route("/preventive", methods=["POST"])
def preventive():
    predicted_disease = request.form["predicted_disease"]
    row = preventive_df[preventive_df["Disease"].str.lower() == predicted_disease.lower()]
    
    if not row.empty:
        preventive_measures = row.iloc[0, 1:].dropna().tolist()
    else:
        preventive_measures = ["❌ No preventive measures found."]

    return render_template("index.html",
                           prediction=f"{predicted_disease}",
                           show_button=True,
                           predicted_disease=predicted_disease,
                           preventive_measures=preventive_measures)

# Show doctor info
@app.route("/doctor", methods=["POST"])
def doctor():
    predicted_disease = request.form["predicted_disease"]
    doc_row = doc_df[doc_df["Disease"].str.lower() == predicted_disease.lower()]
    
    if not doc_row.empty:
        doctor = doc_row["Doctor to Consult"].values[0]
        doctor_link = doc_row["Google Maps Search Link"].values[0]
    else:
        doctor = "❌ No doctor information found."
        doctor_link = "#"

    return render_template("index.html",
                           prediction=f"{predicted_disease}",
                           show_button=True,
                           predicted_disease=predicted_disease,
                           show_doctor=True,
                           doctor=doctor,
                           doctor_link=doctor_link)

if __name__ == "__main__":
    app.run(debug=True)
