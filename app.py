from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# โหลดโมเดลที่ Train ไว้
model = joblib.load("heart_disease_model.pkl")

@app.route("/")
def home():
    return "Heart Disease Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]
        features = np.array(data).reshape(1, -1)  # แปลงให้เป็นรูปแบบที่โมเดลรับได้
        prediction = model.predict(features)[0]
        result = "Presence" if prediction == 1 else "Absence"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
