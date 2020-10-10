# imports
from flask import Flask, jsonify, request
from preprocessing.functions import tokenize
import xgboost as xgb
import joblib

app = Flask(__name__)

# map target variables
target = {0:"Debt collection",
          1:"Credit card or prepaid card",
          2:"Mortgage",
          3:"Checking or savings account",
          4:"Student loan",
          5:"Vehicle loan or lease"}

tfvectorizer = joblib.load("models/tfvectorizer.pkl")
xgb_clf = xgb.Booster({'nthread':3})
xgb_clf.load_model('models/complaints.booster')

def scorer(text):
    encoded_text = tfvectorizer.transform([text])
    score = xgb_clf.predict(xgb.DMatrix(encoded_text))
    return score

@app.route("/score", methods=["POST"])
def predict_fn():
    text = request.get_json()['text']
    predictions = scorer(text)
    predictions = predictions.argmax(axis=1)[0]
    return jsonify({'predictions ':str(predictions), 'Category ': target.get(predictions)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5000')