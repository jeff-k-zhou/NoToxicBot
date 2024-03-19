from transformers import pipeline
from flask import Flask, request

app = Flask(__name__) 

@app.route('/predict', methods=['POST'])
def predict():
    rq = request.get_json()
    classifier = pipeline('sentiment-analysis', model="./result_model/checkpoint-7979")
    result = classifier(rq["text"])
    return result

app.run()


