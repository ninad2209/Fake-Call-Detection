from flask import Flask, request, jsonify
from flask_cors import CORS
from predictor import model, classifier, vectorizer, wordopt, output
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    calltext = data.get('text', '')

    if not calltext:
        return jsonify({'error': 'No text provided'}), 400

    testing_text = {"text": [calltext]}
    new_df = pd.DataFrame(testing_text)
    new_df["text"] = new_df["text"].apply(wordopt)
    new_xv = vectorizer.transform(new_df["text"])

    lr_pred  = model.predict(new_xv)[0]
    svm_pred = classifier.predict(new_xv)[0]

    # get confidence scores
    lr_proba  = model.predict_proba(new_xv)[0]
    svm_proba = classifier.predict_proba(new_xv)[0]

    lr_conf  = round(max(lr_proba) * 100, 1)
    svm_conf = round(max(svm_proba) * 100, 1)

    return jsonify({
        "lr_prediction":  output(int(lr_pred)),
        "svm_prediction": output(int(svm_pred)),
        "lr_confidence":  lr_conf,
        "svm_confidence": svm_conf
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)