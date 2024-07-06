#==============================================================================
# This file is to host the NER as an app using flask so that it can be run with 
# Label Studio.
#==============================================================================

from flask import Flask, request, jsonify
import spacy

# Load your custom model
#nlp = spacy.load("custom_sci_ner_abs_v2")
nlp = spacy.load("custom_web_ner_abs_v382")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text')
    if text:
        doc = nlp(text)
        predictions = []
        for ent in doc.ents:
            predictions.append({
                'start': ent.start_char,
                'end': ent.end_char,
                'label': ent.label_
            })
        return jsonify(predictions)
    return jsonify({"error": "No text provided"})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/setup', methods=['POST'])
def setup():
    # This is a placeholder endpoint that Label Studio needs to validate the backend
    return jsonify({"status": "ready"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
