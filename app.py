from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import joblib
from gtts import gTTS
from io import BytesIO
import numpy as np
from weather import get_weather_anywhere
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
import torch
import base64

app = Flask(__name__, static_folder='static', template_folder='templates')
load_dotenv()

# Load models and data
crop_model = joblib.load("crop_model (1).pkl")
le = joblib.load("label_encoder (1).pkl")
scheme_df = pd.read_csv("gov_schemes_dataset.csv")
scheme_df['context'] = scheme_df[['Scheme Name', 'Description', 'Eligibility', 'Benefits']].astype(str).agg(' '.join, axis=1)
scheme_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
scheme_corpus = scheme_df['context'].tolist()
scheme_embeddings = scheme_model.encode(scheme_corpus, convert_to_tensor=True)
pest_df = pd.read_csv("pest_db.csv")

# Text-to-speech helper
def text_to_speech(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_b64 = base64.b64encode(fp.read()).decode()
        return audio_b64
    except Exception as e:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_weather', methods=['POST'])
def get_weather_api():
    data = request.get_json()
    location = data.get('location')
    lang = data.get('lang', 'English')
    if not location:
        return jsonify({'error': 'No location provided.'})
    try:
        report = get_weather_anywhere(location)
        # Set language code for gTTS
        lang_code = 'en'
        if lang == "తెలుగు":
            lang_code = 'te'
        elif lang == "हिन्दी":
            lang_code = 'hi'
        audio = text_to_speech(report, lang_code)
        return jsonify({'report': report, 'audio': audio})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/crop_recommendation', methods=['POST'])
def crop_recommendation():
    data = request.get_json()
    try:
        features = [[
            data.get('N', 0),
            data.get('P', 0),
            data.get('K', 0),
            data.get('temperature', 0.0),
            data.get('humidity', 0.0),
            data.get('ph', 7.0),
            data.get('rainfall', 0.0)
        ]]
        prediction = crop_model.predict(features)[0]
        crop = le.inverse_transform([prediction])[0]
        lang = data.get('lang', 'English')
        if lang == "తెలుగు":
            msg = f"సిఫారసు చేసిన పంట: {crop}"
            lang_code = 'te'
        elif lang == "हिन्दी":
            msg = f"अनुशंसित फसल: {crop}"
            lang_code = 'hi'
        else:
            msg = f"Recommended crop: {crop}"
            lang_code = 'en'
        audio = text_to_speech(msg, lang_code)
        return jsonify({'recommendation': msg, 'audio': audio})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_scheme', methods=['POST'])
def get_scheme():
    data = request.get_json()
    question = data.get('question', '')
    lang = data.get('lang', 'English')
    if not question:
        return jsonify({'error': 'No question provided.'})
    try:
        question_en = GoogleTranslator(source='auto', target='en').translate(question)
        q_embedding = scheme_model.encode(question_en, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(q_embedding, scheme_embeddings)[0]
        best_idx = torch.argmax(similarities).item()
        answer_en = scheme_df.iloc[best_idx]['Description']
        lang_code = 'en'
        if lang == "తెలుగు":
            lang_code = 'te'
        elif lang == "हिन्दी":
            lang_code = 'hi'
        answer_native = GoogleTranslator(source='en', target=lang_code).translate(answer_en)
        audio = text_to_speech(answer_native, lang_code)
        return jsonify({'answer': answer_native, 'audio': audio})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/pest_management', methods=['POST'])
def pest_management():
    data = request.get_json()
    crop = data.get('crop', '')
    area = data.get('area', 0.0)
    lang = data.get('lang', 'English')
    if not crop or not area:
        return jsonify({'error': 'Please provide crop and area.'})
    filtered = pest_df[pest_df["Crop"].str.lower() == crop.lower()].copy()
    if filtered.empty:
        return jsonify({'error': 'No pest management data found for the selected crop.'})
    filtered["Total_Dose"] = filtered.apply(
        lambda row: round(row["Dose_per_ha"] * area, 2) if pd.notnull(row["Dose_per_ha"]) else None,
        axis=1
    )
    results = []
    for _, row in filtered.iterrows():
        english_text = f"For {row['Crop']} affected by {row['Pest_Disease']}, use {row['Pesticide']}. Required dose: {row['Total_Dose']} {row['Unit']}. Note: {row['Notes']}"
        if lang == "తెలుగు":
            translated = GoogleTranslator(source='en', target='te').translate(english_text)
            audio = text_to_speech(translated, 'te')
            results.append({'text': translated, 'audio': audio})
        elif lang == "हिन्दी":
            translated = GoogleTranslator(source='en', target='hi').translate(english_text)
            audio = text_to_speech(translated, 'hi')
            results.append({'text': translated, 'audio': audio})
        else:
            audio = text_to_speech(english_text, 'en')
            results.append({'text': english_text, 'audio': audio})
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
