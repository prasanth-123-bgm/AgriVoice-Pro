## AgriVoice Assistant (Flask)

Smart agriculture assistant that provides:
- Weather reports via Open‑Meteo + Nominatim, with optional speech via gTTS
- Crop recommendation using a pre‑trained ML model
- Government schemes Q&A using SentenceTransformers semantic search
- Pest management guidance for a selected crop and area

### Features
- **Weather**: Enter a place name, PIN code, or coordinates (`lat,lon`). Supports English, हिंदी, తెలుగు. Returns a human‑readable report and base64 MP3 audio.
- **Crop recommendation**: Predicts the best crop from soil and weather features.
- **Schemes Q&A**: Retrieves the most relevant scheme description for your question.
- **Pest management**: Calculates pesticide dose by area and returns guidance text + audio.

### Project structure
```
CSP project/
  app.py
  weather.py
  crop_model (1).pkl
  label_encoder (1).pkl
  gov_schemes_dataset.csv
  pest_db.csv
  requirements.txt
  static/
    style.css
    csp background.jpg
  templates/
    index.html
```

### Requirements
- Python 3.9+
- Internet access (for Open‑Meteo, Nominatim, gTTS, and Deep Translator)
- First run will download the `paraphrase-MiniLM-L6-v2` model

Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the app
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

### Language support
- UI/API accept `lang` as one of: `English`, `हिन्दी`, `తెలుగు`
- Text‑to‑speech uses `en`/`hi`/`te` internally via gTTS

### API Endpoints

#### GET `/`
Renders the main UI (`templates/index.html`).

#### POST `/get_weather`
Request body (JSON):
```json
{ "location": "Guntur", "lang": "English" }
```
Response (JSON):
```json
{ "report": "Clear, 30°C, Humidity: 60%\nWind Speed: 10 km/h", "audio": "<base64-mp3>" }
```

Example curl:
```bash
curl -X POST http://127.0.0.1:5000/get_weather \
  -H "Content-Type: application/json" \
  -d '{"location": "16.3067,80.4365", "lang": "తెలుగు"}'
```

#### POST `/crop_recommendation`
Request body (JSON):
```json
{
  "N": 50, "P": 40, "K": 40,
  "temperature": 25.0, "humidity": 60.0,
  "ph": 6.5, "rainfall": 120.0,
  "lang": "हिन्दी"
}
```
Response (JSON):
```json
{ "recommendation": "अनुशंसित फसल: Rice", "audio": "<base64-mp3>" }
```

#### POST `/get_scheme`
Request body (JSON):
```json
{ "question": "What is the subsidy for drip irrigation?", "lang": "English" }
```
Response (JSON):
```json
{ "answer": "<localized answer>", "audio": "<base64-mp3>" }
```

#### POST `/pest_management`
Request body (JSON):
```json
{ "crop": "Cotton", "area": 2.5, "lang": "English" }
```
Response (JSON):
```json
{ "results": [ { "text": "...", "audio": "<base64-mp3>" } ] }
```

### Data and models
- `crop_model (1).pkl` and `label_encoder (1).pkl` must be present in the project root
- `gov_schemes_dataset.csv` contains scheme metadata (Name, Description, Eligibility, Benefits)
- `pest_db.csv` contains pesticide guidance and per‑hectare dose

### Notes on `weather.py`
- Contains helper functions used by the app and a demo that invokes gTTS and web calls. Running the app imports these helpers; any demo code in `weather.py` may make network requests during import. Ensure internet access is available.
- The app itself converts responses to audio via `gTTS` and returns base64 MP3.

### Troubleshooting
- **Timeouts / Network errors**: Ensure internet access to `api.open-meteo.com`, `nominatim.openstreetmap.org`, `translate.google.com`.
  - If using a proxy on Windows PowerShell:
    ```powershell
    $env:HTTP_PROXY="http://proxy:port"; $env:HTTPS_PROXY="http://proxy:port"
    ```
- **Model download slow/failing**: SentenceTransformers downloads on first use; try a stable connection or pre‑configure Hugging Face cache.
- **Audio not returned**: gTTS requires internet. If it fails, the API may return `audio: null`.

### Acknowledgements
- Weather: `Open‑Meteo` and `OpenStreetMap Nominatim`
- Speech: `gTTS`
- Translation: `deep-translator`
- Semantic search: `SentenceTransformers` (`paraphrase-MiniLM-L6-v2`) and `PyTorch`

### License
Add your preferred license (e.g., MIT) to the repository.
