# Eye Disease Detection

A deep-learning web application that classifies retinal fundus images into four
categories — **Cataract**, **Diabetic Retinopathy**, **Glaucoma**, and
**Normal** — and returns a plain-language explanation with follow-up
recommendations for each result.

> **Not a medical device.** This is a decision-support demonstration built for
> educational purposes. It is not validated for clinical use and must never
> replace examination by a qualified ophthalmologist.

## How it works

| Stage | Detail |
| --- | --- |
| Backbone | **EfficientNetV2**, fine-tuned for four-class retinal classification |
| Input | RGB image resized to **256 × 256**, passed through `efficientnet_v2.preprocess_input` |
| Output | Softmax over 4 classes; the arg-max label is returned with its confidence |
| Serving | **Flask** with CORS enabled, so the frontend can be hosted separately |

The trained model (`eye_disease_model.keras`, ~55 MB) is committed to this
repository, so no separate download step is needed.

## Install

Requires Python 3.10 or newer.

```bash
git clone https://github.com/victorakor/Eye-disease-detection.git
cd Eye-disease-detection

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open <http://127.0.0.1:5000> and upload a retinal image.

To use a different model file, point `MODEL_PATH` at it:

```bash
MODEL_PATH=/path/to/other_model.keras python app.py
```

## API

### `POST /predict`

Multipart form upload with the image in a `file` field.

```bash
curl -X POST -F "file=@retina.jpg" http://127.0.0.1:5000/predict
```

```json
{
  "diagnosis": "Diabetic Retinopathy",
  "message": "There are signs of diabetic retinopathy. Immediate consultation with a specialist is highly recommended.",
  "recommendations": [
    "Schedule an appointment with an ophthalmologist.",
    "Monitor blood sugar levels closely.",
    "Follow a balanced diet and regular exercise routine."
  ]
}
```

| Status | Meaning |
| --- | --- |
| `200` | Prediction succeeded |
| `400` | No `file` field in the request |
| `500` | Model failed to load, or the image could not be processed |

### `GET /`

Serves the single-page upload interface from `templates/index.html`.

## Class labels

Label order is significant — it must match the order the model's output layer
was trained against, which is the alphabetical order of the training
directories:

```python
['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
```

## Project layout

```
.
├── app.py                     Flask server, preprocessing, prediction endpoint
├── eye_disease_model.keras    trained EfficientNetV2 model (~55 MB)
├── requirements.txt
└── templates/
    └── index.html             upload interface
```

## Notes for production

`app.py` runs with `debug=True` for local development. Before deploying, set
`debug=False` and serve behind a WSGI server such as `gunicorn`:

```bash
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```
