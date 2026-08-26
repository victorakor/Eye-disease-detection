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
| Input | RGB image resized to **256 × 256**, fed as float32 pixels in `[0, 255]` |
| Output | Four **raw logits** — the final `Dense` layer is linear. The server applies softmax to report a confidence; `argmax` is unaffected either way |
| Inference | **LiteRT** (TFLite) on a float16-converted model — no TensorFlow at runtime |
| Serving | **Flask** with CORS enabled, so the frontend can be hosted separately |

Note that `efficientnet_v2.preprocess_input` is a documented **no-op** — the
rescaling is baked into the network, which expects pixels in the `[0, 255]`
range. `tools/convert_to_tflite.py` asserts this on every conversion, so
`app.py` only has to cast to float32.

### Why TFLite

Full TensorFlow needs roughly **600 MB** of RAM just to `import`, which does not
fit in a 512 MB free hosting tier. The float16 TFLite model runs the same
network in an estimated **150–250 MB**, and shrinks the weights from 55 MB to
15.5 MB. Conversion is verified, not assumed: the script aborts unless predicted
classes match the Keras model exactly on probe inputs.

Both models are committed, so there is no separate download step:

| File | Size | Role |
| --- | --- | --- |
| `eye_disease_model.keras` | ~55 MB | source of truth, used for conversion |
| `eye_disease_model.tflite` | ~15.5 MB | what actually gets served |

## Install

Requires Python 3.10 or newer.

```bash
git clone https://github.com/victorakor/Eye-disease-detection.git
cd Eye-disease-detection

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` deliberately contains **no TensorFlow**. On Linux, inference
uses the `ai-edge-litert` wheel. `ai-edge-litert` publishes no Windows wheels,
so local runs on Windows fall back to `tf.lite` — install the dev extras for
that, and to regenerate the TFLite model:

```bash
pip install -r requirements-dev.txt
```

> TensorFlow **2.21.0 is broken on Windows**: it expects a separate `litert`
> package that does not exist on PyPI, so `import tensorflow` fails outright.
> `requirements-dev.txt` therefore stays on a version where `tf.lite` works.

## Run

```bash
python app.py
```

Then open <http://127.0.0.1:5000> and upload a retinal image. Set `PORT` to
serve elsewhere, and `FLASK_DEBUG=1` to enable auto-reload (off by default —
debug mode exposes an interactive console on tracebacks).

To use a different model file, point `MODEL_PATH` at it:

```bash
MODEL_PATH=/path/to/other_model.tflite python app.py
```

## Regenerating the TFLite model

Run this whenever `eye_disease_model.keras` changes:

```bash
python tools/convert_to_tflite.py
```

It reports what `preprocess_input` does, converts to float16, compares TFLite
against Keras across probe inputs, and refuses to write the output if any
predicted class shifts.

Then check the app still serves predictions end to end:

```bash
python tools/smoke_test.py
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
  "confidence": 0.8909,
  "message": "There are signs of diabetic retinopathy. Immediate consultation with a specialist is highly recommended.",
  "recommendations": [
    "Schedule an appointment with an ophthalmologist.",
    "Monitor blood sugar levels closely.",
    "Follow a balanced diet and regular exercise routine."
  ]
}
```

`confidence` is the softmax probability of the predicted class, in `[0, 1]`.

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
├── app.py                      Flask server, preprocessing, prediction endpoint
├── eye_disease_model.keras     trained EfficientNetV2 model (~55 MB)
├── eye_disease_model.tflite    float16 conversion actually served (~15.5 MB)
├── render.yaml                 Render Blueprint (free tier)
├── Dockerfile                  container option for other hosts
├── requirements.txt            runtime deps -- no TensorFlow
├── requirements-dev.txt        adds TensorFlow, for conversion only
├── tools/
│   ├── convert_to_tflite.py    Keras -> TFLite, with verification
│   └── smoke_test.py           end-to-end check of /predict
└── templates/
    └── index.html              upload interface
```

## Deploying

`render.yaml` is a Render Blueprint targeting the free tier. In the Render
dashboard choose **New → Blueprint**, point it at this repository, and apply.
It runs:

```bash
gunicorn -w 1 --threads 4 -b 0.0.0.0:$PORT --timeout 120 app:app
```

One worker keeps a single copy of the interpreter in memory; the threads absorb
concurrent viewers. A TFLite interpreter is **not thread-safe**, so `app.py`
serialises `invoke()` behind a lock.

Free Render instances **sleep after 15 minutes idle** and take roughly 50
seconds to wake, so open the URL a few minutes before demonstrating it.

The `Dockerfile` covers hosts that want a container instead; it serves the same
app under gunicorn.
