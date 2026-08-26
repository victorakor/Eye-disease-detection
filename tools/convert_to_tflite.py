"""Convert the Keras eye-disease model to TFLite (float16) for CPU-only serving.

Run this once, locally, whenever eye_disease_model.keras changes:

    .venv/Scripts/python.exe tools/convert_to_tflite.py

It writes eye_disease_model.tflite next to the Keras file, but only after
verifying the converted model agrees with the original. Two things get checked
because guessing either one wrong would silently corrupt every prediction:

  1. What efficientnet_v2.preprocess_input actually does to its input. The
     server no longer has TensorFlow, so app.py has to reproduce this in plain
     numpy. If it turns out to be a passthrough, app.py can drop it entirely.
  2. Whether the TFLite model's outputs still match the Keras model's on real
     input ranges, within float16 tolerance.
"""

import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERAS_PATH = os.path.join(HERE, 'eye_disease_model.keras')
TFLITE_PATH = os.path.join(HERE, 'eye_disease_model.tflite')

# Matches app.py: images are resized to this before inference.
INPUT_SIZE = (256, 256)


def describe_preprocessing():
    """Work out what preprocess_input does, so app.py can replicate it."""
    print('=' * 60)
    print('1. preprocess_input behaviour')
    print('=' * 60)

    # Span the full 0-255 pixel range, including both endpoints, so any
    # rescaling or mean-subtraction shows up in the min/max.
    probe = np.array([[0.0, 127.5, 255.0]], dtype=np.float32)
    out = np.asarray(preprocess_input(probe.copy()), dtype=np.float32)

    print(f'  in  : {probe.ravel()}')
    print(f'  out : {out.ravel()}')

    passthrough = np.allclose(probe, out, atol=1e-6)
    print(f'  VERDICT: {"PASSTHROUGH (no-op)" if passthrough else "TRANSFORMS INPUT"}')
    if not passthrough:
        print('  !! app.py must reproduce this transform in numpy !!')
    return passthrough


def load_keras_model():
    print()
    print('=' * 60)
    print('2. Loading Keras model')
    print('=' * 60)
    model = tf.keras.models.load_model(KERAS_PATH)
    # Keras 3 hands back a list for multi-input models, so normalise first.
    spec = model.inputs[0] if isinstance(model.inputs, (list, tuple)) else model.inputs
    print(f'  input  : shape={model.input_shape} dtype={spec.dtype}')
    print(f'  output : shape={model.output_shape}')
    print(f'  params : {model.count_params():,}')
    return model


def convert(model):
    """Convert to float16 TFLite, trying the Keras 3 SavedModel route if needed."""
    print()
    print('=' * 60)
    print('3. Converting to TFLite (float16)')
    print('=' * 60)

    def _float16(converter):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        return converter.convert()

    try:
        blob = _float16(tf.lite.TFLiteConverter.from_keras_model(model))
        print('  converted via from_keras_model')
        return blob
    except Exception as exc:
        # Keras 3 models often refuse the direct route; exporting a SavedModel
        # first is the documented fallback.
        print(f'  from_keras_model failed ({type(exc).__name__}), exporting SavedModel')

    tmp = tempfile.mkdtemp(prefix='eyemodel_')
    try:
        export_dir = os.path.join(tmp, 'saved_model')
        model.export(export_dir)
        blob = _float16(tf.lite.TFLiteConverter.from_saved_model(export_dir))
        print('  converted via SavedModel export')
        return blob
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def verify(model, blob, passthrough):
    """Compare TFLite predictions against Keras on realistic input."""
    print()
    print('=' * 60)
    print('4. Verifying TFLite vs Keras')
    print('=' * 60)

    interpreter = tf.lite.Interpreter(model_content=blob)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    print(f'  tflite input  : shape={inp["shape"]} dtype={np.dtype(inp["dtype"]).name}')
    print(f'  tflite output : shape={out["shape"]} dtype={np.dtype(out["dtype"]).name}')

    rng = np.random.default_rng(0)
    worst = 0.0
    disagreements = 0

    for trial in range(5):
        # Real uploads arrive as uint8 pixels, so probe that distribution.
        pixels = rng.integers(0, 256, size=(1, *INPUT_SIZE, 3)).astype(np.float32)
        batch = pixels if passthrough else np.asarray(
            preprocess_input(pixels.copy()), dtype=np.float32)

        keras_out = model.predict(batch, verbose=0)

        interpreter.set_tensor(inp['index'], batch.astype(inp['dtype']))
        interpreter.invoke()
        tflite_out = interpreter.get_tensor(out['index'])

        diff = float(np.max(np.abs(keras_out - tflite_out)))
        worst = max(worst, diff)
        if int(np.argmax(keras_out)) != int(np.argmax(tflite_out)):
            disagreements += 1
        print(f'  trial {trial}: max|diff|={diff:.2e}  '
              f'argmax keras={int(np.argmax(keras_out))} tflite={int(np.argmax(tflite_out))}')

    print(f'  worst max|diff| : {worst:.2e}')
    print(f'  argmax mismatches: {disagreements}/5')
    return worst, disagreements


def main():
    passthrough = describe_preprocessing()
    model = load_keras_model()
    blob = convert(model)
    worst, disagreements = verify(model, blob, passthrough)

    # float16 weights cost a little precision; a shifted argmax does not.
    if disagreements:
        raise SystemExit(f'ABORT: {disagreements}/5 predictions changed class. Not writing output.')
    if worst > 1e-2:
        raise SystemExit(f'ABORT: max diff {worst:.2e} exceeds 1e-2 tolerance. Not writing output.')

    with open(TFLITE_PATH, 'wb') as fh:
        fh.write(blob)

    before = os.path.getsize(KERAS_PATH) / 1e6
    after = os.path.getsize(TFLITE_PATH) / 1e6
    print()
    print('=' * 60)
    print(f'OK  wrote {os.path.basename(TFLITE_PATH)}')
    print(f'    {before:.1f} MB keras  ->  {after:.1f} MB tflite '
          f'({100 * after / before:.0f}%)')
    print(f'    preprocess_input is {"a no-op" if passthrough else "REQUIRED in app.py"}')
    print('=' * 60)


if __name__ == '__main__':
    main()
