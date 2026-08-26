"""Smoke-test the /predict route end to end, without starting a server.

    .venv/Scripts/python.exe tools/smoke_test.py

Builds a synthetic fundus-like image, pushes it through Flask's test client,
and checks the JSON contract the frontend in templates/index.html relies on:
diagnosis, message, recommendations. Also confirms DIAGNOSIS_INFO is not
mutated across requests, and that a missing file still returns 400.
"""

import io
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as application  # noqa: E402


def synthetic_fundus(size=384):
    """A warm orange disc with a bright optic-disc blob, on a dark surround."""
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = size / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    retina = r < size * 0.44

    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[..., 0] = np.where(retina, np.clip(200 - r * 0.3, 0, 255), 10)
    img[..., 1] = np.where(retina, np.clip(110 - r * 0.2, 0, 255), 5)
    img[..., 2] = np.where(retina, 40, 5)

    optic_disc = np.sqrt((yy - cy + 40) ** 2 + (xx - cx - 60) ** 2) < size * 0.07
    img[optic_disc] = [255, 230, 180]
    return Image.fromarray(img)


def main():
    if application.interpreter is None:
        raise SystemExit('FAIL: interpreter did not load')

    print(f'interpreter loaded: input={application.input_detail["shape"]}')

    # Snapshot DIAGNOSIS_INFO so we can prove requests do not mutate it.
    before = json.dumps(application.DIAGNOSIS_INFO, sort_keys=True)

    buf = io.BytesIO()
    synthetic_fundus().save(buf, format='PNG')
    buf.seek(0)

    client = application.app.test_client()

    print()
    print('--- POST /predict with an image ---')
    resp = client.post('/predict', data={'file': (buf, 'fundus.png')},
                       content_type='multipart/form-data')
    print(f'status: {resp.status_code}')
    body = resp.get_json()
    print(json.dumps(body, indent=2))

    assert resp.status_code == 200, f'expected 200, got {resp.status_code}'
    for key in ('diagnosis', 'message', 'recommendations'):
        assert key in body, f'response missing {key!r} (frontend reads it)'
    assert body['diagnosis'] in application.CLASS_LABELS, body['diagnosis']
    assert isinstance(body['recommendations'], list) and body['recommendations']

    print()
    print('--- POST /predict with no file ---')
    resp400 = client.post('/predict', data={}, content_type='multipart/form-data')
    print(f'status: {resp400.status_code} (expect 400)')
    assert resp400.status_code == 400, resp400.status_code

    print()
    print('--- GET / ---')
    root = client.get('/')
    print(f'status: {root.status_code} (expect 200), {len(root.data)} bytes')
    assert root.status_code == 200

    print()
    print('--- DIAGNOSIS_INFO unchanged after requests? ---')
    after = json.dumps(application.DIAGNOSIS_INFO, sort_keys=True)
    assert before == after, 'FAIL: DIAGNOSIS_INFO was mutated by a request'
    print('unchanged (no shared-dict mutation)')

    print()
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
