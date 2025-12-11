import streamlit as st
import requests
from pathlib import Path
import json

import os

DEFAULT_SERVER = os.environ.get('DEMO_SERVER_URL', "http://127.0.0.1:8000/predict")
DEFAULT_API_KEY = os.environ.get('DEMO_API_KEY', "")


def call_predict(server_url, image_path, state_path=None, in_ch=3, api_key=None):
    params = {}
    if state_path:
        params['state_dict'] = str(state_path)
    params['in_channels'] = str(in_ch)

    headers = {}
    if api_key:
        headers['X-API-Key'] = api_key

    files = {"file": open(image_path, "rb")}
    try:
        r = requests.post(server_url, params=params, files=files, headers=headers, timeout=10)
    finally:
        files['file'].close()

    r.raise_for_status()
    return r.json()


def call_health(server_base):
    try:
        # health endpoint is at /health and authorize at /authorize
        if server_base.endswith('/predict'):
            base = server_base[:-len('/predict')]
        else:
            base = server_base
        r = requests.get(base + '/health', timeout=3)
        return (r.status_code, r.text)
    except Exception as e:
        return (None, str(e))


def main():
    st.title("Prostate Severity demo")

    col1, col2 = st.columns(2)

    with col1:
        server = st.text_input("Inference server URL", DEFAULT_SERVER)
        api_key = st.text_input("API key (optional)", type="password", value=DEFAULT_API_KEY)
        state = st.text_input("Model state or ONNX path (optional)", value="./models/prototype_toy.pth")
        st.markdown("---")
        st.markdown("Server health:\n(press 'Check server')")
        if st.button('Check server'):
            code, txt = call_health(server)
            if code is None:
                st.error(f"Server unreachable: {txt}")
            else:
                st.success(f"Status code {code} — {txt}")
        # allow fetching runtime model status
        if st.button('Refresh model status'):
            try:
                base = server[:-len('/predict')] if server.endswith('/predict') else server
                r = requests.get(base + '/model_status', timeout=5)
                r.raise_for_status()
                ms = r.json()
                st.write('Model loaded:', ms.get('loaded'))
                st.write('Model path:', ms.get('model_path'))
                st.write('Device:', ms.get('device'))
                st.write('In-ch:', ms.get('in_channels'))
                if 'train_log_tail' in ms:
                    st.text_area('train_log tail', ms['train_log_tail'], height=200)
            except Exception as e:
                st.error(f'Failed to fetch /model_status: {e}')

    with col2:
        uploaded = st.file_uploader("Upload an image (or pick a local file below)", type=["png", "jpg", "jpeg"])
        uploaded_local = st.text_input("Local image path (optional)")

    image_path = None
    if uploaded is not None:
        # save a temporary copy
        tmp = Path(".streamlit_tmp")
        tmp.mkdir(exist_ok=True)
        fp = tmp / uploaded.name
        with open(fp, "wb") as f:
            f.write(uploaded.getbuffer())
        image_path = fp
    elif uploaded_local:
        image_path = Path(uploaded_local)

    # show image preview
    if image_path and Path(image_path).exists():
        st.image(str(image_path), caption='Selected image', use_column_width=True)

    if image_path and st.button("Run inference"):
        if not Path(image_path).exists():
            st.error(f"Image not found: {image_path}")
            return

        st.info("Calling server — this may take a few seconds...")
        try:
            out = call_predict(server, image_path, state_path=state if state else None, in_ch=3, api_key=api_key if api_key else None)
            st.success("OK")
            st.json(out)
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

    st.markdown('---')
    st.subheader('YOLO detection & severity')
    det_col1, det_col2 = st.columns(2)
    with det_col1:
        yolo_weights = st.text_input('YOLO weights path (local)', './models/yolov8s_prostate/weights/best.pt')
        detect_conf = st.slider('Detection confidence', 0.0, 1.0, 0.25)
        run_detect = st.button('Run detection (YOLO)')

    with det_col2:
        st.write('Severity mapping: small < 10mm, medium 10-20mm, large > 20mm')

    if image_path and run_detect:
        try:
            # run local inference script using ultralytics directly
            from ultralytics import YOLO
            model = YOLO(yolo_weights)
            res = model.predict(source=str(image_path), conf=detect_conf)
            r = res[0]
            boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else []
            scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else []
            labels = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, 'cls') else []

            # load meta if available next to image
            meta_fp = Path(image_path).with_suffix('.json')
            meta = {}
            if meta_fp.exists():
                meta = json.loads(meta_fp.read_text())

            # draw boxes
            from PIL import ImageDraw, ImageFont
            pil = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(pil)
            for i, b in enumerate(boxes):
                x1, y1, x2, y2 = [float(x) for x in b]
                draw.rectangle(((x1, y1), (x2, y2)), outline='red', width=3)
                # compute physical size
                size_mm = None
                if meta.get('pixel_spacing'):
                    ps = meta['pixel_spacing']
                    avg = (float(ps[0]) + float(ps[1])) / 2.0
                    w_px = x2 - x1
                    h_px = y2 - y1
                    size_mm = max(w_px, h_px) * avg
                severity = 'unknown' if size_mm is None else ('small' if size_mm < 10 else ('medium' if size_mm < 20 else 'large'))
                txt = f"{severity} ({size_mm:.1f}mm)" if size_mm else severity
                draw.text((x1 + 4, y1 + 4), txt, fill='yellow')

            st.image(pil, use_column_width=True)
            # show detections table
            dets = []
            for i, b in enumerate(boxes):
                x1, y1, x2, y2 = [float(x) for x in b]
                size_mm = None
                if meta.get('pixel_spacing'):
                    ps = meta['pixel_spacing']
                    avg = (float(ps[0]) + float(ps[1])) / 2.0
                    size_mm = max(x2 - x1, y2 - y1) * avg
                severity = 'unknown' if size_mm is None else ('small' if size_mm < 10 else ('medium' if size_mm < 20 else 'large'))
                dets.append({'score': float(scores[i]) if len(scores) > i else None, 'size_mm': size_mm, 'severity': severity, 'bbox': [x1, y1, x2, y2]})
            st.json(dets)

        except Exception as e:
            st.error(f'local detection failed: {e}')

    st.markdown('---')
    st.subheader('Dataset & DICOM diagnostics')
    # fetch available series sample keys
    try:
        base = server[:-len('/predict')] if server.endswith('/predict') else server
        r = requests.get(base + '/series_status', timeout=5)
        r.raise_for_status()
        sinfo = r.json()
    except Exception as e:
        sinfo = {'error': str(e)}

    if 'error' in sinfo:
        st.warning(f"Unable to query series_status: {sinfo['error']}")
    else:
        st.write(f"Total series available (sample): {sinfo.get('available_series_count', 'n/a')}")
        choices = sinfo.get('sample_keys', [])
        if choices:
            selected = st.selectbox('Pick a sample series UID (or paste one)', choices)
            manual = st.text_input('Or enter Series UID manually')
            final = manual.strip() if manual.strip() else selected
            if st.button('Check selected series') and final:
                try:
                    r2 = requests.get(base + '/series_status', params={'series_uid': final}, timeout=8)
                    r2.raise_for_status()
                    j = r2.json()
                    if not j.get('found'):
                        st.error('Series not found')
                    else:
                        st.success(f"Found {j.get('dcm_count')} DICOM files at: {j.get('path')}")
                        st.json({k: v for k, v in j.items() if k != 'sample_files'})
                        sf = j.get('sample_files', [])
                        if sf:
                            st.write('Sample files:')
                            for f in sf:
                                st.write('-', f)
                except Exception as e:
                    st.error(f'failed to query series_status: {e}')

    st.markdown('---')
    st.subheader('Quick model evaluation (toy)')
    if st.button('Run quick toy eval on server'):
        try:
            params = {}
            # allow user to request eval using current model state path
            if state:
                params['model_path'] = state
            params['toy'] = 'true'
            params['toy_len'] = 64
            params['num_slices'] = 1
            r = requests.get((server[:-len('/predict')] if server.endswith('/predict') else server) + '/quick_eval', params=params, timeout=30)
            r.raise_for_status()
            res = r.json()
            if 'error' in res:
                st.error(res['error'])
            else:
                # display metrics nicely
                cols = st.columns(4)
                cols[0].metric('Accuracy', f"{res.get('accuracy', 'n/a'):.3f}" if res.get('accuracy') is not None else 'n/a')
                cols[1].metric('AUC', f"{res.get('auc', 'n/a'):.3f}" if res.get('auc') is not None else 'n/a')
                cols[2].metric('Precision', f"{res.get('precision', 'n/a'):.3f}" if res.get('precision') is not None else 'n/a')
                cols[3].metric('Recall', f"{res.get('recall', 'n/a'):.3f}" if res.get('recall') is not None else 'n/a')
                st.json(res)
        except Exception as e:
            st.error(f'quick eval failed: {e}')


if __name__ == '__main__':
    main()
