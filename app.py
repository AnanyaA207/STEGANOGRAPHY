from flask import Flask, render_template, request, redirect, url_for
import os, time, base64
from encoder import embed_polymorphic, extract_polymorphic, get_log
from utils import compute_psnr_ssim
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD = "static/uploads"
os.makedirs(UPLOAD, exist_ok=True)

last_uploaded = None
last_stego = None

@app.route('/')
def index():
    return render_template('index.html',
                           original=None,
                           stego=None,
                           psnr=None,
                           ssim=None,
                           decoded=None,
                           log=get_log(),
                           last_uploaded=last_uploaded,
                           last_stego=last_stego)

# New capture endpoint: receive image from browser
@app.route('/capture', methods=['POST'])
def capture():
    global last_uploaded
    img_data = request.form.get('image')
    if not img_data:
        return "No image data received", 400

    # decode base64 image
    header, encoded = img_data.split(",", 1)
    data = base64.b64decode(encoded)
    fname = f"capture_{int(time.time())}.png"
    fpath = os.path.join(UPLOAD, fname)
    with open(fpath, "wb") as f:
        f.write(data)

    last_uploaded = fname
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload():
    global last_uploaded
    f = request.files.get('imagefile')
    if not f:
        return "No file", 400
    fname = f.filename
    save_path = os.path.join(UPLOAD, fname)
    f.save(save_path)
    last_uploaded = fname
    return redirect(url_for('index'))

@app.route('/capture_ajax', methods=['POST'])
def capture_ajax():
    import base64
    global last_uploaded
    data_url = request.json.get('image')
    if not data_url:
        return {"error": "No image data"}, 400

    header, encoded = data_url.split(",", 1)
    data = base64.b64decode(encoded)
    fname = f"capture_{int(time.time())}.png"
    fpath = os.path.join(UPLOAD, fname)
    with open(fpath, "wb") as f:
        f.write(data)
    last_uploaded = fname
    return {"success": True, "filename": fname}

@app.route('/embed', methods=['POST'])
def embed():
    global last_stego
    image_name = request.form.get('image_path')
    message = request.form.get('message','')
    key = request.form.get('key','1')  # default single HUGO

    if not image_name:
        return "No image path provided", 400

    in_path = os.path.join(UPLOAD, image_name)
    if not os.path.exists(in_path):
        return "Image not found", 404

    out_name = f"stego_{int(time.time())}.png"
    out_path = os.path.join(UPLOAD, out_name)

    embed_polymorphic(in_path, message, key, out_path)
    last_stego = out_name

    psnr_val, ssim_val = compute_psnr_ssim(in_path, out_path)

    return render_template('index.html',
                           original=image_name,
                           stego=out_name,
                           psnr=psnr_val,
                           ssim=ssim_val,
                           decoded=None,
                           log=get_log(),
                           last_uploaded=last_uploaded,
                           last_stego=last_stego)

@app.route('/extract', methods=['POST'])
def extract():
    image_name = request.form.get('extract_image')
    if not image_name:
        return "No image name provided", 400
    in_path = os.path.join(UPLOAD, image_name)
    if not os.path.exists(in_path):
        return "Image not found", 404
    msg = extract_polymorphic(in_path)
    return render_template('index.html',
                           decoded=msg,
                           original=None,
                           stego=None,
                           psnr=None,
                           ssim=None,
                           log=get_log(),
                           last_uploaded=last_uploaded,
                           last_stego=last_stego)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

