from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime
import pytz

# --- Inisialisasi Flask ---
app = Flask(__name__)

# --- Folder upload ---
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Konfigurasi PostgreSQL ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:110288@localhost:5432/rokok_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Inisialisasi SQLAlchemy ---
db = SQLAlchemy(app)

# --- Model Tabel ---
class HasilDeteksi(db.Model):
    __tablename__ = 'hasil_deteksi'  # Nama tabel eksplisit
    id = db.Column(db.Integer, primary_key=True)
    tanggal = db.Column(db.DateTime(timezone=True))  # Wajib timestamp with time zone di DB
    jumlah = db.Column(db.Integer)
    filename = db.Column(db.String(120))

# --- Fungsi deteksi rokok ---
def count_rokok(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return 0

    # Resize dan konversi HSV
    image = cv2.resize(image, (600, 400))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Seleksi warna cokelat khas batang rokok
    lower = np.array([10, 60, 50])
    upper = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Deteksi bentuk bulat
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if 0.6 < circularity < 1.3:
            count += 1

    return count

# --- Halaman Utama ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Upload & Simpan Deteksi ---
@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'Tidak ada file yang dipilih.'

    file = request.files['image']
    if file.filename == '':
        return 'Nama file kosong.'

    # Simpan file ke folder uploads
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Ambil waktu lokal WIB
    jakarta = pytz.timezone('Asia/Jakarta')
    waktu_local = datetime.now(jakarta)

    # Hitung jumlah batang rokok
    jumlah = count_rokok(filepath)

    # Simpan ke database
    deteksi = HasilDeteksi(
        tanggal=waktu_local,
        jumlah=jumlah,
        filename=filename
    )
    db.session.add(deteksi)
    db.session.commit()

    # Kirim hasil ke template
    return render_template(
        'result.html',
        filename=filename,
        jumlah=jumlah,
        tanggal=waktu_local.strftime('%Y-%m-%d %H:%M:%S')
    )

# --- Jalankan Aplikasi ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)