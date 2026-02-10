import os
import datetime
import numpy as np
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy

# Fix for Matplotlib
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from PIL import Image

# --- THE MONKEY PATCH (Fixes the Keras 3 Flatten bug) ---
import keras
from keras.layers import Flatten
original_compute_output_spec = Flatten.compute_output_spec
def patched_compute_output_spec(self, inputs, *args, **kwargs):
    if isinstance(inputs, list): inputs = inputs[0]
    return original_compute_output_spec(self, inputs, *args, **kwargs)
Flatten.compute_output_spec = patched_compute_output_spec
# -------------------------------------------------------

app = Flask(__name__)

# Configurations
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Load Model
MODEL_PATH = 'model/model.keras'
model = keras.models.load_model(MODEL_PATH, compile=False)

# Labels
LABELS = ['Glioma', 'Meningioma', 'notumor', 'Pituitary']

# --- Corrected Database Model ---
class Diagnosis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)        # Error was here (Missing Value)
    gender = db.Column(db.String(10), nullable=False)  # Error was here (Missing Value)
    result = db.Column(db.String(50), nullable=False)
    viz_path = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.now)

with app.app_context():
    db.create_all()

# --- Prediction & Visualization ---
def generate_report(img_path, viz_filename):
    img = Image.open(img_path).convert('RGB')
    resized_img = img.resize((299, 299))
    img_array = np.asarray(resized_img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    probs = list(predictions[0])
    result_index = np.argmax(probs)
    predicted_class = LABELS[result_index]

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(resized_img)
    plt.title(f'MRI - Predicted: {predicted_class.upper()}')
    plt.axis('off')

    plt.subplot(2, 1, 2)
    colors = ['#2ecc71' if x == 'notumor' else '#e74c3c' for x in LABELS]
    plt.barh(LABELS, probs, color=colors)
    plt.xlabel('Probability')
    plt.tight_layout()
    
    viz_full_path = os.path.join(app.config['UPLOAD_FOLDER'], viz_filename)
    plt.savefig(viz_full_path)
    plt.close()

    return predicted_class

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diagnosis', methods=['GET', 'POST'])
def diagnosis():
    if request.method == 'POST':
        # Capture ALL fields from the form
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        file = request.files.get('file')
        
        if file and name and age and gender:
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            
            ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            mri_fn = f"mri_{ts}.jpg"
            viz_fn = f"viz_{ts}.png"
            
            mri_path = os.path.join(app.config['UPLOAD_FOLDER'], mri_fn)
            file.save(mri_path)
            
            # Predict
            res = generate_report(mri_path, viz_fn)
            
            # SAVE ALL 4 FIELDS TO DB
            new_entry = Diagnosis(
                patient_name=name, 
                age=int(age), 
                gender=gender, 
                result=res, 
                viz_path=viz_fn
            )
            db.session.add(new_entry)
            db.session.commit()
            
            return render_template('result.html', name=name, age=age, result=res, viz_img=viz_fn)
            
    return render_template('diagnosis.html')

@app.route('/history')
def history():
    records = Diagnosis.query.order_by(Diagnosis.timestamp.desc()).all()
    return render_template('history.html', records=records)

if __name__ == '__main__':
    app.run(debug=True)