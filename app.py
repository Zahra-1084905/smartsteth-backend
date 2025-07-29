from flask import Flask, request, jsonify
import os
import uuid
import requests
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from keras.saving import register_keras_serializable
import tensorflow as tf
import keras
keras.config.enable_unsafe_deserialization()
import firebase_admin
from firebase_admin import credentials, storage, firestore

# NEW: Load service account path from environment variable
# cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Initialize Firebase app
if not firebase_admin._apps:
    cred = credentials.Certificate("/secrets/firebase-service-account")
    # cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'firebz-setup.appspot.com'
        # 'storageBucket': 'firebz-setup.firebasestorage.app'
    })

# cred = credentials.Certificate("serviceAccountKey.json")
# firebase_admin.initialize_app(cred, {
#     'storageBucket': 'firebz-setup.firebasestorage.app'  
# })


#registers for lung custom functions.
@register_keras_serializable(package="Custom", name="expand_dims_fn")
def expand_dims_fn(a):
    return tf.expand_dims(a, axis=-1)

@register_keras_serializable(package="Custom", name="sum_attention_fn")
def sum_attention_fn(x):
    return tf.reduce_sum(x, axis=1)

@register_keras_serializable(name="focal_loss_fixed")
def focal_loss_fixed(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.pow(1 - y_pred, gamma)
    loss = weight * cross_entropy
    return tf.reduce_mean(tf.reduce_sum(loss, axis=1))




from preprocess_heart_lstm import preprocess_heart_lstm_fn
from preprocess_lung_lstm import preprocess_lung_audio
import logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

diagnosis_results = {}

# Load AI models
HEART_MODEL_PATH = os.path.join("models", "best_lstm_flutter_ready4.keras")
LUNG_MODEL_PATH = os.path.join("models", "best_model_merged_3.keras")



heart_model = load_model(
    HEART_MODEL_PATH,
    
    compile=False
)



lung_model = load_model(
    LUNG_MODEL_PATH,
    custom_objects={
        "expand_dims_fn": expand_dims_fn,
        "sum_attention_fn": sum_attention_fn,
        "focal_loss_fixed": focal_loss_fixed
    }
)


print("Heart Model input shape:", heart_model.input_shape)
print("Lung Model input shape:", lung_model.input_shape)

# Class labels
heart_label_map = {
    0: "Aortic Stenosis",
    1: "Mitral Regurgitation",
    2: "Mitral Stenosis",
    3: "Mitral Valve Prolapse",
    4: "Normal"
}


lung_label_map = {
    0: "Asthma",
    1: "COPD",
    2: "Normal",
    3: "Pneumonia"

}
def upload_filtered_to_firebase(file_path, uid, type_tag):
    print("[DEBUG] Uploading filtered audio to Firebase...")
    bucket = storage.bucket()
    blob_path = f'filtered/{uid}/{type_tag}_{uuid.uuid4().hex}.wav'
    print(f"[DEBUG] Blob path: {blob_path}")
    
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(file_path)
    blob.make_public()
    print("[DEBUG] Upload complete. Public URL:", blob.public_url)
    return blob.public_url



@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    if not data or 'url' not in data or 'uid' not in data or 'type' not in data:
        return jsonify({"error": "Missing 'url', 'uid' or 'type' in request body"}), 400

    try:
        audio_url = data['url']
        uid = data['uid']
        audio_type = data['type']  
        temp_path = f"{uuid.uuid4().hex}.wav"

        # For Downloading the audio file
        r = requests.get(audio_url)
        with open(temp_path, 'wb') as f:
            f.write(r.content)

        print(f"[DEBUG] Downloaded audio saved to: {temp_path}")
        print(f"[DEBUG] Audio type: {audio_type}")

        
        if audio_type == "heart":
            model = heart_model
            label_map = heart_label_map
            x, filtered_path = preprocess_heart_lstm_fn(temp_path)
            filtered_url = upload_filtered_to_firebase(filtered_path, uid, audio_type)
            os.remove(filtered_path)
            
        

        elif audio_type == "lung":
            model = lung_model
            label_map = lung_label_map
            x, filtered_path = preprocess_lung_audio(temp_path)
            filtered_url = upload_filtered_to_firebase(filtered_path, uid, audio_type)
            os.remove(filtered_path)

            
        else:
            return jsonify({"error": f"Invalid audio type: {audio_type}"}), 400

        if not np.all(np.isfinite(x)):
            print("[ERROR] x has non-finite values!")
            return jsonify({"error": "Preprocessed spectrogram contains NaNs or Infs"})

        print("[DEBUG] Running model prediction...")
        print("[DEBUG] Input shape to model:", x.shape)
        predictions = model.predict(x)
        print("[DEBUG] Output shape from model:", predictions.shape)
        try:
            print("[DEBUG] Raw predictions:", predictions)
            average = np.mean(predictions, axis=0)
            print("[DEBUG] Prediction average:", average)

            if not np.all(np.isfinite(average)):
                raise ValueError(f"Non-finite values in prediction average: {average}")

            class_index = int(np.argmax(average))
            print("[DEBUG] Predicted class index:", class_index)

            confidence = float(average[class_index])
            print("[DEBUG] Prediction confidence:", confidence)

        except Exception as e:
            print("[ERROR] Post-prediction processing failed:", str(e))
            return jsonify({"error": "4"}), 500

        

        os.remove(temp_path)
        del x, predictions


        
        label = label_map[class_index]

        if uid not in diagnosis_results:
            diagnosis_results[uid] = {"heart": {}, "lung": {}}

        diagnosis_results[uid][audio_type] = {
            label: round(confidence * 100, 2)
        }

        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 4),
            "filtered_url": filtered_url
        })

    except Exception as e:
        import traceback
        traceback.print_exc()  
        return jsonify({"error": str(e)}), 500




@app.route('/get-diagnosis/<uid>', methods=['GET'])
def get_diagnosis(uid):
    full_lung = {"Normal": 0, "Pneumonia": 0, "COPD": 0, "Asthma": 0, "COVID-19": 0}
    full_heart = {
        "Normal": 0,
        "Atrial Stenosis": 0,
        "Mitral Stenosis": 0,
        "Mitral Regurgitation": 0,
        "Mitral Valve Prolapse": 0
    }

    if uid in diagnosis_results:
        lung = {**full_lung, **diagnosis_results[uid].get("lung", {})}
        heart = {**full_heart, **diagnosis_results[uid].get("heart", {})}
    else:
        lung = full_lung
        heart = full_heart

    return jsonify({
        "lung": lung,
        "heart": heart
    })

@app.route("/", methods=["GET"])
def index():
    return "SmartSteth backend is running!"


@app.route("/", methods=["GET"])
def index():
    return "SmartSteth backend is running!"

if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print("Flask server is starting...")
    port = int(os.environ.get("PORT", 8080)) 
    app.run(host="0.0.0.0", port=port, debug=True)

# if __name__ == "__main__":
#     import socket
#     hostname = socket.gethostname()
#     local_ip = socket.gethostbyname(hostname)
#     print("Flask server is starting...")
#     port = int(os.environ.get("PORT", 8080))
#     app.run(host="0.0.0.0", port=port, debug=True)

    # print(f"Flask server running at: http://{local_ip}:5000")
    # app.run(host='0.0.0.0', port=8080, debug=True)



