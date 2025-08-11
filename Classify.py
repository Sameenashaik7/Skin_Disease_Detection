# ml/classify.py
import os
import io
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

MODEL_PATH = os.environ.get("MODEL_PATH", "ml/models/efficientnetb0_best.h5")
CLASS_INDEX = {0: "Eczema", 1: "Psoriasis", 2: "Acne", 3: "Ringworm", 4: "Melasma", 5: "Normal"}  # update to match your labels

# ----------------------
# Preprocessing helpers
# ----------------------
def remove_hair(img_bgr):
    """Simple DullRazor-like hair removal: inpaint hair mask."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # blackhat to detect hair-like structures
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    # dilate mask
    mask = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)
    # inpaint the hair regions
    inpainted = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)
    return inpainted

def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def preprocess_pil_image(pil_img, target_size=(224,224)):
    # Convert to BGR for OpenCV operations
    img = np.array(pil_img.convert("RGB"))[:, :, ::-1].copy()
    img = remove_hair(img)
    img = apply_clahe(img)
    img = cv2.resize(img, target_size)
    # Convert back to RGB and then to model preprocess
    img_rgb = img[:, :, ::-1]
    img_arr = tf.keras.preprocessing.image.img_to_array(img_rgb)
    img_arr = eff_preprocess(img_arr)  # EfficientNet preprocessing
    return img_arr

# ----------------------
# Model & inference
# ----------------------
_model = None
_last_loaded_path = None

def get_model():
    global _model, _last_loaded_path
    if _model is None or _last_loaded_path != MODEL_PATH:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Place your .h5 there.")
        _model = load_model(MODEL_PATH, compile=False)
        _last_loaded_path = MODEL_PATH
    return _model

def predict(pil_img, top_k=3):
    """
    Input: PIL image
    Output: dict with top_k predictions and confidences + gradcam path
    """
    model = get_model()
    x = preprocess_pil_image(pil_img, target_size=(224,224))
    x_batch = np.expand_dims(x, axis=0)
    preds = model.predict(x_batch)[0]
    # top_k
    idx = np.argsort(preds)[::-1][:top_k]
    results = [{"label": CLASS_INDEX.get(int(i), str(i)), "confidence": float(preds[int(i)])} for i in idx]
    # generate gradcam
    cam_path = generate_gradcam(model, x_batch, idx[0])
    return {"predictions": results, "gradcam": cam_path}

# ----------------------
# Grad-CAM
# ----------------------
def generate_gradcam(model, processed_input, class_index, out_path="ml/gradcam_out.jpg"):
    """
    Very small Grad-CAM for Keras functional models.
    processed_input: preprocessed batch with shape (1, H, W, C)
    class_index: scalar index for target class
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(index=-1).output, model.layers[-5].output]
    )
    with tf.GradientTape() as tape:
        conv_outputs = grad_model(processed_input)[1]
        tape.watch(conv_outputs)
        predictions = grad_model(processed_input)[0]
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    # resize heatmap to 224x224 and overlay on original
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # recover original image (de-preprocessed roughly)
    inp = processed_input[0]
    # reverse EfficientNet preprocessing (approx)
    inp = inp.copy()
    inp = (inp + 1.0) * 127.5
    inp = np.uint8(np.clip(inp, 0, 255))
    # overlay
    overlay = cv2.addWeighted(inp, 0.6, heatmap_color, 0.4, 0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return out_path
