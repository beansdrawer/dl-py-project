import shutil
from typing import Optional, List
from fastapi import FastAPI,File, UploadFile
import tensorflow as tf 
from PIL import Image, ImageOps
import numpy as np
import os

os.makedirs("assets", exist_ok=True)
BASE_DIR = "./assets"

model = tf.keras.models.load_model('./model/keras_model.h5')

app = FastAPI()

@app.post("/detect_labels")
def detect_labels(file: UploadFile = File(...)):
    with open(os.path.join(BASE_DIR, file.filename), "wb") as fp:
        shutil.copyfileobj(file.file, fp)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(os.path.join(BASE_DIR, file.filename))
    size = (224, 224)
    image = ImageOps.fit(image, size)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    print(prediction)

        
    return {"Coffee": str(prediction[0][0]), "Monitor" :  str(prediction[0][1]) }