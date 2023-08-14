from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np

app = Flask(__name__)

# Load the model
model = load_model('model.h5')

# Mapping for gender labels
gender_dict = {0: 'Male', 1: 'Female'}


# Utility function for preprocessing image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(128, 128), color_mode='grayscale')
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


@app.route('/', methods=['GET', 'POST'])
def index():
    gender = None
    age = None
    prediction_image = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error="No image uploaded.")

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error="No image selected.")

        if file:
            filename = os.path.join('static', 'uploads', file.filename)
            file.save(filename)

            # Preprocess the image
            image_array = preprocess_image(filename)

            # Predict gender and age
            gender_prob, age_value = model.predict(image_array)
            gender = gender_dict[round(gender_prob[0][0])]
            age = int(age_value[0][0])

            prediction_image = filename

    return render_template('index.html', gender=gender, age=age, prediction_image=prediction_image)


if __name__ == '__main__':
    app.run(debug=True)
