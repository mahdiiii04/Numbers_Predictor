from flask import Flask, render_template, request
import base64
import pickle
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

plt.show()

def rgba_to_rgb(rgba_image):
    width, height = rgba_image.size
    rgb_image = Image.new('RGB', (width, height), (255, 255, 255))

    for x in range(width):
        for y in range(height):
            r, g, b, a = rgba_image.getpixel((x, y))
            blended_pixel = (
                int(r * (a / 255.0) + 255 * (1 - a / 255.0)),
                int(g * (a / 255.0) + 255 * (1 - a / 255.0)),
                int(b * (a / 255.0) + 255 * (1 - a / 255.0))
            )
            rgb_image.putpixel((x, y), blended_pixel)

    return rgb_image






model = pickle.load(open('model.p', 'rb'))['model']


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_data = request.form.get('Data')

    base64_data = img_data.replace('data:image/png;base64,', '')

    binary_data = base64.b64decode(base64_data)

    img = Image.open(BytesIO(binary_data))


    rgb_img = rgba_to_rgb(img)
    
    rgb_array = image.img_to_array(rgb_img)
    input = rgb_array / 255.0
    

    prediction = model.predict(np.asarray([input]))
    predicted_num = prediction.argmax(axis=1)[0]


    return render_template('index.html', prediction_result=predicted_num,wait='')

    

if __name__ == '__main__':
    app.run(debug=True)