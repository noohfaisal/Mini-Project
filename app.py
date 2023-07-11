import streamlit as st
from PIL import Image
import numpy as np

class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('my_model2.hdf5')
    return model

def import_and_predict(image_data, model):
    size = (180, 180)
    image_data.thumbnail(size)
    image_array = np.array(image_data)
    img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Main app code
st.title("Flower Classification")

# Sidebar for input controls
st.sidebar.title("Upload Image")
file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])

# Load the model
model = load_model()

# Perform prediction if file is uploaded
if file is not None:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    score = tf.nn.softmax(prediction[0])
    st.write(prediction)
    st.write(score)
    disp = f"This image most likely belongs to {class_names[np.argmax(score)]} with a {100 * np.max(score):.2f} percent confidence."
    st.write(disp)
