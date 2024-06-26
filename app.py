import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import numpy as np
import tensorflow as tf

# Define the function to preprocess the image
def preprocess_image(img_path, img_size):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read the image directly as grayscale
    resized_img = cv2.resize(img, (img_size[0], img_size[1]))  # Resize to img_size x img_size
    normalized_img = resized_img / 255.0  # Normalize to [0, 1]
    return normalized_img


# Define the function to predict the class of the image
def predict_image(model, img_path, img_size, labels):
    # Preprocess the image
    preprocessed_img = preprocess_image(img_path, img_size)
    # Expand dimensions to match the input shape expected by the model
    img_array = np.expand_dims(preprocessed_img, axis=-1)  # Expand the last dimension
    img_array = np.expand_dims(img_array, axis=0)  # Expand the batch dimension
    img_array/=255.0
    
    # Make prediction
    prediction = model.predict(img_array)
    # Interpret prediction
    prediction_class=np.argmax(prediction[0])
    predicted_class = labels[prediction_class]
    return predicted_class, prediction[0]

# Load Model file 
model_path = './face_model_v1.h5'  # Adjust this path if necessary
model = tf.keras.models.load_model(model_path)
img_size = [650, 450]
labels = ['heart_caramel', 'heart_fair',"oval_caramel","oval_fair","round_fair","round_tan","square_fair","square_wheatish"]

# Load the CSV file
csv_file_path = 'makeup_face_recommendation.csv'
image_data = pd.read_csv(csv_file_path)

# Function to recommend images based on user input using cosine similarity
def recommend_images(input_name, data, top_n=5):
    # Combine input name with existing image names
    names = data['Image Name'].tolist()
    names.append(input_name)
    
    # Convert names to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(names)
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    # Get top N similar images
    similar_indices = cosine_sim[0].argsort()[-top_n:][::-1]
    similar_items = data.iloc[similar_indices]
    
    return similar_items

# Streamlit app
st.title('Image Recommendation System')

# User input for the image name
# user_input = st.text_input('Enter image name:')

user_input=st.file_uploader("Upload Image",type=['jpeg'])

# Display recommendations when user enters a name
if user_input is not None:

    predicted_class,predicted_prob= predict_image(model, user_input, img_size, labels)
    
    recommendations = recommend_images(predicted_class, image_data)
    if not recommendations.empty:
        st.write(f"Top {len(recommendations)} matching images:")
        for index, row in recommendations.iterrows():
            st.write(f"Image Path: {row['Image Path']}")
            st.image(row['Image Path'])  # Display the image
    else:
        st.write('No matching images found.')

