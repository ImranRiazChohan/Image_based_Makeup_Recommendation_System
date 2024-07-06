# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# # Define the function to preprocess the image
# def preprocess_image(img_path, img_size):
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read the image directly as grayscale
#     resized_img = cv2.resize(img, (img_size[0], img_size[1]))  # Resize to img_size x img_size
#     normalized_img = resized_img / 255.0  # Normalize to [0, 1]
#     return normalized_img


# # Define the function to predict the class of the image
# def predict_image(model, img_path, img_size, labels):
#     # Preprocess the image
#     preprocessed_img = preprocess_image(img_path, img_size)
#     # Expand dimensions to match the input shape expected by the model
#     img_array = np.expand_dims(preprocessed_img, axis=-1)  # Expand the last dimension
#     img_array = np.expand_dims(img_array, axis=0)  # Expand the batch dimension
#     img_array/=255.0
    
#     # Make prediction
#     prediction = model.predict(img_array)
#     # Interpret prediction
#     prediction_class=np.argmax(prediction[0])
#     predicted_class = labels[prediction_class]
#     return predicted_class, prediction[0]

# # Load Model file 
# model_path = './face_model_v1.h5'  # Adjust this path if necessary
# model = load_model(model_path)
# img_size = [650, 450]
# labels = ['heart_caramel', 'heart_fair',"oval_caramel","oval_fair","round_fair","round_tan","square_fair","square_wheatish"]

# # Load the CSV file
# csv_file_path = 'makeup_face_recommendation.csv'
# image_data = pd.read_csv(csv_file_path)

# # Function to recommend images based on user input using cosine similarity
# def recommend_images(input_name, data, top_n=5):
#     # Combine input name with existing image names
#     names = data['Image Name'].tolist()
#     names.append(input_name)
    
#     # Convert names to TF-IDF vectors
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(names)
    
#     # Compute cosine similarity
#     cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
#     # Get top N similar images
#     similar_indices = cosine_sim[0].argsort()[-top_n:][::-1]
#     similar_items = data.iloc[similar_indices]
    
#     return similar_items

# # Streamlit app
# st.title('Image Recommendation System')

# # User input for the image name
# # user_input = st.text_input('Enter image name:')

# user_input=st.file_uploader("Upload Image",type=['jpeg'])

# # Display recommendations when user enters a name
# if user_input is not None:

#     predicted_class,predicted_prob= predict_image(model, user_input, img_size, labels)
    
#     recommendations = recommend_images(predicted_class, image_data)
#     if not recommendations.empty:
#         st.write(f"Top {len(recommendations)} matching images:")
#         for index, row in recommendations.iterrows():
#             st.write(f"Image Path: {row['Image Path']}")
#             st.image(row['Image Path'])  # Display the image
#     else:
#         st.write('No matching images found.')






# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from io import BytesIO

# # Define the function to preprocess the image
# def preprocess_image(img, img_size):
#     resized_img = cv2.resize(img, (img_size[0], img_size[1]))  # Resize to img_size x img_size
#     normalized_img = resized_img / 255.0  # Normalize to [0, 1]
#     return normalized_img

# # Function to read image from BytesIO and convert to grayscale
# def read_image_from_bytesio(file):
#     image = np.asarray(bytearray(file.read()), dtype=np.uint8)
#     image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
#     return image

# # Define the function to predict the class of the image
# def predict_image(model, img, img_size, labels):
#     # Preprocess the image
#     preprocessed_img = preprocess_image(img, img_size)
#     # Expand dimensions to match the input shape expected by the model
#     img_array = np.expand_dims(preprocessed_img, axis=-1)  # Expand the last dimension
#     img_array = np.expand_dims(img_array, axis=0)  # Expand the batch dimension
#     img_array /= 255.0
    
#     # Make prediction
#     prediction = model.predict(img_array)
#     # Interpret prediction
#     prediction_class = np.argmax(prediction[0])
#     predicted_class = labels[prediction_class]
#     return predicted_class, prediction[0]

# # Load Model file 
# model_path = './face_model_v1.h5'  # Adjust this path if necessary
# model = load_model(model_path)
# img_size = [650, 450]
# labels = ['heart_caramel', 'heart_fair', 'oval_caramel', 'oval_fair', 'round_fair', 'round_tan', 'square_fair', 'square_wheatish']

# # Load the CSV file
# csv_file_path = 'makeup_face_recommendation.csv'
# image_data = pd.read_csv(csv_file_path)

# # Function to recommend images based on user input using cosine similarity
# def recommend_images(input_name, data, top_n=5):
#     # Combine input name with existing image names
#     names = data['Image Name'].tolist()
#     names.append(input_name)
    
#     # Convert names to TF-IDF vectors
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(names)
    
#     # Compute cosine similarity
#     cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
#     # Get top N similar images
#     similar_indices = cosine_sim[0].argsort()[-top_n:][::-1]
#     similar_items = data.iloc[similar_indices]
    
#     return similar_items

# # Streamlit app
# st.title('Image Recommendation System')

# # User input for the image upload
# user_input = st.file_uploader("Upload Image", type=['jpeg', 'jpg'])

# # Display recommendations when user uploads an image
# if user_input is not None:
#     # Read the uploaded image
#     uploaded_image = read_image_from_bytesio(user_input)
    
#     predicted_class, predicted_prob = predict_image(model, uploaded_image, img_size, labels)
#     print(predicted_class)
    
#     recommendations = recommend_images(predicted_class, image_data)
#     if not recommendations.empty:
#         st.write(f"Top {len(recommendations)} matching images:")
#         for index, row in recommendations.iterrows():
#             st.write(f"Image Path: {row['Image Path']}")
#             st.image(row['Image Path'])  # Display the image
#     else:
#         st.write('No matching images found.')










import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the CSV files
image_data = pd.read_csv('makeup_face_recommendation.csv')
makeup_data = pd.read_csv('makeup_items\Makeup Items.csv')

# Function to recommend images based on user input using cosine similarity
def recommend_images(input_name, data, top_n=5):
    names = data['Image Name'].tolist()
    names.append(input_name)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(names)
    
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    similar_indices = cosine_sim[0].argsort()[-top_n:][::-1]
    similar_items = data.iloc[similar_indices]
    
    return similar_items

# Function to recommend makeup items for a selected image
def recommend_makeup_items(image_path, data):
    return data[data['Image Path'] == image_path]

# Streamlit app
st.title('Image Recommendation System')

# User input for the skin tone and shape
user_input = st.text_input('Enter skin tone and shape:')


# Display recommendations when user enters a skin tone and shape
if user_input:
    st.write(f"User input: {user_input}")
    recommendations = recommend_images(user_input, image_data)
    
    if not recommendations.empty:
        st.write(f"Top {len(recommendations)} matching images:")
        for index, row in recommendations.iterrows():
            st.write(f"Image Path: {row['Image Path']}")
            st.image(row['Image Path'])  # Display the image
            print(row['Image Path'])
            
            if st.button(f"Show makeup items for {row['Image Path']}", key=index):
                makeup_recommendations = recommend_makeup_items(row['Image Path'], makeup_data)
                print(makeup_recommendations)
                if not makeup_recommendations.empty:
                    st.write(f"Makeup items for {row['Image Path']}:")
                    for idx, m_row in makeup_recommendations.iterrows():
                        st.write(f"Makeup Item: {m_row['Item_name']}")
                        
                        # print("'"+m_row["Item_image_path"]+"'")
                        item=m_row["Item_image_path"]
                        st.image(item, caption=f"Brand: {m_row['Brand Name']}")
                else:
                    st.write('No makeup items found.')
    else:
        st.write('No matching images found.')