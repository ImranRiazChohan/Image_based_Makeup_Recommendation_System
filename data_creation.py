import os
import csv
import re

# Specify the directory containing the images
image_directory = 'Makeup Looks'

# Create a list to hold the image paths and names
image_data = []

# Function to remove numerical suffix from the image name
def clean_image_name(name):
    return re.sub(r'_\d+$', '', name)

# Loop through the files in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):  # Adjust extensions as needed
        file_path = os.path.join(image_directory, filename)
        image_name, _ = os.path.splitext(filename)  # Separate the name and extension
        cleaned_name = clean_image_name(image_name)  # Clean the image name
        image_data.append([file_path, cleaned_name])

# Specify the CSV file path
csv_file_path = 'makeup_face_recommendation.csv'

# Write the image data to a CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Image Path', 'Image Name'])  # Write header
    csvwriter.writerows(image_data)

print(f"CSV file '{csv_file_path}' has been created successfully.")