# Import necessary libraries
import os
import pickle
import mediapipe as mp
import cv2

# Define mediapipe components for hand recognition
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Function to process images from a given dataset
def process_images(dataset_path):
    # Initialize hand recognition with certain parameters
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    data = []    # Empty list to store processed data
    labels = []  # Empty list to store labels

    # Iterate over each directory in the dataset
    for dir_ in os.listdir(dataset_path):
        # Iterate over each image in the directory
        for img_path in os.listdir(os.path.join(dataset_path, dir_)):
            # Initialize empty lists for auxiliary data and x and y coordinates
            data_aux = []
            x_ = []
            y_ = []

            # Read image and convert it to RGB
            img = cv2.imread(os.path.join(dataset_path, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process image to detect hands
            results = hands.process(img_rgb)

            # If hands are detected, record their coordinates
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    # Store normalized coordinates in auxiliary data list
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                # Append auxiliary data and corresponding label to main lists
                data.append(data_aux)
                labels.append(dir_)

    # Return processed data and corresponding labels
    return data, labels

# Function to save data to a file
def save_data(data, labels, output_file):
    # Open file in write-binary mode and save data
    with open(output_file, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

# Define path to dataset and output file
dataset_path = './dataset'
output_file = 'data.pickle'

# Process images in the dataset and save the resulting data
data, labels = process_images(dataset_path)
save_data(data, labels, output_file)
