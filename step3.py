# Import necessary libraries
import pickle
import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pygame
import time
import os
import threading

# Load model from pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Create mediapipe hands object
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the hands object
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)

# Define the labels for the different hand gestures
labels_dict = {'1': 'r_mute',
               '2': 'r_kick',
               '3': 'r_closed_hat',
               '4': 'r_open_hat',
               '5': 'r_pedal_hat',
               '6': 'r_ride',
               '7': 'l_mute',
               '8': 'l_snare',
               '9': 'l_floor_tom',
               '10': 'l_mid_tom',
               '11': 'l_high_tom',
               '12': 'l_crash'}

# Initialize the predicted labels and accuracies
predicted_label_l = ''
predicted_label_r = ''
prediction_accuracy_l = 0.0
prediction_accuracy_r = 0.0

# Define cooldowns for left and right hands
cooldown_l = 0
cooldown_r = 0

# Cooldown time in seconds
cooldown_time = 0.5

# Initialize dictionaries to track the last time each pad was played
last_played_time_l = {}
last_played_time_r = {}

# Initialize Pygame for playing sounds
pygame.init()
pygame.mixer.init()

# Define drum pad sounds using Pygame
def play_drum_sound(sample_path):
    # Load the drum sample
    sample = pygame.mixer.Sound(sample_path)
    # Play the drum sample
    sample.play()

# Define the paths to the sound files for each drum pad
drum_sounds = {
    'r_kick': 'drum_sounds/kick.ogg',
    'r_closed_hat': 'drum_sounds/closed_hat.ogg',
    'r_open_hat': 'drum_sounds/open_hat.ogg',
    'r_pedal_hat': 'drum_sounds/pedal_hat.ogg',
    'r_ride': 'drum_sounds/ride_cymbal.ogg',
    'l_snare': 'drum_sounds/snare.ogg',
    'l_floor_tom': 'drum_sounds/floor_tom.ogg',
    'l_mid_tom': 'drum_sounds/mid_tom.ogg',
    'l_high_tom': 'drum_sounds/high_tom.ogg',
    'l_crash': 'drum_sounds/crash_cymbal.ogg'
}

# Define the display names for each drum pad
drum_pads = {
    'r_kick': 'Kick',
    'r_closed_hat': 'Closed Hi-Hat',
    'r_open_hat': 'Open Hi-Hat',
    'r_pedal_hat': 'Pedal Hi-Hat',
    'r_ride': 'Ride Cymbal',
    'l_snare': 'Snare',
    'l_floor_tom': 'Floor Tom',
    'l_mid_tom': 'Mid Tom',
    'l_high_tom': 'High Tom',
    'l_crash': 'Crash Cymbal'
}

highlighted_pads = {}

# Set the dimensions of the screen and grid for pygame
screen_width = 570
screen_height = 250

# Set the number of rows and columns in the grid
num_rows = 2
num_cols = 5

# Calculate the cell size based on the screen dimensions and number of rows/columns
cell_width = 100
cell_height = 100
cell_size = min(cell_width, cell_height)

# Set the padding between cells
padding = 10

# Calculate the actual grid size based on the cell size and padding
grid_width = cell_size * num_cols + padding * (num_cols + 1)
grid_height = cell_size * num_rows + padding * (num_rows + 1)

# Calculate the top-left position of the grid to center it on the screen
grid_x = (screen_width - grid_width) // 2
grid_y = (screen_height - grid_height) // 2

# Set the colors
very_light_grey = (250, 250, 250)
blue = (41, 128, 185)

# Set the corner radius for the cells
corner_radius = 8

# Set the thickness for the cell borders
border_thickness = 2

# Set the font style and size
font_size = 20
font = pygame.font.SysFont(None, font_size)

# Create the screen
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Drum Pad")

# Create the grid
grid = [['Kick', 'Closed Hi-Hat', 'Open Hi-Hat', 'Pedal Hi-Hat', 'Ride Cymbal'],
        ['Snare', 'Floor Tom', 'Mid Tom', 'High Tom', 'Crash Cymbal']]

# Helper function to get the coordinates of a cell
def get_cell_coordinates(letter):
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            if grid[row][col] == letter:
                return row, col

    return None

# Helper function to draw the grid
def draw_grid():
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            cell_x = grid_x + padding + col * (cell_size + padding)
            cell_y = grid_y + padding + row * (cell_size + padding)
            rect = pygame.Rect(cell_x, cell_y, cell_size, cell_size)

            # Fill the cells with the very light grey color (initial appearance)
            pygame.draw.rect(screen, very_light_grey, rect)

            # Draw the blue border around each cell
            pygame.draw.rect(screen, blue, rect, border_thickness, border_radius=corner_radius)

            # Get the cell name
            cell_name = grid[row][col]

            # Determine the text color based on whether the cell is highlighted or not
            text_color = blue  # Set the initial text color to blue

            # Render the cell name and center it within the cell
            text_surface = font.render(cell_name, True, text_color)
            text_rect = text_surface.get_rect(center=(cell_x + cell_size // 2, cell_y + cell_size // 2))
            screen.blit(text_surface, text_rect)

# Helper function to highlight a cell
def highlight_cell(row, col):
    cell_x = grid_x + padding + col * (cell_size + padding)
    cell_y = grid_y + padding + row * (cell_size + padding)
    rect = pygame.Rect(cell_x, cell_y, cell_size, cell_size)

    # Fill the highlighted cell with blue
    pygame.draw.rect(screen, blue, rect, border_radius=corner_radius)

    # Draw the blue border around the highlighted cell
    pygame.draw.rect(screen, blue, rect, border_thickness, border_radius=corner_radius)

    # Get the cell name
    cell_name = grid[row][col]

    # Render the cell name in very light grey and center it within the cell
    text_surface = font.render(cell_name, True, very_light_grey)
    text_rect = text_surface.get_rect(center=(cell_x + cell_size // 2, cell_y + cell_size // 2))
    screen.blit(text_surface, text_rect)
   
 
# Parameterized function to highlight a specific letter in the grid
def highlight_pad(playing_pad):
    # Fill the screen with the very light grey color
    screen.fill(very_light_grey)
    
    highlighted_pads[playing_pad] = time.time()

    # Draw the grid with the selected letter
    draw_grid()

    # Get the cell coordinates for the selected letter
    cell_coords = get_cell_coordinates(playing_pad)
    if cell_coords:
        row, col = cell_coords
        highlight_cell(row, col)

    # Update the display
    pygame.display.flip()

    # Wait for 0.5 seconds
    time.sleep(0.5)

    # Redraw the grid without highlighting any cell
    draw_grid()
    pygame.display.flip()
        
def update_gui():
    # Fill the screen with the very light grey color
    screen.fill(very_light_grey)

    # Draw the grid
    draw_grid()

    # Get the current time
    current_time = time.time()

    # Iterate over the copy of the dictionary to avoid RuntimeError due to changes during iteration
    for playing_pad, highlight_start_time in highlighted_pads.copy().items():
        # Check if the pad should still be highlighted
        if current_time - highlight_start_time < 0.5:
            # Get the cell coordinates for the selected letter
            cell_coords = get_cell_coordinates(playing_pad)
            if cell_coords:
                row, col = cell_coords
                highlight_cell(row, col)
        else:
            # Remove the pad from highlighted_pads if it has been highlighted for 0.5 seconds or more
            del highlighted_pads[playing_pad]
            
    # Remove pads from last_played_time dictionaries if cooldown time has passed
    for pad in list(last_played_time_l.keys()):
        if current_time - last_played_time_l[pad] >= cooldown_time:
            del last_played_time_l[pad]
    
    for pad in list(last_played_time_r.keys()):
        if current_time - last_played_time_r[pad] >= cooldown_time:
            del last_played_time_r[pad]

    # Update the display
    pygame.display.flip()

    # Wait for a short time to reduce CPU usage
    time.sleep(0.01)

# Main loop
while True:
    data_aux = []
    x_ = []
    y_ = []

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Flip the frame horizontally (to mirror the user's movements)
    frame = cv2.flip(frame, 1)

    # Get the shape of the frame
    H, W, _ = frame.shape

    # Convert the frame color from BGR to RGB (mediapipe uses RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to get the hand landmarks
    results = hands.process(frame_rgb)

    # If hand landmarks are found
    if results.multi_hand_landmarks:

        # Draw the landmarks and connections on the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Iterate through hand landmarks
        for i in range(len(results.multi_hand_landmarks)):
            hand_landmarks = results.multi_hand_landmarks[i]

            # Record x, y coordinates for each landmark
            for j in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[j].x
                y = hand_landmarks.landmark[j].y

                x_.append(x)
                y_.append(y)

            # Create data for prediction by getting the normalized position of landmarks
            for j in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[j].x
                y = hand_landmarks.landmark[j].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Define bounding box for hand
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Predict gesture from hand landmark data
            prediction = model.predict([np.asarray(data_aux)])
            prediction_proba = model.predict_proba([np.asarray(data_aux)])

            # Get predicted gesture label and accuracy
            predicted_gesture = labels_dict[prediction[0]]
            prediction_accuracy = np.max(prediction_proba) * 100

            # Assign gesture and accuracy to left or right hand
            if labels_dict[prediction[0]].startswith('l'):
                predicted_label_l = labels_dict[prediction[0]]
                prediction_accuracy_l = prediction_accuracy
            elif labels_dict[prediction[0]].startswith('r'):
                predicted_label_r = labels_dict[prediction[0]]
                prediction_accuracy_r = prediction_accuracy

            # Draw bounding box around hand on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display predicted gesture and accuracy on the frame
            if (predicted_gesture != 'l_mute' and predicted_gesture != 'r_mute'):
                cv2.putText(frame, f"{predicted_gesture} ({prediction_accuracy:.2f}%)",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Reset data for the next hand
            data_aux = []
            x_ = []
            y_ = []

    # Get the current time        
    current_time = time.time()
    
    # If accuracy is above 85%, play the drum sound corresponding to the left hand gesture
    if prediction_accuracy_l > 85.0:
        if predicted_label_l in drum_sounds and predicted_label_l not in last_played_time_l:
            # Start new thread for playing the left hand sound
            thread_l = threading.Thread(target=play_drum_sound, args=(drum_sounds[predicted_label_l],))
            thread_l.start()
            last_played_time_l[predicted_label_l] = current_time
            thread_l_pad = threading.Thread(target=highlight_pad, args=(drum_pads[predicted_label_l],))
            thread_l_pad.start()

    # If accuracy is above 85%, play the drum sound corresponding to the right hand gesture
    if prediction_accuracy_r > 85.0:
        if predicted_label_r in drum_sounds and predicted_label_r not in last_played_time_r:
            # Start new thread for playing the right hand sound
            thread_r = threading.Thread(target=play_drum_sound, args=(drum_sounds[predicted_label_r],))
            thread_r.start()
            last_played_time_r[predicted_label_r] = current_time
            thread_r_pad = threading.Thread(target=highlight_pad, args=(drum_pads[predicted_label_r],))
            thread_r_pad.start()

    # Update GUI
    update_gui()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # If 'q' is pressed on the keyboard, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture when everything done
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# PyGame loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the screen with the very light grey color
    screen.fill(very_light_grey)
    # Draw the grid
    draw_grid()
    
    # Highlight all active pads
    for pad in active_pads:
        cell_coords = get_cell_coordinates(pad)
        if cell_coords:
            row, col = cell_coords
            highlight_cell(row, col)

    # Update the display
    pygame.display.flip()

    # Wait for 0.5 seconds
    time.sleep(0.5)

# Quit the program
pygame.quit()
