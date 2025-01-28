from picamera2 import Picamera2
import cv2
import numpy as np
import pygame
import RPi.GPIO as GPIO
import time

# Initialize the Pygame mixer module to handle sound playback.
pygame.mixer.init()

# --- Haptic Feedback Initialization ---
MOTOR_PIN = 10  # GPIO pin used to control the motor.
GPIO.setmode(GPIO.BCM)  # Set the GPIO mode to BCM (Broadcom SoC channel numbering).
GPIO.setup(MOTOR_PIN, GPIO.OUT)  # Configure the motor pin as an output.

pwm = GPIO.PWM(MOTOR_PIN, 100)  # Initialize PWM on the motor pin with a 100Hz frequency.
pwm.start(0)  # Start the PWM signal with a 0% duty cycle (motor off).

# Function to vibrate the motor with a specified intensity (0-100).
def vibrate_motor(intensity):
    pwm.ChangeDutyCycle(intensity)  # Change PWM duty cycle to control vibration strength.
    time.sleep(0.1)  # Delay to let the vibration effect be noticeable.

# Function to stop the motor vibration.
def stop_motor():
    pwm.ChangeDutyCycle(0)  # Set duty cycle to 0% (turn off the motor).

# --- Sound Playback ---
# Function to play a sound file using Pygame.
def play_sound(sound_file):
    sound = pygame.mixer.Sound(sound_file)  # Load the sound file.
    sound.play()  # Play the sound.
    time.sleep(sound.get_length())  # Wait for the sound to finish playing.

# --- Contour Filtering ---
# Function to filter contours based on their area.
def filter_contours(contours, min_area=500, max_area=15000):
    filtered = []  # List to store filtered contours.
    for cnt in contours:
        area = cv2.contourArea(cnt)  # Calculate the contour's area.
        if min_area < area < max_area:  # Check if the area is within the specified range.
            filtered.append(cnt)  # Add the contour to the filtered list.
    return filtered

# --- Square Detection ---
# Function to detect squares with thick borders in a given frame.
def detect_squares(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale.
    gray = cv2.GaussianBlur(gray, (7, 7), 0)  # Apply Gaussian blur to reduce noise.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Binary thresholding.
    kernel = np.ones((5, 5), np.uint8)  # Create a 5x5 kernel for morphological operations.
    dilated = cv2.dilate(thresh, kernel, iterations=2)  # Dilate the image to enhance contours.

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours.
    squares = []  # List to store detected squares.

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)  # Approximate the contour's shape.

        # Check if the contour has four vertices and is convex (indicating a square/rectangle).
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)  # Get bounding box of the contour.
            aspect_ratio = float(w) / h  # Calculate aspect ratio (width/height).
            area = cv2.contourArea(approx)  # Calculate the area of the contour.
            bounding_box_area = w * h  # Calculate the area of the bounding box.
            extent = area / bounding_box_area  # Calculate how much of the bounding box is filled by the contour.

            # Check if the contour is square-like based on aspect ratio and extent.
            if 0.75 <= aspect_ratio <= 1.25 and extent > 0.6:
                squares.append((approx, (x, y)))  # Save the square's contour and position.

    return squares  # Return the list of detected squares.

# --- Cross Detection ---
# Function to detect crosses with thick borders in a given frame.
def detect_crosses(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale.
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur to reduce noise.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Binary thresholding.
    kernel = np.ones((5, 5), np.uint8)  # Create a 5x5 kernel for morphological operations.
    dilated = cv2.dilate(thresh, kernel, iterations=2)  # Dilate the image to enhance contours.

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours.
    crosses = []  # List to store detected crosses.

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)  # Approximate the contour's shape.

        # Check if the contour has between 8 and 12 vertices (roughly cross-like).
        if 8 <= len(approx) <= 12:
            x, y, w, h = cv2.boundingRect(approx)  # Get bounding box of the contour.
            aspect_ratio = float(w) / h  # Calculate aspect ratio (width/height).

            # Check if the contour's shape is not convex (indicating possible cross).
            if 0.75 < aspect_ratio < 1.3 and not cv2.isContourConvex(approx):
                area = cv2.contourArea(cnt)  # Calculate the area of the contour.
                perimeter = cv2.arcLength(cnt, True)  # Calculate the perimeter of the contour.
                circularity = 4 * np.pi * (area / (perimeter ** 2))  # Measure circularity (to exclude circles).

                if circularity < 0.5:  # Ensure the shape is cross-like, not circular.
                    crosses.append(((x + w // 2, y), cnt))  # Save the center and contour.

    return crosses  # Return the list of detected crosses.

# Function to detect red lines in the frame.
def detect_red_lines(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert the frame to HSV color space.

    # Define HSV ranges for detecting red color.
    lower_red1 = np.array([0, 70, 50])  # Lower bound for the first red range.
    upper_red1 = np.array([10, 255, 255])  # Upper bound for the first red range.
    lower_red2 = np.array([170, 70, 50])  # Lower bound for the second red range (to account for hue wrapping).
    upper_red2 = np.array([180, 255, 255])  # Upper bound for the second red range.

    # Create masks for both red ranges.
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)  # Binary mask for the first red range.
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)  # Binary mask for the second red range.
    mask = cv2.bitwise_or(mask1, mask2)  # Combine both masks to capture all shades of red.

    # Enhance red lines using morphological operations.
    kernel = np.ones((5, 5), np.uint8)  # Create a 5x5 kernel for morphological operations.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small gaps in the red regions.
    mask = cv2.dilate(mask, kernel, iterations=2)  # Dilate to emphasize thick lines.

    cv2.imshow("Red Mask", mask)  # Debug: Display the red mask.

    # Detect edges in the mask using the Canny edge detector.
    edges = cv2.Canny(mask, 50, 150)

    # Detect straight lines using the Probabilistic Hough Transform.
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=80, maxLineGap=30)

    if lines is None:  # If no lines are detected, return None.
        return None, None

    # Extract vertical lines from the detected lines.
    vertical_lines = [line[0] for line in lines if abs(line[0][0] - line[0][2]) < 30]  # Filter near-vertical lines.

    if len(vertical_lines) < 2:  # If fewer than two vertical lines are found, return None.
        return None, None

    # Sort the lines by their x-coordinates to find the leftmost and rightmost lines.
    vertical_lines.sort(key=lambda l: l[0])  # Sort by the x-coordinate of the first point.
    left_line, right_line = vertical_lines[0], vertical_lines[-1]  # Select the outermost vertical lines.

    return left_line, right_line  # Return the left and right vertical lines.

def detect_carton_lines(frame):
    # Convert the frame to HSV color space.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for detecting "carton" color (brown shades).
    lower_brown = np.array([10, 100, 20])  # Lower bound for brown.
    upper_brown = np.array([20, 255, 200])  # Upper bound for brown.

    # Create a mask for the "carton" color.
    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Enhance the lines using morphological operations.
    kernel = np.ones((5, 5), np.uint8)  # Create a 5x5 kernel.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small gaps in the regions.
    mask = cv2.dilate(mask, kernel, iterations=2)  # Dilate to emphasize thick lines.

    cv2.imshow("Carton Mask", mask)  # Debug: Display the carton mask.

    # Detect edges in the mask using the Canny edge detector.
    edges = cv2.Canny(mask, 50, 150)

    # Detect straight lines using the Probabilistic Hough Transform.
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=80, maxLineGap=30)

    if lines is None:  # If no lines are detected, return None.
        return None

    # Extract horizontal lines from the detected lines.
    horizontal_lines = [line[0] for line in lines if abs(line[0][1] - line[0][3]) < 10]  # Filter near-horizontal lines.

    if not horizontal_lines:  # If no horizontal lines are found, return None.
        return None

    # Find the y-coordinates of all detected horizontal lines.
    y_coords = [line[1] for line in horizontal_lines]

    # Return the y-coordinate of the line closest to the bottom.
    return max(y_coords)


# Function to calculate the distance from the frame's center to the detected red lines.
def calculate_distance_to_lines(frame, left_line, right_line):
    height, width, _ = frame.shape  # Get the frame's dimensions (height, width, and channels).
    center_x = width // 2  # Calculate the horizontal center of the frame.

    if left_line is None or right_line is None:  # If any line is missing, return None.
        return None, None

    left_x = left_line[-1]  # X-coordinate of the left line.
    right_x = right_line[0]  # X-coordinate of the right line.

    # Calculate the horizontal distances from the center to each line.
    distance_left = abs(center_x - left_x)
    distance_right = abs(center_x - right_x)

    return distance_left, distance_right  # Return the distances to the left and right lines.

# Function to process a video frame by detecting and annotating objects of interest.
def process_frame(frame):
    squares = detect_squares(frame)  # Detect squares in the frame.
    crosses = detect_crosses(frame)  # Detect crosses in the frame.
    left_line, right_line = detect_red_lines(frame)  # Detect red lines.
    closest_carton_y = detect_carton_lines(frame)  # Detect the closest carton line.

    # Draw contours for detected squares.
    for cnt, (x, y) in squares:
        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)  # Draw the square in green.
        cv2.putText(frame, "Square", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # Label it.

    # Draw contours for detected crosses.
    for (cx, cy), cnt in crosses:
        cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 3)  # Draw the cross in red.
        cv2.putText(frame, "Cross", (cx - 30, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # Label it.

    # Draw red lines if detected.
    if left_line is not None and right_line is not None:
        cv2.line(frame, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (255, 0, 0), 2)  # Draw left line in blue.
        cv2.line(frame, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (255, 0, 0), 2)  # Draw right line in blue.
        cv2.putText(frame, "Red Line", (left_line[0], left_line[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, "Red Line", (right_line[0], right_line[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Draw the closest carton line if detected.
    if closest_carton_y is not None:
        frame_height, frame_width = frame.shape[:2]
        cv2.line(frame, (0, closest_carton_y), (frame_width, closest_carton_y), (0, 165, 255), 2)  # Draw carton line in orange.
        cv2.putText(frame, "Carton Line", (10, closest_carton_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)  # Label it.

    return frame  # Return the processed frame with annotations.

# Function to calculate the distance from the bottom of the frame to detected objects.
def distance_from_bottom(frame):
    height, _, _ = frame.shape  # Get the height of the frame.
    squares = detect_squares(frame)  # Detect squares.
    crosses = detect_crosses(frame)  # Detect crosses.

    distances = []  # List to store distances for each detected object.

    # Calculate distances for squares.
    for cnt, (x, y) in squares:
        _, _, _, h = cv2.boundingRect(cnt)  # Get the bounding box height of the square.
        bottom_y = y + h  # Calculate the Y-coordinate of the bottom of the square.
        distance = height - bottom_y  # Distance from the bottom of the frame.
        distances.append(("Square", x, bottom_y, distance))  # Save the shape type, position, and distance.

    # Calculate distances for crosses.
    for (cx, cy), cnt in crosses:
        _, _, _, h = cv2.boundingRect(cnt)  # Get the bounding box height of the cross.
        bottom_y = cy + h // 2  # Approximate the Y-coordinate of the bottom of the cross.
        distance = height - bottom_y # Distance from the bottom of the frame.
        distances.append(("Cross", cx, bottom_y, distance))  # Save the shape type, position, and distance.

    return distances  # Return a list of tuples with the shape type, position, and distance.

if __name__ == "__main__":
    picam2 = Picamera2()  # Initialize the Picamera2 object.
    picam2.configure(picam2.create_preview_configuration(  # Configure the camera for preview.
        main={"size": (320, 240), "format": "BGR888"},  # Set resolution and format.
        controls={"FrameRate": 60}  # Set frame rate to 60 FPS.
    ))

    picam2.start()  # Start the camera.

    try:
        while True:
            frame = picam2.capture_array()  # Capture a frame from the camera.
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR for OpenCV compatibility.

            processed_frame = process_frame(frame)  # Process the frame (detect shapes and lines).
            left_line, right_line = detect_red_lines(frame)
            distance_left, distance_right = calculate_distance_to_lines(frame, left_line, right_line)
            frame_center = processed_frame.shape[1] // 2
            if distance_left is not None:
                if distance_left > frame_center - 100 and distance_left < frame_center + 100:
                    vibrate_motor(100)
                    stop_motor()
            if distance_right is not None:
                if distance_right > frame_center - 100 and distance_right < frame_center + 100:
                    vibrate_motor(100)
                    stop_motor()
            
            min_y_carton = detect_carton_lines(frame)
            if min_y_carton is not None:  
                frame_height = processed_frame.shape[0]  
                jump_threshold = frame_height * 0.7 

                if min_y_carton > jump_threshold:  
                    play_sound("jump.mp3")  
            distances = distance_from_bottom(frame) 

            if distances:
                shape, x, y, dist = min((t for t in distances if t is not None), key=lambda t: t[3])
                if shape == "Cross":  # Check for crosses.
                    play_sound("cross.mp3")  # Play cross check sound.
                    frame_center = processed_frame.shape[1] // 2  # Horizontal center of the frame.
                    if x < frame_center - 50:
                        play_sound("left.mp3")  # Play left guidance sound.
                    elif x > frame_center + 50:
                        play_sound("right.mp3")  # Play right guidance sound.
                    else:
                        play_sound("straight.mp3")  # Play straight guidance sound.

                if shape == "Square":  # Check for squares.
                    play_sound("square.mp3")  # Play square check sound.
                    frame_center = processed_frame.shape[1] // 2  # Horizontal center of the frame.
                    if x < frame_center - 50:
                        play_sound("straight.mp3")  # Play straight guidance sound.
                    elif x > frame_center + 50:
                        play_sound("straight.mp3")  # Play straight guidance sound.
                    else:
                        play_sound("right.mp3")  # Play right guidance sound.


            cv2.imshow("Processed Frame", processed_frame)  # Show the processed frame.

            if cv2.waitKey(1) == 27:  # Wait for the 'Esc' key to exit.
                break

    finally:
        pwm.stop()  # Stop the PWM signal.
        GPIO.cleanup()  # Clean up GPIO pins.
        cv2.destroyAllWindows()  # Close all OpenCV windows.
