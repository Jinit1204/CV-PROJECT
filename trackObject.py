import numpy as np
import cv2

# Initialize webcam capture
capture = cv2.VideoCapture(0)

# Define the color range of the object in HSV
color_range = [[0, 0, 0], [179, 50, 100]]

points = []
count = 0

# Get the first frame to obtain dimensions
response, frame = capture.read()
height, width = frame.shape[:2]

while True:
    # Capture a frame
    response, frame = capture.read()
    frame = cv2.flip(frame, 1)
    
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask with the defined color range
    mask = cv2.inRange(hsv, np.array(color_range[0]), np.array(color_range[1]))

    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize the centroid and radius
    centroids = (int(width / 2), int(height / 2))
    radius = 0

    if contours:
        # Find the largest contour
        biggest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(biggest_contour)
        M = cv2.moments(biggest_contour)
        
        if M['m00'] != 0:
            centroids = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        else:
            centroids = (int(width / 2), int(height / 2))

        # Draw the largest contour and its centroid if the radius is above a threshold
        if radius > 25:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            cv2.circle(frame, centroids, 5, (0, 255, 0), -1)

    points.append(centroids)
    
    # Clear the points if no object is detected for 10 frames
    if radius <= 25:
        count += 1
        if count == 10:
            points = []
            count = 0

    # Display the frame
    cv2.imshow('Tracked Object', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()
