import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import json

# Load the image
wall = cv.imread('morphing_wall_1118_13_0.05_aligned.png', cv.IMREAD_GRAYSCALE)
window = cv.imread('morphing_window_1118_10.4_0.05_aligned.png', cv.IMREAD_GRAYSCALE)
column = cv.imread('morphing_column_1118_10.18_0.05_aligned.png', cv.IMREAD_GRAYSCALE)

# Create the LineSegmentDetector object
lsd = cv.createLineSegmentDetector(0)  # 0 is the scale parameter

# Detect lines in the image
lines_wall, _, _, _ = lsd.detect(wall)
lines_window, _, _, _ = lsd.detect(window)
lines_column, _, _, _ = lsd.detect(column)

# Create an empty white image
img = np.ones((wall.shape[0], wall.shape[1], 3), dtype=np.uint8) * 255

# Store line segment information
line_segments_info = []

# Function to draw a red dot at the specified point
def draw_red_dot(image, point):
    cv.circle(image, point, 2, (0, 0, 255), -1)

# Function to draw a line segment on the image
def draw_line_segment(image, line, color):
    x1, y1, x2, y2 = map(int, line[0])
    cv.line(image, (x1, y1), (x2, y2), color, 2)

    # Save start and end points to the list
    line_segments_info.append({
        'start_point': {'x': x1, 'y': y1},
        'end_point': {'x': x2, 'y': y2},
        'color': color
    })

    # Draw red dots at start and end points
    draw_red_dot(image, (x1, y1))
    draw_red_dot(image, (x2, y2))

# Draw line segments on the image
for line in lines_wall:
    draw_line_segment(img, line, (0, 0, 0))

for line in lines_window:
    draw_line_segment(img, line, (241, 193, 153))

for line in lines_column:
    draw_line_segment(img, line, (0, 255, 255))

# Save the image with detected lines and red dots
#cv.imwrite('image_with_lines_and_dots.png', img)
cv.imwrite('image_with_lines_column.png', img)

# Save line segments information to a JSON file
with open('line_segments_info.json', 'w') as json_file:
    json.dump(line_segments_info, json_file, indent=4)

plt.imshow(img)
plt.title('Vectorized floor plan')
plt.show()

