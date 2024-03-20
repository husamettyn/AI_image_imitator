from PIL import Image
import numpy as np

def create_circle_matrix(image_path):
    # Get the size of the image
    image = Image.open(image_path)
    width, height = image.size
    
    # Convert the image to grayscale
    grayscale_image = image.convert('L')
    # Convert the grayscale image to a numpy array
    image_array = np.array(grayscale_image)
    # Normalize the pixel values to the range [0, 1]
    normalized_image = image_array / 255.0
    
    # Create a matrix of zeros with the same size as the image
    matrix = np.zeros((height, width))

    return matrix, normalized_image

def get_angle_pixel_coordinates(matrix, angle):
    # Get the dimensions of the matrix
    height, width = matrix.shape
    
    # Calculate the center coordinates of the circle
    center_x = width // 2
    center_y = height // 2
    
    # Calculate the radius of the circle
    radius = min(center_x, center_y)
    
    # Calculate the angle in radians
    angle_rad = np.deg2rad(angle)
    
    # Calculate the x and y coordinates of the pixel for the given angle
    x = int(center_x + radius * np.cos(angle_rad))
    y = int(center_y + radius * np.sin(angle_rad))
    
    return x, y

def draw_circle(matrix):
    for i in range(0, 360):
        x, y = get_angle_pixel_coordinates(matrix, i)
        if 0 <= y < matrix.shape[0] and 0 <= x < matrix.shape[1]:
            matrix[y, x] = 1
    
    return matrix

def draw_line(matrix, start_x, start_y, end_x, end_y):
    # Calculate the difference between the start and end coordinates
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    
    # Calculate the number of steps needed to draw the line
    steps = max(abs(delta_x), abs(delta_y))
    
    # Calculate the step size for each coordinate
    step_x = delta_x / steps
    step_y = delta_y / steps
    
    # Draw the line by updating the matrix values
    x = start_x
    y = start_y
    for _ in range(steps):
        if 0 <= y < matrix.shape[0] and 0 <= x < matrix.shape[1]:
            matrix[int(y), int(x)] = 1
        x += step_x
        y += step_y
    
    return matrix

def draw_path(matrix, path):
    for i in range(len(path) - 1):
        start_x, start_y = get_angle_pixel_coordinates(matrix, path[i])
        end_x, end_y = get_angle_pixel_coordinates(matrix, path[i + 1])
        matrix = draw_line(matrix, start_x, start_y, end_x, end_y)
    
    return matrix

if __name__ == "__main__":
    image_path = 'images/1.png'
    matrix, normalized_image = create_circle_matrix(image_path)
    path = [0, 45, 90, 135, 180, 225, 270, 315, 0]
    angle2 = 45

    draw_path(matrix, path)
    draw_circle(matrix)
    
    # Convert the matrix back to an image
    circle_image = Image.fromarray((matrix * 255).astype(np.uint8))

    # Display the image
    circle_image.show()