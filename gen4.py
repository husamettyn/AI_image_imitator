import time
from PIL import Image
import numpy as np
import random

def create_circle_matrix(image_path):
    # Get the size of the image
    image = Image.open(image_path)
    width, height = image.size

    # Convert the image to grayscale
    grayscale_image = image.convert('L')
    # Convert the grayscale image to a numpy array
    image_array = np.array(grayscale_image)

    # Create a matrix of zeros with the same size as the image
    matrix = np.ones((height, width))

    return matrix * 255, image_array

def get_angle_pixel_coordinates(matrix, angles):
    height, width = matrix.shape
    # Calculate the center coordinates of the circle
    center_x = width // 2
    center_y = height // 2
    radius = min(center_x, center_y)

    # Calculate the angle in radians
    angles_rad = np.deg2rad(angles)
    # Calculate the x and y coordinates of the pixels for the given angles
    x = (center_x + radius * np.cos(angles_rad)).astype(int)
    y = (center_y + radius * np.sin(angles_rad)).astype(int)

    return x, y

def draw_circle(matrix):
    center_x, center_y = matrix.shape[1] // 2, matrix.shape[0] // 2
    radius = min(center_x, center_y)

    angles = np.deg2rad(np.arange(0, 360))
    x = (center_x + radius * np.cos(angles)).astype(int)
    y = (center_y + radius * np.sin(angles)).astype(int)
    matrix[y, x] = 0

    return matrix

def draw_line(matrix, start_x, start_y, end_x, end_y):
    decrease = 20
    # Calculate the difference between the start and end coordinates
    delta_x = end_x - start_x
    delta_y = end_y - start_y

    # Calculate the number of steps needed to draw the line
    steps = max(abs(delta_x), abs(delta_y))

    # Check if steps is zero to avoid division by zero
    if steps != 0:
        # Calculate the step size for each coordinate
        step_x = delta_x / steps
        step_y = delta_y / steps
    else:
        step_x = 0
        step_y = 0

    # Draw the line by updating the matrix values
    x = start_x
    y = start_y
    for _ in range(steps):
        if 0 <= y < matrix.shape[0] and 0 <= x < matrix.shape[1]:
            matrix[int(y), int(x)] -= decrease
        x += step_x
        y += step_y

    return matrix

def draw_path(matrix, path):
    xs, ys = get_angle_pixel_coordinates(matrix, path)
    for i in range(len(xs) - 1):
        matrix = draw_line(matrix, xs[i], ys[i], xs[i + 1], ys[i + 1])

    return matrix

def initpath(path_number):
    # Initialize a path with random angles
    path = np.random.randint(0, 359, size=path_number)
    return path

def fitness(image1, image2):
    # Calculate Mean Squared Error (MSE) between two images
    return np.mean((image1 - image2) ** 2)

def crossover(path1, path2):
    # Perform crossover between two paths
    crossover_point = random.randint(0, len(path1) - 1)
    new_path = np.concatenate((path1[:crossover_point], path2[crossover_point:]))
    return new_path

def mutate(path, mutation_rate):
    # Perform mutation on a path
    mutation_indices = np.random.rand(len(path)) < mutation_rate
    path[mutation_indices] = np.random.randint(0, 359, size=np.sum(mutation_indices))
    return path


if __name__ == "__main__":
    target_image_path = "images/1.png"
    matrix, target_image = create_circle_matrix(target_image_path)

    population_size = 100
    mutation_rate = 0.07
    generations = 100

    # Initialize population with a numpy array of paths
    population = np.array([initpath(200) for _ in range(population_size)])

    generation_times = []  # List to hold the time taken for each generation

    for generation in range(generations):
        start_time = time.time()  # Start the timer

        # Evaluate fitness of each path in the population
        fitness_scores = np.array([fitness(draw_path(matrix.copy(), path), target_image) for path in population])

        # Select paths based on their fitness
        selected_indices = np.argsort(fitness_scores)[:population_size // 2]
        selected_paths = population[selected_indices]

        # Create new paths through crossover
        new_population = []
        while len(new_population) < population_size:
            path1 = selected_paths[np.random.randint(len(selected_paths))]
            path2 = selected_paths[np.random.randint(len(selected_paths))]
            new_path = crossover(path1, path2)
            new_population.append(new_path)

        # Convert new_population list back to numpy array for efficient mutation
        new_population = np.array(new_population)

        # Mutate new population
        population = np.array([mutate(path, mutation_rate) for path in new_population])

        end_time = time.time()  # Stop the timer
        generation_time = end_time - start_time  # Calculate the time taken for the current generation
        generation_times.append(generation_time)  # Add the time to the list

        print(f"Generation {generation} best fitness: {min(fitness_scores)}")

    average_time = sum(generation_times) / len(generation_times)  # Calculate the average time
    print(f"Average time per generation: {average_time} seconds")

    # Get the best path (image) from the final population
    best_fitness_index = np.argmin([fitness(draw_path(matrix.copy(), path), target_image) for path in population])
    best_path = population[best_fitness_index]

    # Draw the best path
    best_image = draw_path(matrix.copy(), best_path)

    normalized_image = (best_image - np.min(best_image)) / (np.max(best_image) - np.min(best_image)) * 255
    normalized_image = normalized_image.astype(np.uint8)
    # Convert the normalized image array to PIL Image
    normalized_image = Image.fromarray(normalized_image)

    # Save the normalized image with the unique name
    timestamp = time.strftime("%Y%m%d%H%M%S")
    normalized_image_name = f"{target_image_path}_{timestamp}.png"
    normalized_image.save(normalized_image_name)

    # Optionally, show the normalized image
    normalized_image.show()


