from PIL import Image
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
import numexpr as ne
import time

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

def get_angle_pixel_coordinates(matrix, angle):
    height, width = matrix.shape
    # Calculate the center coordinates of the circle
    center_x = width // 2
    center_y = height // 2
    radius = min(center_x, center_y)

    # Calculate the angle in radians
    angle_rad = np.deg2rad(angle)
    # Calculate the x and y coordinates of the pixel for the given angle
    x = int(center_x + radius * np.cos(angle_rad))
    y = int(center_y + radius * np.sin(angle_rad))

    return x, y

def draw_circle(matrix):
    center_x, center_y = matrix.shape[1] // 2, matrix.shape[0] // 2
    radius = min(center_x, center_y)

    for i in range(0, 360):
        angle_rad = np.deg2rad(i)
        x = int(center_x + radius * np.cos(angle_rad))
        y = int(center_y + radius * np.sin(angle_rad))
        if 0 <= y < matrix.shape[0] and 0 <= x < matrix.shape[1]:
            matrix[y, x] = 0

    return matrix

def draw_line(matrix, start_x, start_y, end_x, end_y):
    decrease = 20
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
            matrix[int(y), int(x)] -= decrease
        x += step_x
        y += step_y

    return matrix

def draw_path(matrix, path):
    for i in range(len(path) - 1):
        start_x, start_y = get_angle_pixel_coordinates(matrix, path[i])
        end_x, end_y = get_angle_pixel_coordinates(matrix, path[i + 1])
        if start_x != end_x or start_y != end_y:
            matrix = draw_line(matrix, start_x, start_y, end_x, end_y)

    return matrix


def initpath(length):
    path = random.sample(range(0, 360), length)
    return path


def fitness(image1, image2):
    # Calculate Mean Squared Error (MSE) between two images
    return np.mean((image1 - image2) ** 2)  

def fitness_hamming(image1, image2):
    # Calculate Hamming distance between two images
    return np.count_nonzero(image1 != image2)

def fitness_numexpr(image1, image2):
    return ne.evaluate("mean((image1 - image2) ** 2)")

def calculate_fitness_parallel(population, matrix, target_image):
    def worker(path):
        image = draw_path(matrix.copy(), path)
        return fitness_hamming(image, target_image)
    
    with ThreadPoolExecutor() as executor:
        fitness_scores = list(executor.map(worker, population))
    
    return fitness_scores


def crossover(path1, path2):
    # Perform crossover between two paths
    # For example, you can randomly select a point and swap the paths from that point
    crossover_point = random.randint(0, len(path1) - 1)
    new_path = path1[:crossover_point] + path2[crossover_point:]
    return new_path


def mutate(path, mutation_rate):
    # Perform mutation on a path
    # For example, you can randomly select a point and change its value
    mutated_path = path.copy()
    for _ in range(mutation_rate):
        mutated_path[random.randint(0, len(path) - 1)] = random.randint(0, 360)
    return mutated_path

if __name__ == "__main__":
    target_image_path = "images/1.png"
    matrix, target_image = create_circle_matrix(target_image_path)

    length = 350
    population_size = 5
    mutation_rate = int(length * 0.1)
    generations = 100
    
    # Initialize the population
    population = []
    for _ in range(population_size):
        path = initpath(length)
        population.append(path)

    # Evolve the population
    for i in range(generations):
        start_time = time.time()
        
        # Calculate fitness for each individual in the population
        fitness_scores = calculate_fitness_parallel(population, matrix, target_image)

        # Select parents for crossover
        parents = []
        for _ in range(2):
            # Select two individuals with the lowest fitness scores
            min_index = np.argmin(fitness_scores)
            parent = population[min_index]
            parents.append(parent)
            # Remove the selected parent from the population and fitness scores
            population.pop(min_index)
            fitness_scores.pop(min_index)

        # Perform crossover and mutation to create new offspring
        offspring = []
        for _ in range(population_size - 2):
            # Perform crossover between the selected parents
            child = crossover(parents[0], parents[1])
            # Perform mutation on the child
            child = mutate(child, mutation_rate)
            offspring.append(child)

        # Add the parents and offspring to the population
        population.extend(parents)
        population.extend(offspring)

        # Debugging prints
        print("Generation: " + str(i))
        print("Best fitness score:", min(fitness_scores))
        print("Best path:", population[np.argmin(fitness_scores)])
        print("Population size:", len(population))

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time:", elapsed_time)
        print()
        

    # Find the best path with the lowest fitness score
    best_index = np.argmin(fitness_scores)
    best_path = population[best_index]

    # Draw the best path
    best_image = draw_path(matrix.copy(), best_path)
    # Normalize the image data
    normalized_image = (best_image - np.min(best_image)) / (np.max(best_image) - np.min(best_image)) * 255
    normalized_image = normalized_image.astype(np.uint8)

    # Convert the normalized image array to PIL Image
    normalized_image = Image.fromarray(normalized_image)

    # Save the normalized image with the unique name
    timestamp = time.strftime("%Y%m%d%H%M%S")
    normalized_image_name = f"normalized_image_{timestamp}.png"
    normalized_image.save(normalized_image_name)

    # Show the normalized image
    normalized_image.show()
