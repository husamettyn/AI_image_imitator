import numpy as np
import random
from PIL import Image

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
    matrix = np.ones((height, width))

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
            matrix[y, x] = 0
    
    return matrix

def draw_line(matrix, start_x, start_y, end_x, end_y):
    # Calculate the difference between the start and end coordinates
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    
    # Calculate the number of steps needed to draw the line
    steps = max(abs(delta_x), abs(delta_y))
    
    # If steps is 0, return the matrix as is (no line to draw)
    if steps == 0:
        return matrix
    
    # Calculate the step size for each coordinate
    step_x = delta_x / steps
    step_y = delta_y / steps
    
    # Draw the line by updating the matrix values
    x = start_x
    y = start_y
    for _ in range(int(steps) + 1):  # Ensure we step through the entire range, including the end point
        if 0 <= y < matrix.shape[0] and 0 <= x < matrix.shape[1]:
            matrix[int(y), int(x)] = 0
        x += step_x
        y += step_y
    
    return matrix


def draw_path(matrix, path):
    for i in range(len(path) - 1):
        start_x, start_y = get_angle_pixel_coordinates(matrix, path[i])
        end_x, end_y = get_angle_pixel_coordinates(matrix, path[i + 1])
        matrix = draw_line(matrix, start_x, start_y, end_x, end_y)
    
    return matrix

def fitness(matrix, target_image):
    # Calculate the difference between two matrices
    difference = np.sum(np.abs(matrix - target_image))
    return 1 / (1 + difference)

def initialize_population(pop_size, path_length):
    population = []
    for _ in range(pop_size):
        individual = np.random.choice(range(360), size=path_length, replace=True).tolist()
        population.append(individual)
    return population

def select_parent(population, fitnesses):
    # Tournament selection
    tournament_size = 5
    tournament = np.random.choice(range(len(population)), tournament_size, replace=False)
    tournament_fitnesses = [fitnesses[i] for i in tournament]
    winner_index = tournament[np.argmax(tournament_fitnesses)]
    return population[winner_index]

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = np.random.randint(0, 360)
    return individual

def genetic_algorithm(target_image, pop_size, path_length, generations, mutation_rate):
    population = initialize_population(pop_size, path_length)
    best_fit = 0
    best_individual = None

    for generation in range(generations):
        # Evaluate fitness
        fitnesses = []
        for individual in population:
            matrix, _ = create_circle_matrix('images/1.png')
            matrix = draw_path(matrix, individual)
            fit = fitness(matrix, target_image)
            fitnesses.append(fit)
            if fit > best_fit:
                best_fit = fit
                best_individual = individual
        
        # Selection and reproduction
        new_population = []
        for _ in range(pop_size // 2):
            parent1 = select_parent(population, fitnesses)
            parent2 = select_parent(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])
        
        population = new_population
        
        print(f"Generation {generation + 1}: Best Fitness = {best_fit}")
    
    return best_individual

# Parameters
pop_size = 50
path_length = 50
generations = 100
mutation_rate = 0.01

# Assuming 'normalized_image' is the target image
image_path = 'images/1.png'
target_matrix, target_image = create_circle_matrix(image_path)

best_path = genetic_algorithm(target_image, pop_size, path_length, generations, mutation_rate)
print("Best Path:", best_path)

draw_path(target_matrix, best_path)
    
draw_circle(target_matrix)
# Convert the matrix back to an image
circle_image = Image.fromarray((target_matrix * 255).astype(np.uint8))

# Display the image
circle_image.show()
