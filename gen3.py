import time
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

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

def initpath(path_number):
    # Initialize a path with random angles
    path = []
    angle = random.randint(0, 359)
    path.append(angle)
    for i in range(1, path_number):
        previous_angle = path[i - 1]
        angle = random.randint(0, 359)
        while angle == previous_angle:
            angle = random.randint(0, 359)
        path.append(angle)
    return path

def fitness(image1, image2):
    # Calculate the number of pixels where image1 is not white and image2 is black
    matching_pixels = np.sum((image1 != 255) & (image2 == 0))
    return matching_pixels

def crossover(path1, path2):
    new_path = []
    for i in range(len(path1)):
        if random.random() < 0.5:
            new_path.append(path1[i])
        else:
            new_path.append(path2[i])
    return new_path

def mutate(path, mutation_rate):
    # Perform mutation on a path
    # For example, you can randomly select a point and change its value
    mutated_path = path.copy()
    for i in range(len(mutated_path)):
        if random.random() < mutation_rate:
            mutated_path[i] = random.randint(0, 359)
    return mutated_path

def output(best_path, save_image=True, show_plot=True, show_image=True):
    # Draw the best path
    best_image = draw_path(matrix.copy(), best_path)

    normalized_image = (best_image - np.min(best_image)) / (np.max(best_image) - np.min(best_image)) * 255
    normalized_image = normalized_image.astype(np.uint8)
    # Convert the normalized image array to PIL Image
    normalized_image = Image.fromarray(normalized_image)

    if save_image:
        # Save the normalized image with the unique name
        timestamp = time.strftime("%Y%m%d%H%M%S")
        normalized_image_name = f"{target_image_path}_{timestamp}.png"
        normalized_image.save(normalized_image_name)

    # Show the normalized image
    if show_image:
        normalized_image.show()

    # Plot fitness scores over generations
    if show_plot:
        plt.figure()
        for i in range(population_size):
            plt.plot(range(generations), [fitness_scores[i] for fitness_scores in fitness_scores_history], label=f"Path {i + 1}")
        plt.xlabel("Generation")
        plt.ylabel("Fitness (Hamming Distance)")
        plt.title("Fitness Score over Generations")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    target_image_path = "images/6.png"
    matrix, target_image = create_circle_matrix(target_image_path)

    gene_len = 50
    population_size = 20
    mutation_rate = 0.02
    generations = 500

    population = []
    generation_times = []
    fitness_scores_history = []  # Store fitness scores for each generation

    print("Creating population...")
    print("Population size: ", population_size)
    print("Population length: ", gene_len)
    print("Mutation rate: ", mutation_rate)
    print("Number of generations: ", generations)
    print("--------------------\n")

    # Initialize population
    for i in range(population_size):
        path = initpath(gene_len)
        population.append(path)

    fig, ax = plt.subplots()  # Create a figure and axis for the plot
    plt.ion()  # Turn on interactive mode for continuous updating

    for generation in range(generations):
        start_time = time.time()

        # Evaluate fitness of each path in the population
        fitness_scores = [fitness(draw_path(matrix.copy(), path), target_image) for path in population]
        fitness_scores_history.append(fitness_scores)  # Store fitness scores for current generation

        # Select paths based on their fitness
        sorted_indices = np.argsort(fitness_scores)
        selected_paths = [population[i] for i in sorted_indices[:population_size // 2]]

        # Create new paths through crossover
        new_population = []
        while len(new_population) < population_size:
            path1 = random.choice(selected_paths)
            path2 = random.choice(selected_paths)
            new_path = crossover(path1, path2)
            new_population.append(new_path)

        # Mutate new population
        population = [mutate(path, mutation_rate) for path in new_population]

        if generation % 5 == 0:
            print(f"Generation {generation} best fitness: {min(fitness_scores)}")

            end_time = time.time()
            generation_time = end_time - start_time
            print(f"\ttime: {round(generation_time, 3)} seconds")
            generation_times.append(generation_time)

            # Get the best path (image) from the final population
            best_path = min(population, key=lambda path: fitness(draw_path(matrix.copy(), path), target_image))

            best_image_with_real = draw_path(matrix.copy(), best_path)
            best_image_with_real[target_image != 255] = 128

            # Update the plot with the best path matrix and generation number
            ax.imshow(best_image_with_real, cmap='gray')
            ax.set_title(f"Generation {generation}")
            plt.pause(0.001)  # Pause to update the plot

    average_time = sum(generation_times) / len(generation_times)
    print(f"\nAverage time per generation: {average_time} seconds")

    # Get the best path (image) from the final population
    best_path = min(population, key=lambda path: fitness(draw_path(matrix.copy(), path), target_image))

    output(best_path, save_image=False, show_plot=True, show_image=True)