from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
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
    # Calculate Mean Squared Error (MSE) between two images
    return np.mean((image1 - image2) ** 2)


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

    plt.ioff()  # Interaktif modu kapat
    # Plot fitness scores over generations
    if show_plot:
        plt.figure()
        for i in range(population_size):
            plt.plot(range(generations), [fitness_scores[i] for fitness_scores in fitness_scores_history])
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness Score over Generations")
        plt.legend()
        plt.show()

def update_plot(ax, best_image_with_real, generation):
    ax.clear()  # Mevcut çizimi temizle
    ax.imshow(best_image_with_real, cmap='gray')
    ax.set_title(f"Generation {generation}")
    plt.draw()
    plt.pause(0.1)  # Grafik güncellemesi için kısa bir mola ver


if __name__ == "__main__":
    target_image_path = "images/adam.png"
    matrix, target_image = create_circle_matrix(target_image_path)

    population_size = 20
    mutation_rate = 0.01
    generations = 300

    population = []
    # Initialize population
    for i in range(population_size):
        path = initpath(50)
        population.append(path)

    fig, ax = plt.subplots()  # Create a figure and axis for the plot
    plt.ion()  # Turn on interactive mode for continuous updating

    fitness_scores_history = []
    generation_times = []

    for generation in range(generations):
        start_time = time.time()
        # Evaluate fitness of each path in the population
        fitness_scores = [fitness(draw_path(matrix.copy(), path), target_image) for path in population]
        fitness_scores_history.append(fitness_scores)
        # Select paths based on their fitness
        selected_paths = [population[i] for i in np.argsort(fitness_scores)[:population_size // 2]]
        previous_best_path = min(population, key=lambda path: fitness(draw_path(matrix.copy(), path), target_image))

        # Create new paths through crossover
        new_population = []

        while len(new_population) < population_size:
            path1 = random.choice(selected_paths)
            path2 = random.choice(selected_paths)
            new_path = crossover(path1, path2)
            new_population.append(new_path)

        # Mutate new population
        population = [mutate(path, mutation_rate) for path in new_population]
        
        print(f"Generation {generation} best fitness: {min(fitness_scores)}")
        end_time = time.time()
        generation_time = end_time - start_time
        print(f"\ttime: {round(generation_time, 3)} seconds")
        generation_times.append(generation_time)

        #prevent having worse path than previous generation
        population.append(previous_best_path)
        
        if generation % 5 == 0:
            # Get the best path (image) from the final population
            best_path = min(population, key=lambda path: fitness(draw_path(matrix.copy(), path), target_image))

            best_image_with_real = draw_path(matrix.copy(), best_path)
            best_image_with_real[target_image != 255] = 200

            update_plot(ax, best_image_with_real, generation)


    # Get the best path (image) from the final population
    best_path = min(population, key=lambda path: fitness(draw_path(matrix.copy(), path), target_image))

    # Draw the best path

    output(best_path, show_plot=True, show_image=True, save_image=False)
