import cv2
from PIL import Image
import numpy as np
import random

def create_circle_matrix(image_path):
    image = Image.open(image_path)
    width, height = image.size
    grayscale_image = image.convert('L')
    image_array = np.array(grayscale_image)
    matrix = np.ones((height, width)) * 255
    return matrix, image_array

def get_angle_pixel_coordinates(matrix, angle):
    height, width = matrix.shape
    center_x = width // 2
    center_y = height // 2
    radius = min(center_x, center_y)
    angle_rad = np.deg2rad(angle)
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
    decrease = (255 // np.sqrt(len(matrix))) * 3
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    steps = max(abs(delta_x), abs(delta_y))
    step_x = delta_x / steps
    step_y = delta_y / steps
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
    path = []
    angle = random.randint(0, 360)
    path.append(angle)
    for i in range(1, path_number):
        previous_angle = path[i - 1]
        angle = random.randint(0, 360)
        while angle == previous_angle:
            angle = random.randint(0, 360)
        path.append(angle)
    return path



def calc_fit(image, normalized_image):
    hamming_distance = np.sum(image != normalized_image)
    return hamming_distance

image_path = 'images/1.png'
matrix, image = create_circle_matrix(image_path)

path_count = 500
population_size = 50
mutation_rate = 0.1
generations = 20

population = []
for i in range(population_size):
    path = initpath(path_count)
    population.append(path)

for generation in range(generations):
    ranked_population = []
    for path in population:
        matrix = create_circle_matrix(image_path)[0]
        draw_path(matrix, path)
        fitness = calc_fit(image, Image.fromarray((matrix / np.max(matrix) * 255).astype(np.uint8)))
        ranked_population.append((fitness, path))
    ranked_population.sort()
    print(f"=== Generation {generation} best solutions ===")
    for i in range(5):
        print(f"Fitness: {ranked_population[i][0]}")
        print(f"Path: {ranked_population[i][1]}")
        print(" ")
    mating_pool = []
    for i in range(population_size):
        n = int(ranked_population[i][0] * 100) + 1
        for j in range(n):
            mating_pool.append(ranked_population[i][1])
    new_population = []
    for i in range(population_size):
        parent1 = random.choice(mating_pool)
        parent2 = random.choice(mating_pool)
        child = []
        for j in range(path_count):
            if random.random() < mutation_rate:
                child.append(random.randint(0, 360))
            else:
                if random.random() < 0.5:
                    child.append(parent1[j])
                else:
                    child.append(parent2[j])
        new_population.append(child)
    population = new_population

print("=== Best solution ===")
matrix = create_circle_matrix(image_path)[0]
draw_path(matrix, ranked_population[0][1])
print(f"Fitness: {ranked_population[0][0]}")
print(f"Path: {ranked_population[0][1]}")
circle_image = Image.fromarray((matrix / np.max(matrix) * 255).astype(np.uint8))
circle_image.show()
