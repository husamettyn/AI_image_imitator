import cv2
from PIL import Image
import numpy as np
import random
import sewar


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
    # DONE başta siyah üzerine beyaz çiziyormuşum onları düzenledim
    # artık her turda şu aşağıdaki sayı kadar eksiltme yapıyor
    # spesifik bir anlamı yok, karekök yapmayınca çok küçülüyor sayı o yüzden kök var.

    global path_count
    decrease = (255 // np.sqrt(path_count)) * 3
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


# path_number kadar cizgi ile sekil cizilecek
def initpath(path_number):
    # DONE: Create a list to store the angles of the path
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
    # Calculate the absolute difference between the matrix and normalized image
    diff = np.abs(image - normalized_image)

    # Calculate the fitness as the mean of the absolute differences
    fitness = np.mean(diff)

    return fitness



# DONE yeni bir görsel ekledim, 1080*1080 çözünürlüğünde
# elde edilen çizgiler güzel görünüyor. Bu path'i genetik algoritmaya uydurmamız lazım.
# şu an sadece rastgele seçilmiş bir path'i çiziyoruz.

image_path = 'images/1.png'
matrix, image = create_circle_matrix(image_path)


path_count = 500
population_size = 100
mutation_rate = 0.05
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
    ranked_population.reverse()
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
