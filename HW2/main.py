import random
import numpy as np
import cv2



# Constants
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
NUM_INDIVIDUALS = 10
NUM_GENES = 50
NUM_GENERATIONS = 1
MUTATION_PROB = 0.1
NUM_ELITES = 1

source_image = cv2.imread('painting.png')

# Class definitions
class Circle:
    def __init__(self):
        self.x = random.randint(0, IMAGE_WIDTH)
        self.y = random.randint(0, IMAGE_HEIGHT)
        self.radius = random.randint(1, max(IMAGE_WIDTH, IMAGE_HEIGHT) // 10)
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.alpha = random.uniform(0, 1)

class Individual:
    def __init__(self):
        self.chromosome = [Circle() for _ in range(NUM_GENES)]
        self.fitness = float('inf')


def draw_individual(individual):
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    for circle in individual.chromosome:
        overlay = image.copy()
        cv2.circle(overlay, (circle.x, circle.y), circle.radius, circle.color, -1)
        alpha = circle.alpha
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image

def evaluate_individual(individual, source_image):
    temp_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    for circle in individual.chromosome:
        overlay = temp_image.copy()
        cv2.circle(overlay, (circle.x, circle.y), circle.radius, circle.color, -1)
        alpha = circle.alpha
        cv2.addWeighted(overlay, alpha, temp_image, 1 - alpha, 0, temp_image)
    difference = cv2.absdiff(source_image, temp_image)
    individual.fitness = np.sum(difference)  # Sum of absolute differences



# Helper functions for selection, crossover, mutation
def selection(population):
    selected = []
    tournament_size = 5
    while len(selected) < NUM_INDIVIDUALS - NUM_ELITES:
        participants = random.sample(population, tournament_size)
        winner = min(participants, key=lambda ind: ind.fitness)
        selected.append(winner)
    return selected


def crossover(parent1, parent2):
    crossover_point = random.randint(0, NUM_GENES)
    child1_genes = parent1.chromosome[:crossover_point] + parent2.chromosome[crossover_point:]
    child2_genes = parent2.chromosome[:crossover_point] + parent1.chromosome[crossover_point:]
    child1 = Individual()
    child2 = Individual()
    child1.chromosome = child1_genes
    child2.chromosome = child2_genes
    return child1, child2


def mutate(individual):
    for circle in individual.chromosome:
        if random.random() < MUTATION_PROB:
            if random.random() < 0.5:
                circle.x = random.randint(0, IMAGE_WIDTH)
                circle.y = random.randint(0, IMAGE_HEIGHT)
            if random.random() < 0.5:
                circle.radius = random.randint(1, max(IMAGE_WIDTH, IMAGE_HEIGHT) // 10)
            if random.random() < 0.5:
                circle.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if random.random() < 0.5:
                circle.alpha = random.uniform(0, 1)


# Initialize population
population = [Individual() for _ in range(NUM_INDIVIDUALS)]

# Main evolutionary loop
for generation in range(NUM_GENERATIONS):
    # Evaluate individuals
    for individual in population:
        individual.evaluate(source_image)
    
    # Sort by fitness and perform selection
    population.sort(key=lambda ind: ind.fitness)
    next_generation = population[:NUM_ELITES]  # Elites go to next generation
    next_generation += selection(population)   # Other individuals are selected
    
    # Perform crossover and mutation
    for i in range(len(next_generation) // 2):
        parent1, parent2 = next_generation[2 * i], next_generation[2 * i + 1]
        child1, child2 = crossover(parent1, parent2)
        mutate(child1)
        mutate(child2)
        next_generation += [child1, child2]
    
    # Update population
    population = next_generation[:NUM_INDIVIDUALS]  # Keep population size constant

# The best individual at the end
best_individual = min(population, key=lambda ind: ind.fitness)
best_image = best_individual.draw()

best_image = draw_individual(best_individual)
cv2.imshow('Best Image', best_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Optionally save to file
cv2.imwrite('best_image.png', best_image)


# Save or show the best image here using OpenCV
