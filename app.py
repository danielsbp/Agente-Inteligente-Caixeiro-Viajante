from rich import print
import random
from deap import base, creator, tools, algorithms



num_cities = 10  
num_individuals = 100  
num_generations = 100 

cities_distance_matrix = []

for x in range(num_cities):
    distances = [random.randint(1, 1000) for x in range(num_cities)]
    cities_distance_matrix.append(distances)
    

def evalTSP(individual, distance_matrix):
    distance = 0
    for i in range(len(individual) - 1):
        distance += distance_matrix[individual[i]][individual[i + 1]]
    distance += distance_matrix[individual[-1]][individual[0]]  # Volta à cidade de origem
    return distance,

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(num_cities), num_cities)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evalTSP, distance_matrix=cities_distance_matrix)

population = toolbox.population(n=num_individuals)

fitnesses = list(map(toolbox.evaluate, population))
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit

qtd_final_ind = 0 
for gen in range(num_generations):
    print(f"========= Geração {gen + 1} =========")
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)
    fitnesses = list(map(toolbox.evaluate, offspring))
    for ind, fit in zip(offspring, fitnesses):
        print(f"Indivíduo: {ind}, Fitness: {fit}")
        ind.fitness.values = fit
        qtd_final_ind = qtd_final_ind + 1
    population = toolbox.select(offspring + population, k=num_individuals)


best_ind = tools.selBest(population, k=1)[0]
best_route = best_ind
print("Melhor rota encontrada:", best_route)
print("Distância da melhor rota:", best_ind.fitness.values[0])
print(qtd_final_ind)