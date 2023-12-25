import numpy as np
import random
from matplotlib import pyplot as plt

########## Global Data ##########
training_set = []
training_labels = []
population = []

lower_bounds = [0, -2, -1] # lower bounds for (a0, a1, a2)
upper_bounds = [2, 0, 1] # upper bounds for (a0, a1, a2)

########## Hyperparameters ##########
incrementing_factor = 0.5 # determines the size of the population
mating_factor = 3
crossover_probability = 0.80
mutation_probability = 0.068
convergence_termination_rate = 0.031
mutation_range = [-0.25, 0.25]

########## Reading Training Data ##########
def fetchLabels(filename, arr):
	file = open(filename, "r")
	for line in file.readlines():
		label = line.split()
		arr += [float(label[0])]
def fetchSet(filename, arr):
	file = open(filename, "r")
	for line in file.readlines():
		x1,x2,x3 = line.split(sep=',')
		arr += [[float(x1),float(x2),float(x3)]]
def fetchFiles():
	fetchSet("training_set_v2", training_set)
	fetchLabels("training_labels_v2", training_labels)
fetchFiles()

########## Genetic Algorithm Code ##########
def generatePopulation():
	for a0 in np.arange(0, 2, incrementing_factor):
		for a1 in np.arange(-2, 0, incrementing_factor):
			for a2 in np.arange(-1, 1, incrementing_factor):
				individual = [
					random.uniform(a0, a0 + incrementing_factor), 
	                random.uniform(a1, a1 + incrementing_factor), 
	                random.uniform(a2, a2 + incrementing_factor)
	            ]
				population.append(individual)

def f(x, y, constants):
	a0, a1, a2 = constants
	return (a0*np.cbrt(x-5)) + (a1*np.cbrt(y+5)) + a2

def fitness(constants):
	excess = len(training_set) # used to prevent having -ve values for the fitness
	fitness = 0
	for idx, [x,y,z] in enumerate(training_set):
		function_z = f(x, y, constants)
		if z >= function_z and training_labels[idx] == 1:
			fitness += 1
		elif z < function_z and training_labels[idx] == -1:
			fitness += 1
		else:
			fitness -= 1

	fitness += excess
	return fitness

def getPopulationFitness():
	fitnesses = []
	for individual in population:
		fitnesses.append(fitness(individual))

	return fitnesses

def createRouletteWheel(fitnesses):
	fitnessSum = sum(fitnesses)
	# print("Fitness Sum = " + str(fitnessSum)) # uncomment to view the Fitness Sum of the Population throughout the running of the genetic algorithm

	probabilities = []
	for i in range(len(fitnesses)):
		probabilities.append(fitnesses[i] / fitnessSum)

	cumulativeProbabilities = [None] * len(fitnesses)
	cumulativeProbabilities[0] = probabilities[0]
	for i in range(1, len(fitnesses)):
		cumulativeProbabilities[i] = cumulativeProbabilities[i-1] + probabilities[i]

	return cumulativeProbabilities

def selectParent(spin, rouletteWheel):
	for i in range(len(rouletteWheel)):
		if spin < rouletteWheel[i]:
			return i

def selectParents(rouletteWheel):
	parents = []

	matingCount = int(len(population) / mating_factor)
	for i in range(matingCount):
		spin1 = random.random()
		spin2 = random.random()
		parents.append([selectParent(spin1, rouletteWheel), selectParent(spin2, rouletteWheel)]) # what if 2 parents overlap (NOT THAT IMP)

	return parents

def validateUtil(point):
	for i in range(len(point)):
		if not lower_bounds[i] <= point[i] and point[i] <= upper_bounds[i]:
			return False

	return True

def crossover(parent1_idx, parent2_idx):
	parent1 = population[parent1_idx]
	parent2 = population[parent2_idx]

	if random.uniform(0, 1) < crossover_probability:
		offspring1 = [None] * len(parent1)
		offspring2 = [None] * len(parent1)
		
		for i in range(len(parent1)):
			r = random.uniform(0, 1) # Plan around with the position of r
			offspring1[i] = (r * parent1[i]) + ((1-r) * parent2[i])
			offspring2[i] = (r * parent2[i]) + ((1-r) * parent1[i])

		if (not validateUtil(offspring1)):
			print("Exception")

		if (not validateUtil(offspring2)):
			print("Exception 2")

		population[parent1_idx] = offspring1
		population[parent2_idx] = offspring2

def mutateNewPopulation():
	for individual in population:
		for i in range(len(individual)):
			if random.uniform(0, 1) < mutation_probability:
				mutation_value = random.uniform(mutation_range[0], mutation_range[1])
				individual[i] += mutation_value

				individual[i] = max(min(upper_bounds[i], individual[i]), lower_bounds[i])

def generateOffsprings(fitnesses):
	rouletteWheel = createRouletteWheel(fitnesses)
	parents = selectParents(rouletteWheel)
	for parent1, parent2 in parents:
		crossover(parent1, parent2)

	mutateNewPopulation() # maybe try to mutate only the offsprings

def convergenceReached(fitnesses):
	if fitnesses == None:
		return False

	best = max(fitnesses)
	avg = np.average(fitnesses)
	rate = (best - avg) / avg
	return rate < convergence_termination_rate

def genetic():
	generatePopulation()
	fitnesses = None
	iterations = []
	fitness_sum = []

	limit = 0
	while limit < 150 and (not convergenceReached(fitnesses)):
		fitnesses = getPopulationFitness()
		generateOffsprings(fitnesses)

		iterations.append(limit)
		fitness_sum.append(sum(fitnesses))

		limit += 1

	bestIndividual = max(fitnesses)
	bestIdx = fitnesses.index(bestIndividual)

	plt.plot(iterations, fitness_sum)
	plt.xlabel('Iteration')
	plt.ylabel('Sum of Fitnesses')
	plt.title('Genetic Algorithm Convergence')
	plt.show()

	return population[bestIdx]

best = genetic()
for i in range(len(best)):
	print("a" + str(i) + " = " + str(best[i]))