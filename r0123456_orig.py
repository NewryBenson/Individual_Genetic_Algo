import Reporter
import numpy as np
import random

class r0123456:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def length(self, individual, distanceMatrix):
        n = distanceMatrix.shape[0]
        tour = [n-1] + list(individual) + [n-1]
        total = 0
        for i in range(len(tour)-1):
            dist = distanceMatrix[tour[i], tour[i+1]]
            if np.isinf(dist):
                return float('inf')
            total += dist
        return total
    def initialize_population(self, mu, n):
        return [random.sample(range(n-1), n-1) for _ in range(mu)]

    def tournament_selection(self, population, fitnesses, k):
        selected = []
        for _ in range(len(population)):
            candidates = random.sample(list(zip(population, fitnesses)), k)
            winner = min(candidates, key=lambda x: x[1])
            selected.append(winner[0])
        return selected

    def swap_mutation(self, individual, mutation_rate=0.20):
        if random.random() < mutation_rate:
            a, b = random.sample(range(len(individual)), 2)
            individual[a], individual[b] = individual[b], individual[a]
        return individual

    def ox_crossover(self, parent1, parent2):
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None]*size
        child[a:b+1] = parent1[a:b+1]
        fill = [x for x in parent2 if x not in child]
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = fill[idx]
                idx += 1
        return child

    def optimize(self, filename):
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        n = distanceMatrix.shape[0]
        mu = 50
        lambd = 100
        k = 5
        stagnation_limit = 100
        history = []

        population = self.initialize_population(mu, n)
        fitnesses = [self.length(ind, distanceMatrix) for ind in population]

        while True:
            selected = self.tournament_selection(population, fitnesses, k)
            offspring = []
            for _ in range(lambd):
                p1, p2 = random.sample(selected, 2)
                child = self.ox_crossover(p1, p2)
                child = self.swap_mutation(child)
                offspring.append(child)

            offspring_fitnesses = [self.length(ind, distanceMatrix) for ind in offspring]
            combined = population + offspring
            combined_fitnesses = fitnesses + offspring_fitnesses
            sorted_combined = sorted(zip(combined, combined_fitnesses), key=lambda x: x[1])
            population, fitnesses = zip(*sorted_combined[:mu])
            population = list(population)
            fitnesses = list(fitnesses)

            meanObjective = np.mean(fitnesses)
            bestObjective = fitnesses[0]
            bestSolution = np.array([*population[0], n-1])

            history.append(bestObjective)
            if len(history) > stagnation_limit:
                history.pop(0)
                if all(h == history[0] for h in history):
                    break

            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

        return 0
