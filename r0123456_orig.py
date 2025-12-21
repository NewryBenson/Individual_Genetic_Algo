import numpy as np
import Reporter
import random
import itertools
class r0123456:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def length(self, individual, distance_matrix):
        n = distance_matrix.shape[0]
        tour = [n-1] + list(individual) + [n-1]
        total = 0
        for i in range(len(tour)-1):
            dist = distance_matrix[tour[i], tour[i+1]]
            if np.isinf(dist):
                return np.inf
            total += dist
        return total

    #randomly initialize population
    # for n cities, the individuals consist of n-1 entries. The first and last city are implied to be city n
    def random_initialize_population(self, pop_size, n):
        return [np.random.permutation(n-1) for _ in range(pop_size)]

    def greedy_initialize_population(self, pop_size, n, var, distance_matrix):
        pop = []

        for _ in range(pop_size):
            free = list(range(n - 1))
            prev = n-1 #last city is always first and last in individual
            individual = []
            while len(free)>=min(var,8): #avoid too large brute force
                closest_options = sorted(free, key=lambda city: distance_matrix[prev][city])[:min(var, len(free))] #closest var (or less if not enough left) cities to the prev city
                prev = random.choice(closest_options) #random from closest
                individual.append(prev) #append to individual
                free.remove(prev) #remove from unvisited
            #brute force the ending for better last steps
            best_cost = np.inf #inf initial best perm cost
            best_order = free #ordered placeholder best perm
            for perm in itertools.permutations(free): #all perms
                cost = distance_matrix[prev][perm[0]] + distance_matrix[perm[-1]][n-1] #cost to connect this perm to the rest of the individual
                for step in range(len(perm)-1):
                    cost+=distance_matrix[perm[step]][perm[step+1]] # costs of the steps of the perm
                if cost < best_cost: #filter best perm
                    best_cost = cost
                    best_order = perm
            individual.extend(best_order) #extend individual
            pop.append([individual, random.uniform(0.1, 0.2)]) #append individual
        return pop

    def tournament_selection(self, population, fitnesses, k):
        selected = []
        for _ in range(len(population)):
            candidates = random.sample(list(zip(population, fitnesses)), k)
            winner = min(candidates, key=lambda x: x[1])[0]
            selected.append(winner.copy())
        return selected

    def mutate(self, individual, mutation_rate):
        if random.random() < mutation_rate:
            method = random.choice(['inversion', 'scramble'])
            i, j = sorted(random.sample(range(len(individual[0])), 2))
            if method == 'inversion':
                individual[0][i:j+1] = individual[0][i:j+1][::-1]
            elif method == 'scramble':
                segment = individual[0][i:j+1]
                np.random.shuffle(segment)
                individual[0][i:j+1] = segment
        return individual

    def order_crossover(self, parent1, parent2):
        size = len(parent1[0])
        start, end = sorted(random.sample(range(size), 2))
        child = [[None]*size,parent1[1]]
        child[0][start:end+1] = parent1[0][start:end+1]
        fill = [gene for gene in parent2[0] if gene not in child[0]]
        idx = 0
        for i in range(size):
            if child[0][i] is None:
                child[0][i] = fill[idx]
                idx += 1
        return child

    def eliminate(self, parents, offspring, distance_matrix, k):
        combined = parents + offspring
        fitnesses = [self.length(ind, distance_matrix) for ind in combined]
        survivors = []
        for _ in range(len(parents)):
            candidates = random.sample(list(zip(combined, fitnesses)), k)
            winner = min(candidates, key=lambda x: x[1])[0]
            survivors.append(winner.copy())
        return survivors

    def two_opt(self, individual, distance_matrix):
        """
        Perform 2-opt local search on a tour.
        individual: a tuple (tour, mutation_rate) or just tour
        Returns improved tour (tuple with mutation_rate preserved)
        """
        if isinstance(individual, tuple):
            tour, rate = individual
        else:
            tour = individual
            rate = None

        n = len(tour)
        improved = True
        while improved:
            improved = False
            for i in range(n - 2):
                for j in range(i + 2, n):
                    # calculate change if we swap edges (i,i+1) and (j,j+1)
                    a, b = tour[i], tour[i + 1]
                    c, d = tour[j], tour[(j + 1) % n]  # wrap around
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[
                        c, d]
                    if delta < 0:  # swap improves tour
                        tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                        improved = True
            if not improved:
                break

        if rate is not None:
            return (tour, rate)
        else:
            return tour

    def optimize(self, filename):
        distance_matrix = np.loadtxt(open(filename), delimiter=',')
        n = distance_matrix.shape[0]

        pop_size = 300
        tournament_k = 5
        stagnation_limit = 100
        max_iterations = 1000
        initialize_var = 5
        print("start initialization")
        population = self.greedy_initialize_population(pop_size, n, initialize_var , distance_matrix)
        print("finish initialization")
        best_history = []
        iteration = 0
        elite = [list(range(n)), 0.2] #bogus elite for first iteration
        while True:
            population.append(elite) #add elite to new population, increases pop size with 1

            fitnesses = [self.length(ind[0], distance_matrix) for ind in population]
            meanObjective = np.mean(fitnesses)
            best_idx = np.argmin(fitnesses)
            worst_idx = np.argmax(fitnesses)
            bestObjective = fitnesses[best_idx]
            bestSolution = np.concatenate(([n-1], population[best_idx][0], [n-1]))
            elite = population[best_idx].copy() #keep best individual as elite

            #reduce the population size to pop_size again
            population.pop(worst_idx)
            fitnesses.pop(worst_idx)

            best_history.append(bestObjective)
            if len(best_history) > stagnation_limit:
                if all(best_history[-stagnation_limit]-x < 1e-6 for x in best_history[-stagnation_limit:]): #no significant better best solutions
                    break
            if iteration >= max_iterations:
                break

            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

            selected = self.tournament_selection(population, fitnesses, tournament_k) #parent selection
            offspring = []
            for i in range(0, len(selected), 2):
                if i+1 < len(selected):
                    child1 = self.order_crossover(selected[i], selected[i+1])
                    child2 = self.order_crossover(selected[i+1], selected[i])
                    # self-adaptive mutation
                    child1 = self.mutate(child1, child1[1])
                    child2 = self.mutate(child2, child2[1])

                    # 2-opt local search
                    child1 = self.two_opt(child1, distance_matrix)
                    child2 = self.two_opt(child2, distance_matrix)

                    offspring.extend([child1, child2])
            population = self.eliminate(population, offspring, distance_matrix, tournament_k)

            iteration += 1

        return 0