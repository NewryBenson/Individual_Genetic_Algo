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
            pop.append(individual) #append individual
        return pop

    def tournament_selection(self, population, fitnesses, k):
        selected = []
        for _ in range(len(population)):
            candidates = random.sample(list(zip(population, fitnesses)), k)
            winner = min(candidates, key=lambda x: x[1])[0]
            selected.append(winner.copy())
        return selected

    def mutate(self, individual):
        #inversion:
        if random.random() < individual[1][0]:
            i, j = sorted(random.sample(range(len(individual[0])), 2))
            individual[0][i:j + 1] = individual[0][i:j + 1][::-1]
        #scramble:
        if random.random() < individual[1][1]:
            i, j = sorted(random.sample(range(len(individual[0])), 2))
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

    def edge_recombination(self, parent1, parent2):
        tour1, rate1 = parent1
        tour2, rate2 = parent2
        n = len(tour1)

        # Build adjacency lists (successors only)
        adj = {i: set() for i in tour1}
        for tour in (tour1, tour2):
            for i in range(n):
                a = tour[i]
                b = tour[(i + 1) % n]
                adj[a].add(b)

        # Start from random city
        current = random.choice(list(adj.keys()))
        child = [current]

        while len(child) < n:
            # Remove current from all adjacency lists
            for s in adj.values():
                s.discard(current)

            if adj[current]:
                # Pick successor with smallest adjacency list
                next_city = min(adj[current], key=lambda x: len(adj[x]))
            else:
                # Fallback: pick any unused city
                next_city = random.choice([c for c in adj if c not in child])

            child.append(next_city)
            current = next_city

        rate = (np.array(rate1) + np.array(rate2)) / 2
        return [np.array(child), rate]

    def eliminate(self, parents, offspring, distance_matrix, k):
        combined = parents + offspring
        fitnesses = [self.length(ind[0], distance_matrix) for ind in combined]
        survivors = []
        for _ in range(len(parents)):
            candidates = random.sample(list(zip(combined, fitnesses)), k)
            winner = min(candidates, key=lambda x: x[1])[0]
            survivors.append(winner.copy())
        return survivors


    def local_swap_opt(self, individual, distance_matrix):
        n = len(individual[0])
        prev = n-1
        for i in range(len(individual[0])-2):
            if distance_matrix[prev][individual[0][i]] + distance_matrix[individual[0][i+1]][individual[0][i+2]] - (distance_matrix[prev][individual[0][i+1]] + distance_matrix[individual[0][i]][individual[0][i+2]]) > 0:
                individual[0][i], individual[0][i+1] = individual[0][i+1], individual[0][i]
            prev = individual[0][i]
        return individual


    def optimize(self, filename):
        distance_matrix = np.loadtxt(open(filename), delimiter=',')
        n = distance_matrix.shape[0]



        pop_size = 300
        tournament_k = 3
        stagnation_limit = 100
        initialize_var = 5
        print("start initialization")
        population = self.greedy_initialize_population(pop_size, n, initialize_var , distance_matrix)
        print("finish initialization")
        best_history = []
        iteration = 0
        elite = [list(range(n)), [0.15, 0.15, 0.15]] #bogus elite for first iteration
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
                    print("Terminated due to stagnation")
                    break

            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                print("Terminated due to time limit")
                break

            selected = self.tournament_selection(population, fitnesses, tournament_k) #parent selection
            offspring = []
            for i in range(0, len(selected), 2):
                if i+1 < len(selected):
                    child1 = self.edge_recombination(selected[i], selected[i+1])
                    child2 = self.edge_recombination(selected[i+1], selected[i])
                    # self-adaptive mutation
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)

                    # 2-opt local search
                    child1 = self.local_swap_opt(child1, distance_matrix)
                    child2 = self.local_swap_opt(child2, distance_matrix)

                    offspring.extend([child1, child2])
            #population = self.eliminate(population, offspring, distance_matrix, tournament_k)
            population=offspring
            iteration += 1

        return 0