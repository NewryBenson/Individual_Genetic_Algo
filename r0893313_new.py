import math
from itertools import combinations
import numpy as np
import Reporter
import random
import itertools
class r0123456:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def length(self, individual, distance_matrix):
        tour = np.concatenate((individual, [individual[0]]))
        dists = distance_matrix[tour[:-1], tour[1:]]
        if np.isinf(dists).any():
            return np.inf
        return dists.sum()

    def greedy_initialize_population(self, pop_size, n, var, distance_matrix):
        pop = []
        max_bruteforce = min(var, 8)

        # Precompute sorted neighbors for each city (huge speedup)
        sorted_neighbors = [
            np.argsort(distance_matrix[i, :n - 1]) for i in range(n)
        ]

        for _ in range(pop_size):
            free = set(range(n))
            prev = random.choice(list(range(n)))
            individual = [prev]
            free.remove(prev)
            # Greedy phase
            while len(free) >= max_bruteforce:
                # Take the closest available neighbors
                candidates = []
                for city in sorted_neighbors[prev]:
                    if city in free:
                        candidates.append(city)
                        if len(candidates) == var:
                            break

                # Pick one of the closest
                next_city = random.choice(candidates)
                individual.append(next_city)
                free.remove(next_city)
                prev = next_city

            # Brute-force final segment (small set)
            free_list = list(free)
            best_cost = np.inf
            best_order = free_list

            # Precompute pairwise distances for the remaining nodes
            # This avoids repeated indexing inside the loop
            subdist = distance_matrix[np.ix_(free_list, free_list)]

            for perm in itertools.permutations(range(len(free_list))):
                # Convert perm indices to actual city numbers
                route = [free_list[i] for i in perm]

                # Cost from prev → first
                cost = distance_matrix[prev, route[0]]

                # Internal perm cost
                for i in range(len(route) - 1):
                    cost += distance_matrix[route[i], route[i + 1]]

                # Cost from last → return city (n-1)
                cost += distance_matrix[route[-1], individual[0]]

                if cost < best_cost:
                    best_cost = cost
                    best_order = route

            individual.extend(best_order)
            pop.append(individual)

        return np.array(pop)

    def tournament_selection(self, population, fitnesses, k):
        pop_size = len(population)
        selected = []

        for _ in range(pop_size):
            idxs = random.sample(range(pop_size), k)
            best_idx = idxs[np.argmin(fitnesses[idxs])]
            selected.append(population[best_idx].copy())

        return selected

    def mutate(self, individual, rates):
        n = len(individual)

        #swap mutation
        if random.random() < rates[0]:
            i, j = sorted(random.sample(range(n), 2))
            individual[i], individual[j] = individual[j], individual[i]

        #insert mutation
        if random.random() < rates[1]:
            i, j = sorted(random.sample(range(n), 2))
            individual[i+2:j+1], individual[i+1] = individual[i+1:j], individual[j]

        #scramble mutation
        if random.random() < rates[2]:
            i, j = sorted(random.sample(range(n), 2))
            segment = individual[i:j + 1]
            np.random.shuffle(segment)

        #inversion mutation
        if random.random() < rates[3]:
            i, j = sorted(random.sample(range(n), 2))
            individual[i:j + 1] = individual[i:j + 1][::-1]

        return individual

    def partially_mapped_crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = parent1[start:end]
        for i in np.concatenate([np.arange(0,start), np.arange(end,size)]):
            candidate = parent2[i]
            while candidate in parent1[start:end]:
                candidate = parent2[np.where(parent1 == candidate)[0][0]]
            child[i] = candidate
        return child

    def edge_assembly_crossover(self, parent1, parent2, amount):
        size = len(parent1)
        Ea = np.empty(size, dtype=int)
        Eb = np.empty(size, dtype=int)
        for i in range(size):
            Ea[parent1[i]] = parent1[(i + 1) % size]
            Eb[parent2[i]] = parent2[(i + 1) % size]
        invEa = np.empty(size, dtype=int)
        for i in range(size):
            invEa[Ea[i]] = i
        EbinvEa = np.empty(size, dtype=int)
        for i in range(size):
            EbinvEa[i] = Eb[invEa[i]]
        cycles = []
        indices = list(range(size))
        while indices:
            prev = indices[0]
            indices.remove(prev)
            cycles.append([prev])
            next = EbinvEa[cycles[-1][-1]]
            while next != cycles[-1][0]:
                prev = next
                cycles[-1].append(prev)
                next = EbinvEa[cycles[-1][-1]]
                indices.remove(prev)
        #unions = [list(c) for r in range(len(cycles) + 1) for c in combinations(cycles, r)]
        #random.shuffle(unions)
        children = []
        while True:
            number_of_cycles = random.choice(list(range(len(cycles))))
            subset = random.sample(cycles, number_of_cycles)
            Ex = Eb.copy()
            for cycle in subset:
                for i in range(len(cycle)):
                    location = np.where(Eb == cycle[i])
                    Ex[location] = cycle[i - 1]
            child = np.zeros(size, dtype=int)
            child[0] = 0
            for i in range(size - 1):
                child[i + 1] = Ex[child[i]]
            if len(set(child)) == size:
                children.append(child)
            if len(children)==amount:
                break
        return children




    def order_crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end + 1] = parent1[start:end + 1]
        taken = set(child[start:end + 1])
        fill_idx = 0
        for gene in parent2:
            if gene not in taken:
                while child[fill_idx] is not None:
                    fill_idx += 1
                child[fill_idx] = gene
        return child

    import random
    import numpy as np

    def greedy_crossover(self, parent1, parent2, distance_matrix):
        n = len(parent1)

        # Child starts at the same start as parent1 (or random)
        current = parent1[0]
        child = [current]

        # Track visited cities
        visited = set([current])

        # Precompute successor maps for speed
        # If city is last in parent, successor is None
        succ1 = {parent1[i]: parent1[i + 1] if i + 1 < n else None for i in range(n)}
        succ2 = {parent2[i]: parent2[i + 1] if i + 1 < n else None for i in range(n)}

        for _ in range(n - 1):
            # Candidate successors from both parents
            c1 = succ1[current]
            c2 = succ2[current]

            candidates = []

            if c1 is not None and c1 not in visited:
                candidates.append(c1)
            if c2 is not None and c2 not in visited and c2 != c1:
                candidates.append(c2)

            if candidates:
                # Choose the candidate with the smallest outgoing distance
                next_city = min(candidates, key=lambda c: distance_matrix[current][c])
            else:
                # No parent-based candidate → choose the nearest unvisited city
                remaining = [c for c in parent1 if c not in visited]
                next_city = min(remaining, key=lambda c: distance_matrix[current][c])

            child.append(next_city)
            visited.add(next_city)
            current = next_city

        return child

    def eliminate(self, parents, offspring, distance_matrix):
        fitnesses = np.array([self.length(ind, distance_matrix) for ind in offspring])
        survivors_index = np.argpartition(fitnesses, len(parents))[:len(parents)]
        return np.array(offspring)[survivors_index]

    def local_search_shift(self, individual, distance_matrix):
        n = len(individual)

        # Try relocating one random city to a better position
        i = random.randrange(n)
        city = individual[i]

        best_delta = 0
        best_j = None

        # Precompute neighbors
        old_prev = individual[i - 1] if i > 0 else None
        old_next = individual[i + 1] if i < n - 1 else None

        for j in range(n):
            if j == i:
                continue

            new_prev = individual[j - 1] if j > 0 else None
            new_next = individual[j] if j < n else None

            removed = 0
            if old_prev is not None:
                removed += distance_matrix[old_prev][city]
            if old_next is not None:
                removed += distance_matrix[city][old_next]
            if old_prev is not None and old_next is not None:
                removed -= distance_matrix[old_prev][old_next]

            added = 0
            if new_prev is not None:
                added += distance_matrix[new_prev][city]
            if new_next is not None:
                added += distance_matrix[city][new_next]
            if new_prev is not None and new_next is not None:
                added -= distance_matrix[new_prev][new_next]

            delta = added - removed
            if delta < best_delta:
                best_delta = delta
                best_j = j

        if best_j is not None:
            individual.pop(i)
            individual.insert(best_j if best_j < i else best_j - 1, city)

        return individual

    def hamming_distance(self, a, b):
        return sum(x != y for x, y in zip(a, b))

    def edge_distance(self, a, b):
        n = len(a)

        # Build directed edge sets
        edges_a = {(a[i], a[i + 1]) for i in range(n - 1)}
        edges_b = {(b[i], b[i + 1]) for i in range(n - 1)}

        # Symmetric difference = edges not shared
        return len(edges_a.symmetric_difference(edges_b))

    def crowd_elimination(self, population, offspring, distance_matrix, crowd_size=4):
        # Combine and shuffle
        remaining = population + offspring
        random.shuffle(remaining)

        new_population = []

        while remaining:
            # Pick a reference individual
            ref = remaining[0]

            # Compute distances to all others
            distances = [self.edge_distance(ref, ind) for ind in remaining]

            # Get indices of the closest crowd_size individuals
            idxs = sorted(range(len(remaining)), key=lambda i: distances[i])[:crowd_size]

            # Select the best among them
            best_idx = min(idxs, key=lambda i: self.length(remaining[i], distance_matrix))
            new_population.append(remaining[best_idx])

            # Remove the entire crowd from remaining
            for i in sorted(idxs, reverse=True):
                remaining.pop(i)

        return new_population

    def optimize(self, filename):
        # --- Load distance matrix efficiently ---
        distanceMatrix = np.loadtxt(filename, delimiter=",")
        n = distanceMatrix.shape[0]

        # --- GA parameters ---
        pop_size = 200
        tournament_k = 3
        stagnation_limit = 50
        initialize_var = 6
        mutation_rates = [0.3, 0.2, 0.1, 0.01]

        print("start initialization")
        population = self.greedy_initialize_population(pop_size, n, initialize_var, distanceMatrix)
        print("finish initialization")

        best_history = []
        elite = population[0]  # placeholder elite

        while True:
            # Add elite (no need to copy here)
            population[pop_size-1] = elite

            # --- Compute fitnesses (vectorized loop) ---
            fitnesses = np.empty(len(population), dtype=float)
            for i, ind in enumerate(population):
                fitnesses[i] = self.length(ind, distanceMatrix)


            #insert elite
            worst_idx = np.argmax(fitnesses)
            population[worst_idx] = elite
            fitnesses[worst_idx] = self.length(elite, distanceMatrix)

            best_idx = np.argmin(fitnesses)
            bestObjective = fitnesses[best_idx]
            meanObjective = fitnesses.mean()

            bestSolution = population[best_idx]
            elite = population[best_idx].copy()

            # --- Stagnation check ---
            best_history.append(bestObjective)
            if len(best_history) > stagnation_limit:
                window = best_history[-stagnation_limit:]
                if max(window) - min(window) < 1e-6:
                    print("Terminated due to stagnation")
                    print(meanObjective, bestObjective)
                    break

            # --- Reporter callback ---
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                print("Terminated due to time limit")
                print(meanObjective, bestObjective)
                break

            # --- Parent selection ---
            selected = self.tournament_selection(population, fitnesses, tournament_k)

            # --- Crossover + mutation ---
            offspring = []
            for i in range(0, len(selected) - 1, 2):
                p1, p2 = selected[i], selected[i + 1]
                child1, child2 = self.edge_assembly_crossover(p1, p2, 2)
                self.mutate(child1, mutation_rates)
                self.mutate(child2, mutation_rates)
                offspring.append(child1)
                offspring.append(child2)

            # --- Survivor selection ---
            population = np.array(offspring)

        return 0
