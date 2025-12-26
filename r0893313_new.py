import math
from itertools import combinations
import numpy as np
from numpy.f2py.auxfuncs import throw_error

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

    def random_initialize_population(self, pop_size, n):
        return np.array([np.random.permutation(n) for _ in range(pop_size)])

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

    def greedy_crossover(self, parent1, parent2, distance_matrix):
        n = len(parent1)

        current = random.choice(parent1)
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

        return np.array(child)

    def fast_edge_assembly_crossover(self, parent1, parent2, amount, distance_matrix):
        size = len(parent1)
        Ea = np.empty(size, dtype=int)
        Eb = np.empty(size, dtype=int)
        for i in range(size):
            Ea[parent1[i]] = parent1[(i + 1) % size]
            Eb[parent2[i]] = parent2[(i + 1) % size]
        invEa = np.empty(size, dtype=int)
        invEb = np.empty(size, dtype=int)
        for i in range(size):
            invEa[Ea[i]] = i
            invEb[Eb[i]] = i
        EbinvEa = Eb[invEa]

        cycles = []
        visited = np.zeros(size, dtype=bool)

        for start in range(size):
            if visited[start]:
                continue

            cur = start
            cycle = []
            while not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = EbinvEa[cur]

            cycles.append(cycle)

        children = []
        for _ in range(amount):
            size_cycles = len(cycles)
            number_of_cycles = np.random.choice(list(range(size_cycles + 1)), p=[math.comb(size_cycles, k) / (2 ** size_cycles) for k in range(size_cycles + 1)])
            subset = random.sample(cycles, number_of_cycles)
            Ex = Eb.copy()

            for cycle in subset:
                for k in range(len(cycle)):
                    city = cycle[k]
                    prev_city = cycle[k - 1]
                    pos = invEb[city]
                    Ex[pos] = prev_city
            visited = np.zeros(size, dtype=bool)
            subtours = []
            for start in range(size):
                if visited[start]:
                    continue
                cur = start
                tour = []
                while not visited[cur]:
                    visited[cur] = True
                    tour.append(cur)
                    cur = Ex[cur]
                subtours.append(np.asarray(tour, dtype=np.int32))

            while len(subtours) > 1:

                U = min(subtours, key=len)

                best_dist = np.inf
                best_connect = None

                longest_dist = 0
                longest_v1 = None
                for i in range(len(U)):
                    dist = distance_matrix[U[i], U[(i+1)%len(U)]]
                    if dist>longest_dist:
                        longest_dist = dist
                        longest_v1 = i
                v1 = U[longest_v1]
                for tour in subtours:
                    if tour is U:
                        continue

                    for j in range(len(tour)):
                        v3 = tour[j]
                        dist = distance_matrix[v1,v3]

                        if dist < best_dist:
                            best_dist = dist
                            best_connect = (tour, j)

                T, j = best_connect

                new_tour = np.concatenate((U[:longest_v1], T[j:], T[:j], U[longest_v1:]))

                # remove U
                idx = next(i for i, t in enumerate(subtours) if t is U)
                del subtours[idx]

                # remove T
                idx = next(i for i, t in enumerate(subtours) if t is T)
                del subtours[idx]
                subtours.append(new_tour)
            children.append(subtours[0])
        return children

    def edge_assembly_crossover(self, parent1, parent2, amount):
        size = len(parent1)
        Ea = np.empty(size, dtype=int)
        Eb = np.empty(size, dtype=int)
        for i in range(size):
            Ea[parent1[i]] = parent1[(i + 1) % size]
            Eb[parent2[i]] = parent2[(i + 1) % size]
        invEa = np.empty(size, dtype=int)
        invEb = np.empty(size, dtype=int)
        for i in range(size):
            invEa[Ea[i]] = i
            invEb[Eb[i]] = i
        EbinvEa = Eb[invEa]

        cycles = []
        visited = np.zeros(size, dtype=bool)

        for start in range(size):
            if visited[start]:
                continue

            cur = start
            cycle = []
            while not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = EbinvEa[cur]

            cycles.append(cycle)

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
            if len(children) == amount:
                break
        return children

    def eliminate(self, parents, offspring, distance_matrix):
        fitnesses = np.array([self.length(ind, distance_matrix) for ind in offspring])
        survivors_index = np.argpartition(fitnesses, len(parents))[:len(parents)]
        return np.array(offspring)[survivors_index]

    def three_opt(self, tour, distance_matrix):
        n = len(tour)
        improved = True
        tour = list(tour)

        while improved:
            improved = False

            for i in range(n - 5):
                A = tour[i]
                B = tour[i + 1]

                for j in range(i + 2, n - 3):
                    C = tour[j]
                    D = tour[j + 1]

                    for k in range(j+2, n-1):
                        E = tour[k]
                        F = tour[k + 1]

                        # current cost of edges
                        current = distance_matrix[A,B] + distance_matrix[C,D] + distance_matrix[E,F]

                        # proposed edges (no reversal!)
                        proposed = distance_matrix[A,D] + distance_matrix[E,B] + distance_matrix[C,F]

                        if proposed < current:
                            tour = tour[:i+1] + tour[j+1:k+1] + tour[i+1:j+1] + tour[k+1:]
                            return np.array(tour)

        return np.array(tour)

    def swap_opt(self, tour, distance_matrix):
        r = list(range(len(tour)-1))
        random.shuffle(r)
        for i in range(len(tour)-1):
            curLoc = r[i]
            propLoc = r[(i+1)%len(r)]
            A = tour[curLoc]
            B = tour[propLoc]
            curLeft = tour[curLoc - 1]
            curRight = tour[curLoc + 1]
            propLeft = tour[propLoc - 1]
            propRight = tour[propLoc + 1]
            curDist = (distance_matrix[curLeft, A] + distance_matrix[A, curRight] +
                       distance_matrix[propLeft, B] + distance_matrix[B, propRight])
            propDist = (distance_matrix[curLeft, B] + distance_matrix[B, curRight] +
                       distance_matrix[propLeft, A] + distance_matrix[A, propRight])
            if curDist>propDist:
                tour[curLoc], tour[propLoc] = B, A
                return np.array(tour)

        return np.array(tour)


    def optimize(self, filename):
        # --- Load distance matrix efficiently ---
        distanceMatrix = np.loadtxt(filename, delimiter=",")
        n = distanceMatrix.shape[0]

        # --- GA parameters ---
        pop_size_random = 200
        pop_size_greedy = 20
        pop_size=pop_size_greedy+pop_size_random
        tournament_k = 3
        stagnation_limit = 500
        initialize_var = 4
        mutation_rates = [0.1, 0.1, 0.1, 0.05]

        print("start initialization")
        populationRandom = self.random_initialize_population(pop_size_random, n)#, initialize_var, distanceMatrix)
        populationGreedy = self.greedy_initialize_population(pop_size_greedy, n , initialize_var, distanceMatrix)
        population = np.concatenate((populationRandom, populationGreedy))
        print("finish initialization")

        best_history = []
        elite = population[0]  # placeholder elite

        while True:
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
                child1 = self.greedy_crossover(p1,p2,distanceMatrix)
                child2 = self.greedy_crossover(p2,p1,distanceMatrix)
                child1 = self.mutate(child1, mutation_rates)
                child2 = self.mutate(child2, mutation_rates)
                child1 = self.swap_opt(child1, distanceMatrix)
                child2 = self.swap_opt(child2, distanceMatrix)
                offspring.append(child1)
                offspring.append(child2)
            # --- Survivor selection ---
            elite = self.three_opt(elite, distanceMatrix)
            population = offspring

        return 0
