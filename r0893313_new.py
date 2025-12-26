import math
import numpy as np
import Reporter
import random
import itertools

class r0123456:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def length(self, individual, distance_matrix):
        """
        The fitness measure
        :param individual: The individual of which the length should be calculated
        :param distance_matrix: The distance matrix
        :return: The length of the individual
        """
        tour = np.concatenate((individual, [individual[0]]))
        dists = distance_matrix[tour[:-1], tour[1:]]
        if np.isinf(dists).any():
            return np.inf
        return dists.sum()


    def random_initialize_population(self, pop_size, n):
        """
        Random initialization of the population
        :param pop_size: The size of the population that should be generated
        :param n: The amount of cities in each individual
        :return: A numpy array containing pop_size numpy arrays of size n representing the population
        """
        return np.array([np.random.permutation(n) for _ in range(pop_size)])

    def greedy_initialize_population(self, pop_size, n, var, distance_matrix):
        """
        Greedy initialization of the population
        :param pop_size: The size of the population that should be generated
        :param n: The amount of cities in each individual
        :param var: The randomness that should be inflicted upon the population. 1 returns identical individuals, var > pop_size returns random initialization
        :param distance_matrix: The distance matrix
        :return: A numpy array containing pop_size numpy arrays of size n representing the population
        """
        pop = np.zeros((pop_size,n), dtype=int)
        max_bruteforce = min(var, 8)
        sorted_neighbors = [np.argsort(distance_matrix[i, :n - 1]) for i in range(n)]

        for ind in range(pop_size):
            free = set(range(n))
            cur = random.choice(list(range(n)))
            # Greedy phase
            for i in range(n - max_bruteforce):
                pop[ind,i] = cur
                free.remove(cur)
                # Take the closest available neighbors
                candidates = []
                for city in sorted_neighbors[cur]:
                    if city in free:
                        candidates.append(city)
                        if len(candidates) == var:
                            break
                # Pick one of the closest
                cur = random.choice(candidates)

            # Brute-force final segment
            free_list = list(free)
            best_cost = np.inf
            best_order = free_list
            for perm in itertools.permutations(range(len(free_list))):
                route = [free_list[i] for i in perm]
                cost = distance_matrix[pop[ind,-max_bruteforce-1], route[0]] #connection start cost
                for i in range(len(route) - 1): #internal cost
                    cost += distance_matrix[route[i], route[i + 1]]
                cost += distance_matrix[route[-1], pop[ind,0]] #connection end cost

                if cost < best_cost:
                    best_cost = cost
                    best_order = route

            pop[ind,-max_bruteforce:] = best_order

        return pop

    def tournament_selection(self, population, fitnesses, k):
        """
        Tournament-k selection
        :param population: the population from which to select
        :param fitnesses: the fitnesses of the population
        :param k: k in tournament k
        :return: A selection of individuals in the same shape as population
        """
        pop_size, ind_size = population.shape
        selected = np.zeros((pop_size, ind_size), dtype=int)

        for i in range(pop_size):
            idxs = np.random.randint(pop_size, size = k)
            best_idx = idxs[np.argmin(fitnesses[idxs])]
            selected[i] = population[best_idx].copy()

        return selected

    def mutate(self, individual, rates):
        """
        Mutate the given individual using different mutation types
        :param individual: The individual to mutate
        :param rates: The rates at which each type of mutation occurs, this is of the form: [swap, insert, scramble, inversion]
        :return: The same individual with any potential mutations
        """
        n = individual.size

        #swap mutation
        if random.random() < rates[0]:
            i, j = np.sort(np.random.randint(n, size = 2))
            individual[i], individual[j] = individual[j], individual[i]

        #insert mutation
        if random.random() < rates[1]:
            i, j = np.sort(np.random.randint(n, size = 2))
            individual[i+2:j+1], individual[i+1] = individual[i+1:j], individual[j]

        #scramble mutation
        if random.random() < rates[2]:
            i, j = np.sort(np.random.randint(n, size = 2))
            segment = individual[i:j + 1]
            np.random.shuffle(segment)

        #inversion mutation
        if random.random() < rates[3]:
            i, j = np.sort(np.random.randint(n, size = 2))
            individual[i:j + 1] = individual[i:j + 1][::-1]

        return individual

    def greedy_crossover(self, parent1, parent2, distance_matrix):
        """
        A crossover algorithm that seeks to greedily merge 2 parents
        :param parent1: The first parent
        :param parent2: The second parent
        :param distance_matrix: The distance matrix
        :return: A valid individual as numpy array
        """
        n = parent1.size
        child = np.zeros(n, dtype=int)

        #boolean mask
        visited = np.zeros(n, dtype=bool)

        current = random.choice(parent1)
        child[0] = current
        visited[current] = True

        #lookup tables
        pos1 = np.empty(n, dtype=int)
        pos2 = np.empty(n, dtype=int)

        for i in range(n):
            pos1[parent1[i]] = i
            pos2[parent2[i]] = i

        for i in range(n - 1):
            distance_matrix_current_row = distance_matrix[current]
            c1 = parent1[(pos1[current] + 1) % n]
            c2 = parent2[(pos2[current] + 1) % n]

            next_city = -1

            if not visited[c1]:
                if visited[c2] or c1 == c2:
                    next_city = c1
                else:
                    if distance_matrix_current_row[c1] <= distance_matrix_current_row[c2]:
                        next_city = c1
                    else:
                        next_city = c2
            elif not visited[c2]:
                next_city = c2
            else:
                mask = np.where(visited, np.inf, distance_matrix_current_row)
                next_city = int(mask.argmin())

            child[i+1] = next_city
            visited[next_city] = True
            current = next_city

        return child

    def fast_edge_assembly_crossover(self, parent1, parent2, amount, distance_matrix):
        """
        Edge assembly crossover as described in Merlevede A. 2020, but with non-complete subtour reconection.
        :param parent1: first parent
        :param parent2: second parent
        :param amount: amount of children to return
        :param distance_matrix: the distance matrix
        :return: amount of children that closely follow the parents but make greedy connections sometimes
        """
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

                idx = next(i for i, t in enumerate(subtours) if t is U)
                del subtours[idx]

                idx = next(i for i, t in enumerate(subtours) if t is T)
                del subtours[idx]
                subtours.append(new_tour)
            children.append(subtours[0])
        return children

    def three_opt(self, individual, distance_matrix):
        """
        A version of the three opt local search algorithm for a directed graph where the result is returned after any improvement is found
        :param individual: the individual to run three_opt on
        :param distance_matrix: the distance matrix
        :return: a potentially improved version of individual
        """
        n = individual.size

        for i in range(n - 5):
            A = individual[i]
            B = individual[i + 1]
            current1 = distance_matrix[A,B]

            for j in range(i + 2, n - 3):
                C = individual[j]
                D = individual[j + 1]
                current2 = distance_matrix[C, D]
                proposed1 = distance_matrix[A, D]

                for k in range(j+2, n-1):
                    E = individual[k]
                    F = individual[k + 1]

                    current = current1 + current2 + distance_matrix[E,F]
                    proposed = proposed1 + distance_matrix[E,B] + distance_matrix[C,F]

                    if proposed < current:
                        individual = np.concatenate((individual[:i+1], individual[j+1:k+1], individual[i+1:j+1], individual[k+1:]))
                        return individual

        return individual

    def swap_opt(self, individual, distance_matrix):
        """
        Local search algoritm that searches for any two cities to swap that would be beneficial
        :param individual: The individual to optimize
        :param distance_matrix: the distance matrix
        :return: A potentially improved version of individual
        """
        n = individual.size
        r = np.arange(n)
        np.random.shuffle(r)
        for i in range(n):
            curLoc = r[i]
            propLoc = r[(i+1)%len(r)]
            A = individual[curLoc]
            B = individual[propLoc]
            curLeft = individual[curLoc - 1]
            curRight = individual[(curLoc + 1)%n]
            propLeft = individual[propLoc - 1]
            propRight = individual[(propLoc + 1)%n]
            if curRight == B:
                curDist = (distance_matrix[curLeft, A] + distance_matrix[A, B] +
                           distance_matrix[B, propRight])
                propDist = (distance_matrix[curLeft, B] + distance_matrix[B, A] +
                           distance_matrix[A, propRight])
            elif curLeft == B:
                curDist = (distance_matrix[A, curRight] +
                           distance_matrix[propLeft, B] + distance_matrix[B, A])
                propDist = (distance_matrix[B, curRight] +
                           distance_matrix[propLeft, A] + distance_matrix[A, B])
            else:
                curDist = (distance_matrix[curLeft, A] + distance_matrix[A, curRight] +
                           distance_matrix[propLeft, B] + distance_matrix[B, propRight])
                propDist = (distance_matrix[curLeft, B] + distance_matrix[B, curRight] +
                           distance_matrix[propLeft, A] + distance_matrix[A, propRight])

            if curDist>propDist:
                individual[curLoc], individual[propLoc] = B, A
                return individual

        return individual


    def optimize(self, filename):
        """
        A genetic algorithm which optimizes the traveling salesman problem.
        :param filename: the file containing the (assymetric) distance matrix describing the TSP
        :return: 0
        """
        distanceMatrix = np.loadtxt(filename, delimiter=",")
        n = distanceMatrix.shape[0]

        pop_size_random = 200
        pop_size_greedy = 20
        pop_size=pop_size_greedy+pop_size_random
        tournament_k = 3
        stagnation_limit = 50
        initialize_var = 6
        mutation_rates = [0.1, 0.1, 0.1, 0.05]

        populationRandom = self.random_initialize_population(pop_size_random, n)
        populationGreedy = self.greedy_initialize_population(pop_size_greedy, n , initialize_var, distanceMatrix)
        population = np.concatenate((populationRandom, populationGreedy))

        best_history = []
        elite = population[0]  # placeholder elite
        while True:
            fitnesses = np.empty(len(population), dtype=float)
            for i, ind in enumerate(population):
                fitnesses[i] = self.length(ind, distanceMatrix)

            worst_idx = np.argmax(fitnesses)
            population[worst_idx] = elite
            fitnesses[worst_idx] = self.length(elite, distanceMatrix)

            best_idx = np.argmin(fitnesses)
            bestObjective = fitnesses[best_idx]
            meanObjective = fitnesses.mean()

            bestSolution = population[best_idx]
            elite = population[best_idx].copy()

            best_history.append(bestObjective)
            if len(best_history) > stagnation_limit:
                window = best_history[-stagnation_limit:]
                if max(window) - min(window) < 1e-6:
                    break

            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

            selected = self.tournament_selection(population, fitnesses, tournament_k)

            offspring = np.zeros((pop_size, n), dtype=int)
            for i in range(0, len(selected) - 1, 2):
                p1, p2 = selected[i], selected[i + 1]
                child1 = self.greedy_crossover(p1,p2,distanceMatrix)
                child2 = self.greedy_crossover(p2,p1,distanceMatrix)
                child1 = self.mutate(child1, mutation_rates)
                child2 = self.mutate(child2, mutation_rates)
                child1 = self.swap_opt(child1, distanceMatrix)
                child2 = self.swap_opt(child2, distanceMatrix)
                offspring[i] = child1
                offspring[i+1] = child2

            elite = self.swap_opt(elite, distanceMatrix)
            population = offspring

        return 0
