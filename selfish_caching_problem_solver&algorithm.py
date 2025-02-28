import random
import math
import networkx as nx
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, PULP_CBC_CMD, LpStatus, value
import numpy as np
from itertools import product
import time
import csv

def generate_distance_matrix(n, max_distance):
    """
    Generuje symetryczną macierz odległości z losowymi wartościami w zakresie [1, max_distance].

    Parameters:
    - n: Liczba węzłów
    - max_distance: Maksymalna odległość

    Returns:
    - d: Macierz odległości n x n
    """
    d = [[0 if i == j else random.randint(1, max_distance) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d[j][i] = d[i][j]
    return d

def generate_network_topology(n, topology="complete", d=None, edge_prob=0.5, grid_size=None, star_count=1, bus_length=0):
    """
    Generuje topologię sieci na podstawie podanej macierzy odległości.

    Parameters:
    - n: Liczba węzłów
    - topology: Typ topologii ('complete', 'grid', 'star', 'large_star', 'extended_star', 'bus', 'random')
    - d: Macierz odległości n x n
    - edge_prob: Prawdopodobieństwo dodania krawędzi (używane w 'random')
    - grid_size: Tuple określający wymiary siatki (używane w 'grid')
    - star_count: Liczba centrów gwiazd (używane w 'extended_star')
    - bus_length: Długość magistrali (używane w 'bus')

    Returns:
    - G: Wygenerowany graf NetworkX
    """
    if d is None:
        raise ValueError("Macierz odległości d musi być dostarczona.")

    G = nx.Graph()
    G.add_nodes_from(range(n))

    if topology == 'complete':
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(i, j, weight=d[i][j])

    elif topology == 'grid':
        if grid_size is None:
            grid_size = (int(math.sqrt(n)), int(math.sqrt(n)))
            while grid_size[0] * grid_size[1] < n:
                grid_size = (grid_size[0], grid_size[1] + 1)
        # Tworzenie siatki 2D
        grid_graph = nx.grid_2d_graph(grid_size[0], grid_size[1])
        mapping = {node: idx for idx, node in enumerate(grid_graph.nodes()) if idx < n}
        grid_graph = nx.relabel_nodes(grid_graph, mapping, copy=True)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for (u, v) in grid_graph.edges():
            G.add_edge(u, v, weight=d[u][v])

    elif topology == 'star':
        if n < 2:
            raise ValueError("Topologia gwiazdy wymaga przynajmniej 2 węzłów.")
        center = 0
        for i in range(1, n):
            G.add_edge(center, i, weight=d[center][i])

    elif topology == 'large_star':
        # Podobnie jak 'star', można dodać więcej centralnych węzłów
        if n < 2:
            raise ValueError("Topologia large_star wymaga przynajmniej 2 węzłów.")
        center = 0
        for i in range(1, n):
            G.add_edge(center, i, weight=d[center][i])

    elif topology == 'extended_star':
        """
        Generuje topologię extended_star z wieloma centrami gwiazd.
        """
        if n < 2:
            raise ValueError("Topologia extended_star wymaga przynajmniej 2 węzłów.")
        m = star_count  # Liczba centrów gwiazd
        if m < 1 or m > n // 2:
            raise ValueError("star_count musi być pomiędzy 1 a n//2 dla topologii extended_star.")

        nodes = list(range(n))
        random.shuffle(nodes)
        centers = nodes[:m]
        leaves = nodes[m:]

        for center in centers:
            num_leaves = len(leaves) // m
            assigned_leaves = leaves[:num_leaves]
            leaves = leaves[num_leaves:]
            for leaf in assigned_leaves:
                G.add_edge(center, leaf, weight=d[center][leaf])

        for leaf in leaves:
            center = random.choice(centers)
            G.add_edge(center, leaf, weight=d[center][leaf])

        if m > 1:
            for i in range(m - 1):
                G.add_edge(centers[i], centers[i + 1], weight=d[centers[i]][centers[i + 1]])

    elif topology == 'bus':
        """
        Generuje topologię bus (magistrali): liniową magistralę z dodatkowymi węzłami.
        """
        if bus_length <= 0 or bus_length > n:
            bus_length = min(5, n)  # Domyślna długość magistrali
        backbone = list(range(bus_length))
        for i in range(bus_length - 1):
            G.add_edge(backbone[i], backbone[i + 1], weight=d[backbone[i]][backbone[i + 1]])

        remaining_nodes = list(range(bus_length, n))
        for node in remaining_nodes:
            backbone_node = random.randint(0, bus_length - 1)
            G.add_edge(node, backbone_node, weight=d[node][backbone_node])

    elif topology == 'random':
        """
        Generuje losową topologię na podstawie macierzy odległości.
        """
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < edge_prob:
                    G.add_edge(i, j, weight=d[i][j])
        # Upewnienie się, że graf jest spójny
        while not nx.is_connected(G):
            u, v = random.sample(range(n), 2)
            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=d[u][v])

    else:
        raise ValueError(
            "Nieobsługiwana topologia. Wybierz z 'complete', 'grid', 'star', 'large_star', 'extended_star', 'bus', 'random'.")

    return G

def generate_random_input(n, m=1, demand_probability=0.5, max_distance=10, distance_distribution='uniform',
                         topology='complete', **kwargs):
    """
    Generuje losowe dane wejściowe, w tym macierz zapotrzebowania i macierz odległości na podstawie topologii.
    """
    # Generowanie macierzy zapotrzebowania (w)
    w = [[1 if random.random() < demand_probability else 0 for _ in range(m)] for _ in range(n)]

    # Generowanie macierzy odległości (d)
    d = generate_distance_matrix(n, max_distance)

    # Tworzenie grafu na podstawie topologii i macierzy odległości
    G = generate_network_topology(n, topology, d, **kwargs)

    # Używanie macierzy d jako distance_matrix
    distance_matrix = d

    return w, distance_matrix, G

def draw_graph(G, replication_status, title="Graf serwerów i replikacji"):
    """
    Rysuje graf serwerów i replikacji, oznaczając zreplikowane węzły na zielono, a inne na czerwono.
    """
    node_colors = ["green" if replication_status[node] == 1 else "red" for node in G.nodes]

    # Etykiety krawędzi
    edge_labels = {edge: f"{G.edges[edge]['weight']}" for edge in G.edges}

    # Rysowanie grafu
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=600, font_size=12, font_weight="bold")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="blue")

    # Legenda
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Replikowany', markerfacecolor='green', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Nie replikowany', markerfacecolor='red', markersize=12)
    ]
    plt.legend(handles=legend_elements, loc="upper left")
    plt.title(title)
    plt.axis('off')
    plt.show()

def solve_global_replication_problem(n, m, alpha, w, d, max_distance):
    """
    Rozwiązuje globalny problem replikacji używając solvera PuLP z ograniczeniem maksymalnej odległości.
    """
    problem = LpProblem("Global_Replication_Minimization", LpMinimize)
    x = LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(m)), cat='Binary')
    y = LpVariable.dicts("y", ((i, j, k) for i in range(n) for j in range(m) for k in range(n)), cat='Binary')

    # Funkcja celu
    problem += lpSum(alpha * x[i, j] for i in range(n) for j in range(m)) + \
               lpSum(w[i][j] * d[i][k] * y[i, j, k] for i in range(n) for j in range(m) for k in range(n))

    # Ograniczenia
    for i in range(n):
        for j in range(m):
            problem += lpSum(y[i, j, k] for k in range(n)) == w[i][j], f"Access_{i}_{j}"

    for i in range(n):
        for j in range(m):
            for k in range(n):
                problem += x[k, j] >= y[i, j, k], f"Replication_Constraint_{i}_{j}_{k}"
                if d[i][k] > max_distance:
                    problem += y[i, j, k] == 0, f"MaxDistance_Constraint_{i}_{j}_{k}"

    solver = PULP_CBC_CMD(msg=False)
    problem.solve(solver)

    status = LpStatus[problem.status]
    total_cost = value(problem.objective)

    replication_status = [0] * n
    final_configuration = [[0] * n for _ in range(m)]

    for j in range(m):
        for i in range(n):
            if value(x[i, j]) == 1:
                replication_status[i] = 1
                final_configuration[j][i] = 1

    return status, total_cost, replication_status, final_configuration

class BasicGame:
    def __init__(self, n_nodes, alpha, distance_matrix, w, max_distance):
        self.n = n_nodes
        self.alpha = alpha
        self.distance_matrix = distance_matrix
        self.w = w  # Macierz zapotrzebowania
        self.nodes = [0] * n_nodes
        self.access = [[] for _ in range(len(w[0]))]  # Lista dostępu dla każdego obiektu
        self.max_distance = max_distance

    def initialize(self):
        L1 = random.sample(range(self.n), random.randint(1, self.n))
        for i in range(self.n):
            self.nodes[i] = 1 if i in L1 else 0

    def assign_access(self):
        """
        Przypisz dostęp do obiektów na najbliższy zreplikowany serwer w ramach max_distance.
        """
        self.access = [[] for _ in range(len(self.w[0]))]  # Reset access
        for j in range(len(self.w[0])):  # Dla każdego obiektu
            for i in range(self.n):  # Dla każdego serwera
                if self.w[i][j] == 1:
                    # Znajdź najbliższego zreplikowanego serwera w ramach max_distance
                    min_distance = float('inf')
                    assigned_server = -1
                    for k in range(self.n):
                        if self.nodes[k] == 1 and self.distance_matrix[i][k] < min_distance and self.distance_matrix[i][k] <= self.max_distance:
                            min_distance = self.distance_matrix[i][k]
                            assigned_server = k
                    if assigned_server != -1:
                        self.access[j].append(assigned_server)
                    else:
                        # Jeśli nie ma żadnego zreplikowanego serwera w ramach max_distance, replikuj na tym serwerze
                        self.nodes[i] = 1
                        self.access[j].append(i)

    def move_selection(self):
        """
        Algorytm heurystyczny do wyboru replikacji z uwzględnieniem max_distance.
        """
        improved = True
        while improved:
            improved = False
            for i in range(self.n):
                if self.nodes[i] == 1:
                    continue

                # Oblicz koszt replikacji na serwerze i
                cost_replicate = self.alpha

                # Oblicz koszt pozostawienia bez replikacji
                cost_access = 0
                for j in range(len(self.w[0])):
                    if self.w[i][j] == 1:
                        # Aktualny koszt dostępu
                        current_min = min(
                            [self.distance_matrix[i][k] for k in range(self.n)
                             if self.nodes[k] == 1 and self.distance_matrix[i][k] <= self.max_distance],
                            default=float('inf'))
                        # Potencjalny nowy koszt po replikacji na serwerze i
                        new_min = self.distance_matrix[i][i]  # 0
                        cost_access += min(current_min, new_min)

                # Decyzja
                if cost_replicate < cost_access:
                    self.nodes[i] = 1
                    improved = True
                    # Po replikacji, ponownie przypisz dostęp, aby uwzględnić nową replikę
                    self.assign_access()

    def run(self):
        start_time = time.perf_counter()
        self.initialize()
        self.assign_access()
        self.move_selection()
        self.assign_access()
        final_config = self.nodes.copy()
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        return final_config, self.access, elapsed_time

def calculate_cost(alpha, replication_status, access, w, d):
    """
    Oblicz całkowity koszt dla danej konfiguracji.
    """
    replication_cost = alpha * sum(replication_status)
    access_cost = 0
    for j in range(len(access)):  # Dla każdego obiektu
        for i in range(len(w)):  # Dla każdego serwera
            if w[i][j] == 1:
                # Minimalna odległość do zreplikowanego serwera
                min_distance = min([d[i][k] for k in access[j]], default=0)
                access_cost += min_distance
    total_cost = replication_cost + access_cost
    return total_cost

def is_nash_equilibrium(config, n, m, alpha, w, d, max_distance):
    """
    Sprawdza, czy dana konfiguracja replikacji jest równowagą Nasha z uwzględnieniem max_distance.
    """
    for i in range(n):
        # Oblicz koszt dla serwera i
        if config[i] == 1:
            # Koszt replikacji
            current_cost = alpha
            # Koszt dostępu dla obiektów, które tego serwera używają
            access_cost = 0
            for j in range(m):
                if w[i][j] == 1:
                    # Minimalny koszt dostępu dla obiektu j
                    min_distance = min([d[i][k] for k in range(n) if config[k] == 1 and d[i][k] <= max_distance], default=float('inf'))
                    access_cost += min_distance
            total_cost_with_replicate = current_cost + access_cost

            # Koszt bez replikacji
            access_cost = 0
            for j in range(m):
                if w[i][j] == 1:
                    min_distance = min([d[i][k] for k in range(n) if config[k] == 1 and k != i and d[i][k] <= max_distance], default=float('inf'))
                    if min_distance == float('inf'):
                        # Serwer musi zreplikować, aby obsłużyć swoje wymagania
                        return False  # Brak równowagi, ponieważ serwer ma motywację do repliki
                    access_cost += min_distance
            total_cost_without_replicate = access_cost

            # Sprawdzenie, czy serwer i ma motywację do zmiany swojej decyzji
            if total_cost_with_replicate > total_cost_without_replicate:
                return False
        else:
            # Koszt replikacji (jeśli zdecyduje się replikować)
            replicate_cost = alpha
            access_cost = 0
            for j in range(m):
                if w[i][j] == 1:
                    min_distance = min([d[i][k] for k in range(n) if config[k] == 1 or k == i and d[i][k] <= max_distance], default=float('inf'))
                    access_cost += min_distance
            total_cost_with_replicate = replicate_cost + access_cost

            # Koszt bez replikacji
            access_cost = 0
            for j in range(m):
                if w[i][j] == 1:
                    min_distance = min([d[i][k] for k in range(n) if config[k] == 1 and d[i][k] <= max_distance], default=float('inf'))
                    if min_distance == float('inf'):
                        # Brak replikacji oznacza nieskończony koszt dostępu
                        return False  # Serwer ma motywację do repliki
                    access_cost += min_distance
            total_cost_without_replicate = access_cost

            # Sprawdzenie, czy serwer i ma motywację do zmiany swojej decyzji
            if total_cost_with_replicate < total_cost_without_replicate:
                return False

    return True

def enumerate_nash_equilibria(n, m, alpha, w, d, max_distance):
    """
    Enumeruje wszystkie równowagi Nasha w problemie replikacji z uwzględnieniem max_distance.
    """
    equilibria = []
    start_time = time.perf_counter()
    total_configs = 2 ** n
    print(f"Rozpoczynam enumerację {total_configs} konfiguracji...")

    for bits in product([0, 1], repeat=n):
        config = list(bits)
        if is_nash_equilibrium(config, n, m, alpha, w, d, max_distance):
            equilibria.append(config)

    end_time = time.perf_counter()
    print(f"Enumeracja zakończona. Znaleziono {len(equilibria)} równowag Nasha w czasie {end_time - start_time:.6f} sekund.")
    return equilibria

def compute_price_of_anarchy(equilibria, social_optimum_cost, d, w, alpha, n, m, max_distance):
    """
    Oblicza Price of Anarchy i Optimistic Price of Anarchy z uwzględnieniem max_distance.
    """
    costs = []
    for eq in equilibria:
        access = [[] for _ in range(m)]
        for j in range(m):
            for i in range(n):
                if w[i][j] == 1:
                    min_distance = min([d[i][k] for k in range(n) if eq[k] == 1 and d[i][k] <= max_distance], default=float('inf'))
                    if min_distance < float('inf'):
                        # Znajdź wszystkie serwery z minimalną odległością
                        closest_servers = [k for k in range(n) if eq[k] == 1 and d[i][k] == min_distance]
                        # Wybierz najniższy numer serwera jako reprezentanta
                        assigned_server = min(closest_servers)
                        access[j].append(assigned_server)
                    else:
                        access[j].append(i)  # Replikuj na tym serwerze
        cost = calculate_cost(alpha, eq, access, w, d)
        costs.append(cost)
    if not costs:
        return None, None
    worst_cost = max(costs)
    best_cost = min(costs)
    poa = worst_cost / social_optimum_cost
    opt_poa = best_cost / social_optimum_cost
    return poa, opt_poa

def compare_algorithms(n, alpha, m=1, demand_probability=0.5, max_distance=10, distance_distribution='uniform',
                       topology='complete', runs=1000, **kwargs):
    """
    Porównuje czas i efektywność solvera PuLP oraz algorytmu heurystycznego BasicGame.
    Zapisuje wyniki do pliku CSV, wykorzystując te same dane wejściowe dla wszystkich uruchomień.
    Dodatkowo wyświetla dane wejściowe.
    """
    # Przygotowanie pliku CSV
    csv_file = 'results.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Nagłówki kolumn
        headers = [
            'Run',
            'Solver_Status',
            'Solver_Total_Cost',
            'Solver_Num_Replicas',
            'Solver_Time_Microseconds',
            'Heuristic_Total_Cost',
            'Heuristic_Num_Replicas',
            'Heuristic_Time_Microseconds',
            'PoA',
            'Opt_PoA'
        ]
        writer.writerow(headers)

    # Inicjalizacja list do przechowywania wyników PoA i OptPoA
    poa_list = []
    opt_poa_list = []

    # Generowanie losowych danych wejściowych raz przed pętlą
    w, d, G = generate_random_input(n, m, demand_probability, max_distance, distance_distribution, topology, **kwargs)

    # Wyświetlenie danych wejściowych
    print("\n=== Dane Wejściowe ===")
    print("\nMacierz zapotrzebowania (w):")
    for i in range(n):
        print(f"Serwer {i}: {w[i]}")

    print("\nMacierz odległości (d):")
    for i in range(n):
        distances = ', '.join(str(d[i][j]) for j in range(n))
        print(f"Serwer {i}: [{distances}]")

    print("\nTopologia sieci (G) - Krawędzie z wagami:")
    for u, v, data in G.edges(data=True):
        print(f"({u}, {v}) - Weight: {data['weight']}")

    print("\n=== Rozpoczynanie porównania ===\n")

    for run in range(1, runs + 1):
        # W każdym uruchomieniu używamy tych samych danych 'w', 'd', 'G'

        # Rozwiązanie za pomocą solvera
        start_time_solver = time.perf_counter()
        status, total_cost_solver, replication_status_solver, final_config_solver = solve_global_replication_problem(n, m, alpha, w, d, max_distance)
        end_time_solver = time.perf_counter()
        elapsed_time_solver = end_time_solver - start_time_solver
        solver_replicas = sum(replication_status_solver)

        # Rozwiązanie za pomocą algorytmu heurystycznego
        start_time_heuristic = time.perf_counter()
        game = BasicGame(n, alpha, d, w, max_distance)
        final_config_game, access_game, elapsed_time_heuristic = game.run()
        end_time_heuristic = time.perf_counter()
        elapsed_time_heuristic_total = end_time_heuristic - start_time_heuristic
        cost_game = calculate_cost(alpha, final_config_game, access_game, w, d)
        heuristic_replicas = sum(final_config_game)

        # Obliczenie PoA i OptPoA (jeśli n <= 30)
        if n <= 25:
            equilibria = enumerate_nash_equilibria(n, m, alpha, w, d, max_distance)
            poa, opt_poa = compute_price_of_anarchy(equilibria, total_cost_solver, d, w, alpha, n, m, max_distance)
            # Obsługa przypadków, gdy PoA lub OptPoA nie mogą być obliczone
            poa_str = f"{poa:.6f}" if poa is not None else ''
            opt_poa_str = f"{opt_poa:.6f}" if opt_poa is not None else ''
            poa_list.append(poa if poa is not None else 0)
            opt_poa_list.append(opt_poa if opt_poa is not None else 0)
        else:
            poa, opt_poa = '', ''
            poa_str, opt_poa_str = '', ''

        # Zapisanie wyników do CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            row = [
                run,
                status,
                total_cost_solver,
                solver_replicas,
                elapsed_time_solver * 1e6,  # Mikrosekundy
                cost_game,
                heuristic_replicas,
                elapsed_time_heuristic_total * 1e6,  # Mikrosekundy
                poa_str,
                opt_poa_str
            ]
            writer.writerow(row)

        # Opcjonalne: Drukowanie postępu
        if run % max(1, runs // 10) == 0:
            print(f"Run {run}/{runs} completed.")

    # Po zakończeniu zapisów do CSV, odczytujemy dane dla podsumowania
    # Odczytanie wyników z pliku CSV
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        solver_times = []
        heuristic_times = []
        solver_costs = []
        heuristic_costs = []
        solver_replicas = []
        heuristic_replicas = []
        poa_values = []
        opt_poa_values = []

        for row in reader:
            if row['Run'] == 'Run':  # Skip header
                continue
            solver_times.append(float(row['Solver_Time_Microseconds']))
            heuristic_times.append(float(row['Heuristic_Time_Microseconds']))
            solver_costs.append(float(row['Solver_Total_Cost']))
            heuristic_costs.append(float(row['Heuristic_Total_Cost']))
            solver_replicas.append(int(row['Solver_Num_Replicas']))
            heuristic_replicas.append(int(row['Heuristic_Num_Replicas']))
            if row['PoA']:
                poa_values.append(float(row['PoA']))
            if row['Opt_PoA']:
                opt_poa_values.append(float(row['Opt_PoA']))

    # Obliczenie średnich wartości
    average_time_solver = sum(solver_times) / runs
    average_time_heuristic = sum(heuristic_times) / runs
    average_cost_solver = sum(solver_costs) / runs
    average_cost_heuristic = sum(heuristic_costs) / runs
    average_replicas_solver = sum(solver_replicas) / runs
    average_replicas_heuristic = sum(heuristic_replicas) / runs

    # Obliczenie średnich wartości PoA i OptPoA, jeśli są dostępne
    if poa_values and opt_poa_values:
        average_poa = sum(poa_values) / len(poa_values)
        average_opt_poa = sum(opt_poa_values) / len(opt_poa_values)
    else:
        average_poa = None
        average_opt_poa = None

    print("\n=== Solver (PuLP) ===")
    print(f"Średni czas wykonania Solver (PuLP) po {runs} uruchomieniach: {average_time_solver:.3f} mikrosekund")
    print(f"Średni całkowity koszt Solver: {average_cost_solver}")
    print(f"Średnia liczba replikowanych serwerów (Solver): {average_replicas_solver:.2f}")

    print("\n=== Algorytm Heurystyczny (BasicGame) ===")
    print(f"Średni czas wykonania Heurystyka (BasicGame) po {runs} uruchomieniach: {average_time_heuristic:.3f} mikrosekund")
    print(f"Średni całkowity koszt Heurystyki: {average_cost_heuristic}")
    print(f"Średnia liczba replikowanych serwerów (Heurystyka): {average_replicas_heuristic:.2f}")

    # Rysowanie wyników (wykonujemy tylko raz z ostatniego runa)
    draw_graph(G, replication_status_solver, title="Solver (PuLP) - Replikacja")
    draw_graph(G, final_config_game, title="Algorytm Heurystyczny (BasicGame) - Replikacja")

    # Porównanie wyników
    print("\n=== Porównanie ===")
    print(f"Solver - Średni całkowity koszt: {average_cost_solver}")
    print(f"Heurystyka - Średni całkowity koszt: {average_cost_heuristic}")
    print(f"Solver - Średnia liczba replikowanych serwerów: {average_replicas_solver:.2f}")
    print(f"Heurystyka - Średnia liczba replikowanych serwerów: {average_replicas_heuristic:.2f}")
    print(f"Solver - Średni czas wykonania: {average_time_solver:.3f} mikrosekund")
    print(f"Heurystyka - Średni czas wykonania: {average_time_heuristic:.3f} mikrosekund")

    # Obliczenie Równowag Nasha (jeśli n <= 25)
    if n <= 25:
        if poa_values and opt_poa_values:
            poa_avg = average_poa
            opt_poa_avg = average_opt_poa
            print("\n=== Wskaźniki Równowagi Nasha ===")
            print(f"Średnia Price of Anarchy (PoA): {poa_avg:.6f}")
            print(f"Średnia Optimistic Price of Anarchy (OptPoA): {opt_poa_avg:.6f}")
        else:
            print("\nNie znaleziono żadnych równowag Nasha.")
    else:
        print("\nEnumeracja równowag Nasha dla n > 25 jest zbyt czasochłonna.")

    print(f"\nWyniki zostały zapisane do pliku {csv_file}.")

def calculate_cost(alpha, replication_status, access, w, d):
    """
    Oblicz całkowity koszt dla danej konfiguracji.
    """
    replication_cost = alpha * sum(replication_status)
    access_cost = 0
    for j in range(len(access)):  # Dla każdego obiektu
        for i in range(len(w)):  # Dla każdego serwera
            if w[i][j] == 1:
                # Minimalna odległość do zreplikowanego serwera
                min_distance = min([d[i][k] for k in access[j]], default=0)
                access_cost += min_distance
    total_cost = replication_cost + access_cost
    return total_cost

def is_nash_equilibrium(config, n, m, alpha, w, d, max_distance):
    """
    Sprawdza, czy dana konfiguracja replikacji jest równowagą Nasha z uwzględnieniem max_distance.
    """
    for i in range(n):
        # Oblicz koszt dla serwera i
        if config[i] == 1:
            # Koszt replikacji
            current_cost = alpha
            # Koszt dostępu dla obiektów, które tego serwera używają
            access_cost = 0
            for j in range(m):
                if w[i][j] == 1:
                    # Minimalny koszt dostępu dla obiektu j
                    min_distance = min([d[i][k] for k in range(n) if config[k] == 1 and d[i][k] <= max_distance], default=float('inf'))
                    access_cost += min_distance
            total_cost_with_replicate = current_cost + access_cost

            # Koszt bez replikacji
            access_cost = 0
            for j in range(m):
                if w[i][j] == 1:
                    min_distance = min([d[i][k] for k in range(n) if config[k] == 1 and k != i and d[i][k] <= max_distance], default=float('inf'))
                    if min_distance == float('inf'):
                        # Serwer musi zreplikować, aby obsłużyć swoje wymagania
                        return False  # Brak równowagi, ponieważ serwer ma motywację do repliki
                    access_cost += min_distance
            total_cost_without_replicate = access_cost

            # Sprawdzenie, czy serwer i ma motywację do zmiany swojej decyzji
            if total_cost_with_replicate > total_cost_without_replicate:
                return False
        else:
            # Koszt replikacji (jeśli zdecyduje się replikować)
            replicate_cost = alpha
            access_cost = 0
            for j in range(m):
                if w[i][j] == 1:
                    min_distance = min([d[i][k] for k in range(n) if config[k] == 1 or k == i and d[i][k] <= max_distance], default=float('inf'))
                    access_cost += min_distance
            total_cost_with_replicate = replicate_cost + access_cost

            # Koszt bez replikacji
            access_cost = 0
            for j in range(m):
                if w[i][j] == 1:
                    min_distance = min([d[i][k] for k in range(n) if config[k] == 1 and d[i][k] <= max_distance], default=float('inf'))
                    if min_distance == float('inf'):
                        # Brak replikacji oznacza nieskończony koszt dostępu
                        return False  # Serwer ma motywację do repliki
                    access_cost += min_distance
            total_cost_without_replicate = access_cost

            # Sprawdzenie, czy serwer i ma motywację do zmiany swojej decyzji
            if total_cost_with_replicate < total_cost_without_replicate:
                return False

    return True

def enumerate_nash_equilibria(n, m, alpha, w, d, max_distance):
    """
    Enumeruje wszystkie równowagi Nasha w problemie replikacji z uwzględnieniem max_distance.
    """
    equilibria = []
    start_time = time.perf_counter()
    total_configs = 2 ** n
    print(f"Rozpoczynam enumerację {total_configs} konfiguracji...")

    for bits in product([0, 1], repeat=n):
        config = list(bits)
        if is_nash_equilibrium(config, n, m, alpha, w, d, max_distance):
            equilibria.append(config)

    end_time = time.perf_counter()
    print(f"Enumeracja zakończona. Znaleziono {len(equilibria)} równowag Nasha w czasie {end_time - start_time:.6f} sekund.")
    return equilibria

def compute_price_of_anarchy(equilibria, social_optimum_cost, d, w, alpha, n, m, max_distance):
    """
    Oblicza Price of Anarchy i Optimistic Price of Anarchy z uwzględnieniem max_distance.
    """
    costs = []
    for eq in equilibria:
        access = [[] for _ in range(m)]
        for j in range(m):
            for i in range(n):
                if w[i][j] == 1:
                    min_distance = min([d[i][k] for k in range(n) if eq[k] == 1 and d[i][k] <= max_distance], default=float('inf'))
                    if min_distance < float('inf'):
                        # Znajdź wszystkie serwery z minimalną odległością
                        closest_servers = [k for k in range(n) if eq[k] == 1 and d[i][k] == min_distance]
                        # Wybierz najniższy numer serwera jako reprezentanta
                        assigned_server = min(closest_servers)
                        access[j].append(assigned_server)
                    else:
                        access[j].append(i)  # Replikuj na tym serwerze
        cost = calculate_cost(alpha, eq, access, w, d)
        costs.append(cost)
    if not costs:
        return None, None
    worst_cost = max(costs)
    best_cost = min(costs)
    poa = worst_cost / social_optimum_cost
    opt_poa = best_cost / social_optimum_cost
    return poa, opt_poa

def compare_algorithms(n, alpha, m=1, demand_probability=0.5, max_distance=10, distance_distribution='uniform',
                       topology='complete', runs=1000, **kwargs):
    """
    Porównuje czas i efektywność solvera PuLP oraz algorytmu heurystycznego BasicGame.
    Zapisuje wyniki do pliku CSV, wykorzystując te same dane wejściowe dla wszystkich uruchomień.
    Dodatkowo wyświetla dane wejściowe.
    """
    # Przygotowanie pliku CSV
    csv_file = 'results.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Nagłówki kolumn
        headers = [
            'Run',
            'Solver_Status',
            'Solver_Total_Cost',
            'Solver_Num_Replicas',
            'Solver_Time_Microseconds',
            'Heuristic_Total_Cost',
            'Heuristic_Num_Replicas',
            'Heuristic_Time_Microseconds',
            'PoA',
            'Opt_PoA'
        ]
        writer.writerow(headers)

    # Inicjalizacja list do przechowywania wyników PoA i OptPoA
    poa_list = []
    opt_poa_list = []

    # Generowanie losowych danych wejściowych raz przed pętlą
    w, d, G = generate_random_input(n, m, demand_probability, max_distance, distance_distribution, topology, **kwargs)

    # Wyświetlenie danych wejściowych
    print("\n=== Dane Wejściowe ===")
    print("\nMacierz zapotrzebowania (w):")
    for i in range(n):
        print(f"Serwer {i}: {w[i]}")

    print("\nMacierz odległości (d):")
    for i in range(n):
        distances = ', '.join(str(d[i][j]) for j in range(n))
        print(f"Serwer {i}: [{distances}]")

    print("\nTopologia sieci (G) - Krawędzie z wagami:")
    for u, v, data in G.edges(data=True):
        print(f"({u}, {v}) - Weight: {data['weight']}")

    print("\n=== Rozpoczynanie porównania ===\n")

    for run in range(1, runs + 1):
        # W każdym uruchomieniu używamy tych samych danych 'w', 'd', 'G'

        # Rozwiązanie za pomocą solvera
        start_time_solver = time.perf_counter()
        status, total_cost_solver, replication_status_solver, final_config_solver = solve_global_replication_problem(n, m, alpha, w, d, max_distance)
        end_time_solver = time.perf_counter()
        elapsed_time_solver = end_time_solver - start_time_solver
        solver_replicas = sum(replication_status_solver)

        # Rozwiązanie za pomocą algorytmu heurystycznego
        start_time_heuristic = time.perf_counter()
        game = BasicGame(n, alpha, d, w, max_distance)
        final_config_game, access_game, elapsed_time_heuristic = game.run()
        end_time_heuristic = time.perf_counter()
        elapsed_time_heuristic_total = end_time_heuristic - start_time_heuristic
        cost_game = calculate_cost(alpha, final_config_game, access_game, w, d)
        heuristic_replicas = sum(final_config_game)

        # Obliczenie PoA i OptPoA (jeśli n <= 25)
        if n <= 25:
            equilibria = enumerate_nash_equilibria(n, m, alpha, w, d, max_distance)
            poa, opt_poa = compute_price_of_anarchy(equilibria, total_cost_solver, d, w, alpha, n, m, max_distance)
            # Obsługa przypadków, gdy PoA lub OptPoA nie mogą być obliczone
            poa_str = f"{poa:.6f}" if poa is not None else ''
            opt_poa_str = f"{opt_poa:.6f}" if opt_poa is not None else ''
            poa_list.append(poa if poa is not None else 0)
            opt_poa_list.append(opt_poa if opt_poa is not None else 0)
        else:
            poa, opt_poa = '', ''
            poa_str, opt_poa_str = '', ''

        # Zapisanie wyników do CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            row = [
                run,
                status,
                total_cost_solver,
                solver_replicas,
                elapsed_time_solver * 1e6,  # Mikrosekundy
                cost_game,
                heuristic_replicas,
                elapsed_time_heuristic_total * 1e6,  # Mikrosekundy
                poa_str,
                opt_poa_str
            ]
            writer.writerow(row)

        # Opcjonalne: Drukowanie postępu
        if run % max(1, runs // 10) == 0:
            print(f"Run {run}/{runs} completed.")

    # Po zakończeniu zapisów do CSV, odczytujemy dane dla podsumowania
    # Odczytanie wyników z pliku CSV
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        solver_times = []
        heuristic_times = []
        solver_costs = []
        heuristic_costs = []
        solver_replicas = []
        heuristic_replicas = []
        poa_values = []
        opt_poa_values = []

        for row in reader:
            if row['Run'] == 'Run':  # Skip header
                continue
            solver_times.append(float(row['Solver_Time_Microseconds']))
            heuristic_times.append(float(row['Heuristic_Time_Microseconds']))
            solver_costs.append(float(row['Solver_Total_Cost']))
            heuristic_costs.append(float(row['Heuristic_Total_Cost']))
            solver_replicas.append(int(row['Solver_Num_Replicas']))
            heuristic_replicas.append(int(row['Heuristic_Num_Replicas']))
            if row['PoA']:
                poa_values.append(float(row['PoA']))
            if row['Opt_PoA']:
                opt_poa_values.append(float(row['Opt_PoA']))

    # Obliczenie średnich wartości
    average_time_solver = sum(solver_times) / runs
    average_time_heuristic = sum(heuristic_times) / runs
    average_cost_solver = sum(solver_costs) / runs
    average_cost_heuristic = sum(heuristic_costs) / runs
    average_replicas_solver = sum(solver_replicas) / runs
    average_replicas_heuristic = sum(heuristic_replicas) / runs

    # Obliczenie średnich wartości PoA i OptPoA, jeśli są dostępne
    if poa_values and opt_poa_values:
        average_poa = sum(poa_values) / len(poa_values)
        average_opt_poa = sum(opt_poa_values) / len(opt_poa_values)
    else:
        average_poa = None
        average_opt_poa = None

    print("\n=== Solver (PuLP) ===")
    print(f"Średni czas wykonania Solver (PuLP) po {runs} uruchomieniach: {average_time_solver:.3f} mikrosekund")
    print(f"Średni całkowity koszt Solver: {average_cost_solver}")
    print(f"Średnia liczba replikowanych serwerów (Solver): {average_replicas_solver:.2f}")

    print("\n=== Algorytm Heurystyczny (BasicGame) ===")
    print(f"Średni czas wykonania Heurystyka (BasicGame) po {runs} uruchomieniach: {average_time_heuristic:.3f} mikrosekund")
    print(f"Średni całkowity koszt Heurystyki: {average_cost_heuristic}")
    print(f"Średnia liczba replikowanych serwerów (Heurystyka): {average_replicas_heuristic:.2f}")

    # Rysowanie wyników (wykonujemy tylko raz z ostatniego runa)
    draw_graph(G, replication_status_solver, title="Solver (PuLP) - Replikacja")
    draw_graph(G, final_config_game, title="Algorytm Heurystyczny (BasicGame) - Replikacja")

    # Porównanie wyników
    print("\n=== Porównanie ===")
    print(f"Solver - Średni całkowity koszt: {average_cost_solver}")
    print(f"Heurystyka - Średni całkowity koszt: {average_cost_heuristic}")
    print(f"Solver - Średnia liczba replikowanych serwerów: {average_replicas_solver:.2f}")
    print(f"Heurystyka - Średnia liczba replikowanych serwerów: {average_replicas_heuristic:.2f}")
    print(f"Solver - Średni czas wykonania: {average_time_solver:.3f} mikrosekund")
    print(f"Heurystyka - Średni czas wykonania: {average_time_heuristic:.3f} mikrosekund")

    # Obliczenie Równowag Nasha (jeśli n <= 30)
    if n <= 30:
        if poa_values and opt_poa_values:
            poa_avg = average_poa
            opt_poa_avg = average_opt_poa
            print("\n=== Wskaźniki Równowagi Nasha ===")
            print(f"Średnia Price of Anarchy (PoA): {poa_avg:.6f}")
            print(f"Średnia Optimistic Price of Anarchy (OptPoA): {opt_poa_avg:.6f}")
        else:
            print("\nNie znaleziono żadnych równowag Nasha.")
    else:
        print("\nEnumeracja równowag Nasha dla n > 25 jest zbyt czasochłonna.")

    print(f"\nWyniki zostały zapisane do pliku {csv_file}.")

if __name__ == "__main__":
    # Ustawienia parametrów
    n = 15  # Liczba serwerów
    alpha = 8  # Koszt replikacji
    m = 1  # Liczba obiektów (ustawione na 1 zgodnie z wymaganiem)
    demand_probability = 0.5  # Prawdopodobieństwo zapotrzebowania
    max_distance = 10  # Maksymalna odległość
    distance_distribution = 'normal'  # Rozkład odległości ('uniform' lub 'normal')
    topology = 'complete'  # Typ topologii ('complete', 'grid', 'star', 'large_star', 'extended_star', 'bus', 'random')

    # Dodatkowe parametry dla wybranej topologii
    topology_params = {}
    if topology == 'grid':
        topology_params['grid_size'] = (3, 4)  # np. 3x4 siatka dla n=12
    elif topology in ['extended_star']:
        topology_params['star_count'] = 3  # Liczba centralnych gwiazd
    elif topology == 'bus':
        topology_params['bus_length'] = 5  # Długość magistrali
    elif topology == 'random':
        topology_params['edge_prob'] = 0.3  # Prawdopodobieństwo dodania krawędzi w topologii losowej

    # Liczba uruchomień dla dokładniejszego pomiaru czasu
    runs = 1  # Można dostosować w zależności od potrzeb

    # Uruchomienie porównania z wielokrotnymi uruchomieniami dla dokładniejszego pomiaru czasu
    compare_algorithms(n, alpha, m, demand_probability, max_distance, distance_distribution, topology,
                       runs=runs, **topology_params)
