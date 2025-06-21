"""classroom_allocation.py – 
Modelo e heurísticas para otimizar a alocação de salas minimizando
custo de locomoção de professores e alunos.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import math
import random

# ----------------
# Modelagem básica
# ----------------
@dataclass
class Room:
    id: str
    capacity: int
    coord: Tuple[float, float]  # (x, y) no plano do campus
    schedule: Dict[int, str] = field(default_factory=dict)  # timeslot -> course_id

    def is_available(self, timeslot: int, size: int) -> bool:
        return timeslot not in self.schedule and self.capacity >= size

@dataclass
class Person:
    id: str
    coord: Tuple[float, float]

@dataclass
class Course:
    id: str
    size: int
    teacher: Person
    students: List[Person]
    timeslot: int  # inteiro representando horário (ex.: 0 = seg‑8h, 1 = seg‑10h …)

# ---------------
# Funções utilitárias
# ---------------

def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.dist(a, b)

def travel_cost(course: Course, room: Room) -> float:
    """Soma distâncias professor + alunos (pode amostrar p/ ganhar desempenho)."""
    teacher_cost = euclidean(course.teacher.coord, room.coord)
    # Amostragem dos alunos p/ evitar O(N) em turmas grandes (parâmetro k)
    k = min(30, len(course.students))
    sample = random.sample(course.students, k) if k < len(course.students) else course.students
    student_cost = sum(euclidean(s.coord, room.coord) for s in sample) * (len(course.students) / k)
    return teacher_cost + student_cost

# -----------------------
# 1. Algoritmo Guloso
# -----------------------

def greedy_allocate(courses: List[Course], rooms: List[Room]) -> Dict[str, str]:
    """Retorna dict course_id -> room_id"""
    allocation: Dict[str, str] = {}
    # Ordena cursos decrescentemente pelo tamanho (first‑fit decreasing)
    for course in sorted(courses, key=lambda c: c.size, reverse=True):
        feasible = [r for r in rooms if r.is_available(course.timeslot, course.size)]
        if not feasible:
            raise RuntimeError(f"Sem sala disponível para disciplina {course.id} no horário {course.timeslot}")
        best_room = min(feasible, key=lambda r: travel_cost(course, r))
        best_room.schedule[course.timeslot] = course.id
        allocation[course.id] = best_room.id
    return allocation

# -----------------------
# 2. Busca Local (swap‑based)
# -----------------------

def local_search(allocation: Dict[str, str], courses: List[Course], rooms: Dict[str, Room],
                 max_iter: int = 10_000) -> Dict[str, str]:
    """Hill climbing + retrocesso se atingir mínimo local."""
    def cost_of_alloc(alloc: Dict[str, str]) -> float:
        return sum(travel_cost(course, rooms[alloc[course.id]]) for course in courses)

    current = allocation.copy()
    current_cost = cost_of_alloc(current)
    for _ in range(max_iter):
        c1, c2 = random.sample(courses, 2)
        if c1.timeslot != c2.timeslot:  # só trocamos se horários coincidem
            continue
        r1, r2 = rooms[current[c1.id]], rooms[current[c2.id]]
        # Verifica restrições de capacidade
        if r1.capacity < c2.size or r2.capacity < c1.size:
            continue
        # Custo após a troca
        delta = (
            travel_cost(c1, r2) + travel_cost(c2, r1)
            - travel_cost(c1, r1) - travel_cost(c2, r2)
        )
        if delta < 0:  # melhora!
            current[c1.id], current[c2.id] = r2.id, r1.id
            current_cost += delta
    return current

# ------------------------------------------
# 3. Algoritmo Genético (versão compacta)
# ------------------------------------------

def genetic_allocate(courses: List[Course], rooms: List[Room], pop_size: int = 50,
                     generations: int = 200, crossover_rate: float = 0.8, mutation_rate: float = 0.2) -> Dict[str, str]:
    """Chromossomo: lista de indices de salas para cada curso (ordem fixa)."""
    rng = random.Random(42)
    n_courses = len(courses)

    # Pré‑processa salas viáveis por curso/horário
    feasible_rooms = [
        [i for i, r in enumerate(rooms) if r.capacity >= course.size and course.timeslot not in r.schedule]
        for course in courses
    ]

    def random_chrom():
        return [rng.choice(fr) for fr in feasible_rooms]

    def fitness(chrom):
        return sum(travel_cost(courses[i], rooms[idx]) for i, idx in enumerate(chrom))

    population = [random_chrom() for _ in range(pop_size)]

    for _ in range(generations):
        # Seleção por torneio 2‑way
        parents = [min(random.sample(population, 2), key=fitness) for _ in range(pop_size)]
        offspring = []
        for i in range(0, pop_size, 2):
            p1, p2 = parents[i], parents[i + 1]
            if rng.random() < crossover_rate:
                cut = rng.randrange(1, n_courses)
                child1 = p1[:cut] + p2[cut:]
                child2 = p2[:cut] + p1[cut:]
            else:
                child1, child2 = p1[:], p2[:]
            offspring.extend([child1, child2])
        # Mutação
        for chrom in offspring:
            if rng.random() < mutation_rate:
                idx = rng.randrange(n_courses)
                chrom[idx] = rng.choice(feasible_rooms[idx])
        # Substituição: elitismo 1
        population.extend(offspring)
        population.sort(key=fitness)
        population = population[:pop_size]
    best = min(population, key=fitness)
    return {course.id: rooms[idx].id for course, idx in zip(courses, best)}

# --------------------------
# Exemplo mínimo de uso real
# --------------------------
# if __name__ == "__main__":
#     # Geração de dados sintéticos p/ demonstração
#     rng = random.Random(0)
#     rooms = [Room(f"R{i}", capacity=rng.randint(30, 120), coord=(rng.random()*100, rng.random()*100)) for i in range(20)]

#     teachers = [Person(f"T{i}", coord=(rng.random()*100, rng.random()*100)) for i in range(10)]
#     students_pool = [Person(f"S{i}", coord=(rng.random()*100, rng.random()*100)) for i in range(300)]

#     courses = []
#     for i in range(25):
#         size = rng.randint(20, 80)
#         teacher = rng.choice(teachers)
#         students = rng.sample(students_pool, size)
#         timeslot = rng.randint(0, 4)  # 5 horários distintos
#         courses.append(Course(f"C{i}", size, teacher, students, timeslot))

#     # Alocação gulosa seguida de busca local
#     base_alloc = greedy_allocate(courses, rooms)
#     improved_alloc = local_search(base_alloc, courses, {r.id: r for r in rooms})

#     print("Alocação final (curso -> sala):")
#     for cid, rid in improved_alloc.items():
#         print(cid, "->", rid)
if __name__ == "__main__":
    # Geração de dados sintéticos p/ demonstração
    rng = random.Random(0)
    rooms = [Room(f"R{i}", capacity=rng.randint(30, 120), coord=(rng.random()*100, rng.random()*100)) for i in range(20)]

    teachers = [Person(f"T{i}", coord=(rng.random()*100, rng.random()*100)) for i in range(10)]
    students_pool = [Person(f"S{i}", coord=(rng.random()*100, rng.random()*100)) for i in range(300)]

    courses = []
    for i in range(25):
        size = rng.randint(20, 80)
        teacher = rng.choice(teachers)
        students = rng.sample(students_pool, size)
        timeslot = rng.randint(0, 4)  # 5 horários distintos
        courses.append(Course(f"C{i}", size, teacher, students, timeslot))

    # ------------------------------
    # Execução dos 3 métodos
    # ------------------------------

    # 1) Algoritmo Guloso
    greedy_alloc = greedy_allocate(courses, rooms)

    # 2) Busca Local sobre a solução gulosa
    local_alloc = local_search(greedy_alloc, courses, {r.id: r for r in rooms})

    # 3) Algoritmo Genético
    genetic_alloc = genetic_allocate(courses, rooms)

    # Função auxiliar para custo total
    def total_cost(alloc):
        return sum(travel_cost(c, {r.id: r for r in rooms}[alloc[c.id]]) for c in courses)

    print(" Resultados ---")
    print(f"Guloso:        custo = {total_cost(greedy_alloc):.2f}")
    print(f"Busca Local:   custo = {total_cost(local_alloc):.2f}")
    print(f"Genético:      custo = {total_cost(genetic_alloc):.2f}")

    print("Alocação final (Busca Local) – curso -> sala:")
    for cid, rid in local_alloc.items():
        print(cid, "->", rid)

    print("Alocação (Guloso) – curso -> sala:")
    for cid, rid in greedy_alloc.items():
        print(cid, "->", rid)

    print("Alocação (Genético) – curso -> sala:")
    for cid, rid in genetic_alloc.items():
        print(cid, "->", rid)