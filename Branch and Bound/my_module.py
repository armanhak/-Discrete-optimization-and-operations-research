import networkx as nx
import cplex
from numba import jit
from math import floor, ceil
import time
from cplex.exceptions import CplexSolverError

class CplexSolver():
    """В этом классе читаем граф и создаем cplex солвер"""
    def __init__(self, path):
        """
        Принимает на вход строку, путь к гарфу .clq.
        Атрибуты:
            graph: networkx graph
            solver: cplex maximize relaxation solver for clique
            path: str, путь к графу
        """
        self.path = path
        self.graph = self.read_graph(self.path)
        self.solver = self.create_cplex_solver(self.graph)
    def read_graph(self,path):
        with open(path, 'r') as f:
            graph_text = f.readlines()
        graph = [tuple(map(lambda node: int(node),
                           edges.replace('e','').strip().split(' ')
                          )
                      )
                 for edges in graph_text if edges.startswith('e ')
                ]
        return nx.Graph(graph)
    def create_cplex_solver(self,graph, var_types = 'C'):
        """
        На вход принимает граф и возвращает релаксированый симплекс солвер
        для максимального
        клика.
        graph: networkx graph
        return: cplex
        """
        def get_indep_nodes(graph):
            """
            На вход принимает граф и возвращает не независимые множнества
            вершин. Независимые множества найдем с помощью раскраски графа,
            использую разные стратегии стратегии при 100 запусков

            graph: networkx.graph
            return: indep_nodes
            """
            indep_nodes = []
            for i in range(100):
                strategies = [nx.coloring.strategy_independent_set,
                                  nx.coloring.strategy_random_sequential,
                                  nx.coloring.strategy_largest_first,
                                  nx.coloring.strategy_connected_sequential_bfs,
                                  nx.coloring.strategy_connected_sequential_dfs,
                                  nx.coloring.strategy_saturation_largest_first
                             ]
                for strategy in strategies:
                    node_color_dict = nx.coloring.greedy_color(graph, strategy = strategy)
                    for color in set(node_color_dict.values()):
                        indep_nodes.append(tuple(node
                                               for node, col in node_color_dict.items()
                                               if col==color
                                            ))
            #берем только уникальные независимые множество узлов и те, размер которых > 1
            indep_nodes = set([ind for ind in indep_nodes if len(ind)>1])
            return indep_nodes
        c = cplex.Cplex()
        c.set_log_stream(None)
        c.set_error_stream(None)
        c.set_warning_stream(None)
        c.set_results_stream(None)
        #Variables
        var_names = list(map(lambda x: f'x{x}', graph.nodes())) # variable names
        lb = [0]*len(var_names) # lower bounds
        ub = [1]*len(var_names) # upper bounds
        var_types = [var_types]*len(var_names)  # variables types, I-integer, C-contiues,
        objective = [1]*len(var_names)
        variable_indices = c.variables.add(names = var_names,
                                            obj=objective,
                                            lb = lb,
                                            ub = ub,
                                            types = var_types
                                            )
        #Linear constraints
        indep_nodes = get_indep_nodes(graph)
        lin_expr = [[list(map(lambda x:f'x{x}', nodes)), [1]*len(nodes)] for nodes in indep_nodes]
        lin_constr_rhs = [1]*len(lin_expr)
        lin_constr_senses = ["L"]*len(lin_expr)
        lin_constr_names = [f'linear_constraint{i}' for i in range(1, 1+len(lin_expr))]
        lin_constr_indeces = c.linear_constraints.add(lin_expr=lin_expr,
                                                      rhs=lin_constr_rhs,
                                                      senses=lin_constr_senses,
                                                      names=lin_constr_names
                                                      )
        # Objective function
    #     linear_objective = list(zip(var_names, [1]*len(var_names)))# [('x1', 1), ('x2',1), ...]
    #     c.objective.set_linear(linear_objective)
        c.objective.set_sense(c.objective.sense.maximize)
        return c
class BnBSolver():
    """С помощью этого класса решаем задачу максимального клика методом
        Branch and Bound
    """
    def __init__(self, graph, solver, time_limit):
        """
        graph: networkx graph
        solver: симплекс солвер со всеми настройками и ограничениями
        time_limit: int. Максимальное время выполнения алгоритма
        """
        self.MAX_CLIQUE = 0
        self.SOLUTION = []
        self.graph = graph
        self.solver = solver
        self.work_time = 0
        self.time_limit = time_limit
    @staticmethod
    @jit(nopython=True)
    def _round_variables(x,eps=1e-6):
        """
        Используется для округления переменных релаксированной задачи симплекс

        Если число очень близко к 1 или к 0,
        но из за погрешности дробных чисел отличается от них,
        тогда мы заменим это число либо на 1 либо на 0.
        Если разница большая, то оставимм как есть.
        eps = 1e-6
        """
        if abs(x-1)<eps:
            return 1.0
        elif abs(x) <= eps:
            return 0.0
        else: return x
    @staticmethod
    @jit(nopython=True)
    def _ub_round(x,eps = 1e-6):
        """
        Используется для округления uper bound в релаксированной
        задачи симплекса
        """
        if abs(ceil(x) - x) <= eps:
            return ceil(x)
        else:
            return floor(x)
    def get_branch_index(self,solution):
        """
        Эвристика выбора индекса для дальнейшего ветвления.
        Выбираем индекс наибольшего узла (т.е. индекс переменой,
        которая имеет максимальное значение при релаксированной задачи)
        """
        ind =  max([(ind,BnBSolver._round_variables(var))
                                 for ind, var in enumerate(solution) if not var.is_integer()
                                ],
                                key = lambda x: x[1],default=(None, None)
                               )[0]
        return list(self.graph.nodes())[ind]
    def add_constraint(self, index, rhs):
        """
        Добавим ограничение в модель, либо x_index=0, либо x_index = 1.
        index: индекс переменной
        rhs: int, принимает либо 0 (левая ветка) либо 1 (правая ветка)
             значение переменной с инедексом index
         """
        self.solver.linear_constraints.add(lin_expr=[[[f'x{index}'], [1]]] ,
                                      rhs=[rhs],
                                      senses=["E"],
                                      names=[f'branch_constraint_x{index}={rhs}'],
                                      )
    def delete_constraint(self, index, rhs):
        """
        Удалим ограничение из модель (либо x_index=0 удалим, либо x_index = 1).
        index: индекс переменной,
        rhs: int, принимает либо 0 (левая ветка) либо 1 (правая ветка),
        значение переменной с инедексом index
         """
        self.solver.linear_constraints.delete(f'branch_constraint_x{index}={rhs}')
    def run_bnb(self):
        """Рекурсивно ветвимся, алгоритм branch and bound"""
        #Если время работы bnb не перевышает заданное время
        if self.work_time<self.time_limit:
            try:
                #для splex зададим ограничение по времении выполнения
                self.solver.parameters.timelimit.set(self.time_limit - self.work_time)
                now = time.time()
                self.solver.solve()
                self.work_time+=time.time()-now
                solution = list(map(BnBSolver._round_variables, self.solver.solution.get_values()))
                ub = self._ub_round(self.solver.solution.get_objective_value())
#                 checkKeyboardInterrupt(solver)
        #         print("*"*5, 'UPER BOUND', ub)
            except CplexSolverError as no_solution_err:
                print(no_solution_err, 'errr')
#                 if self.work_time>self.time_limit:
#                     return self.MAX_CLIQUE, self.SOLUTION
#                 else:
                return 0
            except KeyboardInterrupt:
                print('1',KeyboardInterrupt)
                return self.MAX_CLIQUE, self.SOLUTION
            #check, if all variables are integer
            if sum(map(float.is_integer, solution)) == self.graph.number_of_nodes():
                if ub > self.MAX_CLIQUE:
                    self.MAX_CLIQUE = ub
                    self.SOLUTION = solution.copy()
                    print('---------UPDATE MAXCLIQUE-------------', self.MAX_CLIQUE)
                    return self.MAX_CLIQUE, self.SOLUTION


            # ub - уже округлен
            if ub > self.MAX_CLIQUE:
                branch_i = self.get_branch_index(solution)
                self.add_constraint(branch_i, 1)
        #         print(f'ADD branch x{branch_i}=1',f'MAX CLIQUE = {MAX_CLIQUE}')
                right_branch = self.run_bnb()
                self.delete_constraint(branch_i, 1)
        #         print(f'DEL branch x{branch_i}=1',f'MAX CLIQUE = {MAX_CLIQUE}')
        #         print("#"*35)

                self.add_constraint(branch_i, 0)
        #         print(f'ADD branch x{branch_i}=0', f'MAX CLIQUE = {MAX_CLIQUE}')
                left_branch  = self.run_bnb()
                self.delete_constraint(branch_i, 0)
        #         print(f'DEL branch x{branch_i}=0',f'MAX CLIQUE = {MAX_CLIQUE}')
        #         print("#"*35)
                return max(right_branch, left_branch, key = lambda x:x[0] if isinstance(x, tuple) else x )
            else:
                return 0
        else:
#             self.work_time = round(self.work_time,3)
            return self.MAX_CLIQUE, self.SOLUTION
