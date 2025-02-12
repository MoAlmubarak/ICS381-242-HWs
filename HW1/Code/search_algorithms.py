import heapq

class PriorityQueue:
    def __init__(self, items=(), priority_function=(lambda x: x)): 
        self.priority_function = priority_function
        self.pqueue = []
        # add the items to the PQ
        for item in items:
            self.add(item)
    
    """
    Add item to PQ with priority-value given by call to priority_function
    """
    def add(self, item):
        pair = (self.priority_function(item), item)
        heapq.heappush(self.pqueue, pair)

    """
    pop and return item from PQ with min priority-value
    """
    def pop(self):
        return heapq.heappop(self.pqueue)[1]

    """
    gets number of items in PQ
    """
    def __len__(self):
        return len(self.pqueue)

class Node:
    def __init__(self, state, parent_node=None, action_from_parent=None, path_cost=0):
        self.state = state
        self.parent_node = parent_node
        self.action_from_parent = action_from_parent
        self.path_cost = path_cost
        if self.parent_node is None:
            self.depth = 0
        else:
            self.depth = self.parent_node.depth + 1
    
    def __lt__(self, other):
        return self.state < other.state 


############################## Search Algorithms #############################

# implement best-first-search template
# see https://aima.cs.berkeley.edu/figures.pdf#page=11
def expand(problem, node):
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s1)
        yield Node(s1, node, action, cost)
        
def best_first_search(problem, f):
    node = Node(problem.initial_state)
    frontier = PriorityQueue([node], priority_function=f)
    reached = {problem.initial_state: node}
    while len(frontier) > 0:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.add(child)
    return None

def best_first_search_treelike(problem, f):
    node = Node(problem.initial_state)
    frontier = PriorityQueue([node], priority_function=f)
    while len(frontier) > 0:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            frontier.add(child)
    return None
    


def get_path_actions(node):
    if node is None or node.parent_node is None:
        return []
    return get_path_actions(node.parent_node) + [node.action_from_parent]

def get_path_states(node):
    if node is None: 
        return []
    return get_path_states(node.parent_node) + [node.state]
    
def breadth_first_search(problem, treelike=False):
    bfs_f = (lambda node: node.depth)
    if treelike:
        return best_first_search_treelike(problem, f=bfs_f)
    else:
        return best_first_search(problem, f=bfs_f)

def depth_first_search(problem, treelike=False):
    dfs_f = (lambda node: -node.depth)
    if treelike:
        return best_first_search_treelike(problem, f=dfs_f)
    else:
        return best_first_search(problem, f=dfs_f)

def uniform_cost_search(problem, treelike=False):
    ucs_f = (lambda node: node.path_cost)
    if treelike:
        return best_first_search_treelike(problem, f=ucs_f)
    else:
        return best_first_search(problem, f=ucs_f)

def greedy_search(problem, h, treelike=False):
    if treelike:
        return best_first_search_treelike(problem, f=h)
    else:
        return best_first_search(problem, f=h)

def astar_search(problem, h, treelike=False):
    if treelike:
        return best_first_search_treelike(problem, f=(lambda node: node.path_cost + h(node)))
    else:
        return best_first_search(problem, f=(lambda node: node.path_cost + h(node)))