import copy
import networkx as nx
import numpy as np

class CayleyGraph:
    """Defines a cayley graph of a finetly generated free group of two functions and allows
    you to act on the graph by supplying a initial value.
    """
    
    def __init__(self, steps, func_dict, func_bases, scale = None):
        
        if scale is None:
            scale = 5*2**(steps+1)
        
        # Setup functions to be used
        self.identity = lambda x: x
        self.func_dict = func_dict
        self.func_bases = func_bases
        self.coord_diff = lambda i: scale/(2**i)
        
        # Initialize iteration
        self.G = nx.Graph()
        
        origin = np.array((0.0, 0.0))
        initial_node = self.create_node(0, self.identity, self.identity, origin)
        
        self.G.add_node(initial_node[0], **initial_node[1])
        self.prev_nodes = [initial_node]
        self.next_id = 1
        
        for step in range(steps):
            self.iterate(step)
    
    def create_node(self, node_id, func, inv_func, coord_vec, value=None):
        
            node_data = {'func': func,
                         'inv_func': inv_func,
                         'coord': coord_vec,
                         'value': value}
            
            return (node_id, node_data)
    
    def create_new_node(self, prev_node, new_id, func, step):
        
        # Calculate the new euclidean coordinates of the node
        current_pos = prev_node[1]['coord']
        delta_pos = self.coord_diff(step)*self.func_bases[func]
        new_coord = current_pos + delta_pos
        
        # Create new node data and add it to the graph
        new_node = self.create_node(self.next_id,
                                    func,
                                    self.func_dict[func],
                                    new_coord)
        
        return new_node

    def iterate(self, step):
        
        new_nodes = []
        
        for prev_node in self.prev_nodes:
            for func in self.func_dict:
                if func != prev_node[1]['inv_func']:
                    new_node = self.create_new_node(prev_node, self.next_id, func, step)
                    new_nodes.append(new_node)
                    self.next_id += 1
                    self.G.add_node(new_node[0], **new_node[1])
                    self.G.add_edge(prev_node[0], new_node[0])
                    
        self.prev_nodes = new_nodes
    
    def act_on(self, initial_value):
        
        self.G.nodes[0]['value'] = initial_value
        
        edges = nx.edge_bfs(self.G, source=0)
        
        for edge in edges:
            node_start = self.G.nodes[edge[0]]
            node_end = self.G.nodes[edge[1]]
            node_end['value'] = node_end['func'](node_start['value'])
            
    def export(self, filename):
        
        G_copy = copy.deepcopy(self.G)
        
        for node_id in G_copy.nodes:
            node = G_copy.nodes[node_id]
            node['x_coord'] = node['coord'][0]
            node['y_coord'] = node['coord'][1]
            node['value'] = round(node['value'], 3)
            del node['coord']
            del node['func']
            del node['inv_func']
            
        nx.write_gexf(G_copy, filename)
            
def generate_graph(steps, initial_value, func_map, func_bases, filename):
    
    free_group = CayleyGraph(steps, func_map, func_bases)
    free_group.act_on(float(initial_value))
    free_group.export(filename + '.gexf')
    
    return free_group
           