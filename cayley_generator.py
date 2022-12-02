import copy


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def line(a, p):
    
    def line_func(x):
        return a*x + (1-a)*p
    
    return line_func

def line_inv(a, p):
    
    if a == 0:
        raise ValueError("Lines with a slope of 0 can not be inverted.")
        
    def line_func(x):
        return (1/a)*x - (1-a)/a * p

    return line_func

def polar_to_cart(coord):
    
    r = coord[0]
    theta = coord[1]
    
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    
    return np.array((x, y))

class LinearFunctionSet:
    
    def __init__(self, m_f, p_f, m_g, p_g):
        
        self.__initialize_functions__(m_f, p_f, m_g, p_g)
        self.__define_inverse_map__()
        self.__define_radial_basis__()
        self.__define_cartesian_basis__()
        
    def __initialize_functions__(self, m_f, p_f, m_g, p_g):
        
        self.f = line(m_f, p_f)
        self.f_inv = line_inv(m_f, p_f)
        self.g = line(m_g, p_g)
        self.g_inv = line_inv(m_g, p_g)
    
    def __define_inverse_map__(self):
        
        self.func_map = {self.f     : self.f_inv,
                        self.f_inv : self.f,
                        self.g     : self.g_inv,
                        self.g_inv : self.g}
        
    def __define_radial_basis__(self):
        
        e_r = np.array((1.0, 0.0))
        e_t = np.array((0.0, 1.0))
        
        self.radial_basis = {self.f     : e_r + e_t,
                             self.f_inv : e_r - e_t,
                             self.g     : -e_r + e_t,
                             self.g_inv : -e_r - e_t}
        
    def __define_cartesian_basis__(self):
        
        e_x = np.array((1.0, 0.0))
        e_y = np.array((0.0, 1.0))
        
        self.cartesian_basis = {self.f     : e_x,
                                self.f_inv : -e_x,
                                self.g     : e_y,
                                self.g_inv : -e_y}


class AffineActionGraph:
    """ Generates a graph of an affine free group over some domain.
    """
    
    def __init__(self, max_steps, function_set, domain):
    
        self.funcs = function_set.func_map.keys()
        self.G = nx.Graph()
        self.max_steps = max_steps
        
        for value in domain:
            self.iterate(0, value)

    def iterate(self, step, value):
        
        if step >= self.max_steps:
            return
        
        for func in self.funcs:
            new_value = func(value)
            if is_integer(new_value):
                self.G.add_edge(value, new_value)
                self.iterate(step + 1, new_value)
        
class CayleyGraph:
    """Defines a cayley graph of a finitely generated free group of two functions and allows
    you to act on the graph by supplying a initial value.
    """
    
    def __init__(self, steps, func_dict, func_bases, rad_bases, scale = None):
        
        if scale is None:
            scale = 5*2**(steps+1)
        
        # Setup functions to be used
        self.identity = lambda x: x
        self.func_dict = func_dict
        self.func_bases = func_bases
        self.rad_bases = rad_bases
        self.coord_diff = lambda i: scale/(2**i)
        
        # Initialize iteration
        self.G = nx.Graph()
        
        origin = np.array((0.0, 0.0))
        initial_node = self.create_node(0, self.identity, self.identity, origin, origin)
        
        self.G.add_node(initial_node[0], **initial_node[1])
        self.prev_nodes = [initial_node]
        self.next_id = 1
        
        for step in range(steps):
            self.iterate(step)
    
    def create_node(self, node_id, func, inv_func, rad_vec, cart_vec, value=None):
        
            node_data = {'func': func,
                         'inv_func': inv_func,
                         'polar_coord': rad_vec,
                         'cart_coord': cart_vec,
                         'color': 'white',
                         'value': value}
            
            return (node_id, node_data)
    
    def create_new_node(self, prev_node, new_id, new_func, step):
        
        # Calculate the new euclidean coordinates of the node
        new_polar_coord = self.new_polar_coord(prev_node, new_func, step)
        new_cart_coord = self.new_cartesian_coord(prev_node, new_func, step)
        
        # Create new node data and add it to the graph
        new_node = self.create_node(self.next_id,
                                    new_func,
                                    self.func_dict[new_func],
                                    new_polar_coord,
                                    new_cart_coord)
        
        return new_node

    def new_polar_coord(self, prev_node, new_func, step):
        
        prev_coord = prev_node[1]['polar_coord']
        
        
        if prev_node[1]['func'] == new_func:
            new_coord = prev_coord + np.array((1.0, 0.0))
        elif step == 0:
            basis = self.rad_bases[new_func]
            scale = np.pi/(4*3**step)
            theta = basis[1] * scale + np.pi * min(-basis[0], 0)
            new_coord = np.array((1, theta))
        else:
            basis = self.rad_bases[new_func]
            scale = 2*np.pi/(4*3**step)
            new_coord = prev_coord + (1, basis[1]*(scale))
            
        return new_coord
    
    def new_cartesian_coord(self, prev_node, new_func, step):
        
        current_pos = prev_node[1]['cart_coord']
        delta_pos = self.coord_diff(step)*self.func_bases[new_func]
        new_coord = current_pos + delta_pos
        
        return new_coord
        
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
    
    def assign_property_by(self, prop_name, func):
        """Iterate through all nodes and assign some property to each node
        using the given function.
        """
        
        for node in self.G.nodes:
            self.G.nodes[node][prop_name] = func(node)
    
    def count_nodes_by_condition(self, counting_func):
        
        count = 0
        for node in self.G.nodes:
            if counting_func(self.G.nodes[node]):
                count += 1
        
        return count
    
    def draw(self, layout='Cartesian', with_labels=False, ax=None):

        if layout == 'Cartesian':
            coord_type = 'cart_coord'
        elif layout == 'Bethe':
            coord_type = 'polar_coord'
        else:
            raise ValueError(f"Layout type {layout} does not exist.")
        
        if ax is None:
            fig = plt.figure(figsize=(20, 20), dpi=200)
            
        nodes = self.G.nodes
        
        if with_labels:
            labels = dict([(node, str(nodes[node]['value'])) for node in nodes])
        else:
            labels = dict()
            
        if layout =='Bethe':
            pos = [polar_to_cart(nodes[node][coord_type]) for node in nodes]
        else:
            pos = [nodes[node][coord_type] for node in nodes]
            
        color = [nodes[node]['color'] for node in self.G.nodes]
        
        nx.draw_networkx(self.G, pos=pos, node_size=3000, node_color=color,
                         with_labels=with_labels, labels=labels, ax=ax)
    
    def prop_data_as_list(self, prop_name):
        props = [n_prop for n, n_prop in self.G.nodes.data(prop_name)]
        return props
    
    def export(self, filename):
        
        G_copy = copy.deepcopy(self.G)
        
        for node_id in G_copy.nodes:
            node = G_copy.nodes[node_id]
            node['x_coord'] = node['cart_coord'][0]
            node['y_coord'] = node['cart_coord'][1]
            node['value'] = round(node['value'], 3)
            del node['cart_coord']
            del node['func']
            del node['inv_func']
            
        nx.write_gexf(G_copy, filename)

class CayleyAnimator:
    """Handles animating a free group when given a function which defines the
    free group over an index.
    
    """
    
    def __init__(self, cayley_graph_func, frames):

        self.graph_func = cayley_graph_func
        self.frames = frames    
    
    def animate(self):
        
        fig = plt.figure(1, figsize=(15, 15), dpi=200)
        ax = plt.axes()
        
        def anim_iteration(i):
            ax.clear()
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            
            fg = self.graph_func(i, ax)
            fg.draw(ax)
            
            return [fig]
        
        anim = animation.FuncAnimation(fig, anim_iteration, frames=self.frames,
                                            interval=300, blit=True)
        
        anim.save('cayley.gif', writer='imagemagick')

def is_integer(x):
    return int(x) == x

def generate_graph(steps, initial_value, func_map, func_bases, filename):
    
    free_group = CayleyGraph(steps, func_map, func_bases)
    free_group.act_on(float(initial_value))
    free_group.export(filename + '.gexf')
    
    return free_group
           