import networkx as nx
import matplotlib.pyplot as plt

def visualize_beam_search_tree(tokens, beam_width=2, tree_depth=3):
    G = nx.Graph()
    
    def add_node(G, node_id, label):
        G.add_node(node_id, label=label)
    
    def add_edge(G, parent_id, child_id):
        G.add_edge(parent_id, child_id)
    
    # Add root node
    root_id = 0
    add_node(G, root_id, ' '.join(tokens[:3]) + '...')
    
    node_counter = 1
    current_level = [root_id]
    
    for depth in range(tree_depth):
        next_level = []
        for parent_id in current_level:
            for _ in range(beam_width):
                child_id = node_counter
                node_counter += 1
                
                # Simulate a substitution (replace a random token)
                child_tokens = tokens.copy()
                child_tokens[depth + 1] = f'SUB_{depth}_{child_id}'
                
                child_label = ' '.join(child_tokens[:3]) + '...'
                add_node(G, child_id, child_label)
                add_edge(G, parent_id, child_id)
                next_level.append(child_id)
        
        current_level = next_level
    
    # Visualization
    pos = nx.spring_layout(G, k=0.9, iterations=50)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=3000, font_size=8)
    
    # Add labels to nodes
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title("Beam Search Tree Visualization")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
tokens = ['[CLS]', 'The', 'movie', 'was', 'great', '[SEP]']
visualize_beam_search_tree(tokens, beam_width=2, tree_depth=3)