import matplotlib.pyplot as plt

# Posisi node sesuai permintaan (dirotasi)
nodes = {
    'A': (-0.5, -1),  # F menjadi A
    'B': (-1, 0),     # A menjadi B
    'C': (-0.5, 1),   # B menjadi C
    'D': (0.5, 1),    # C menjadi D
    'E': (1, 0),      # D menjadi E
    'F': (0.5, -1),   # E menjadi F
    'G': (0, 0)       # F menjadi G
}

# Urutan langkah (edges sesuai permintaan)
steps = [
    [],  # Step 0 (tidak ada edge)
    [('A', 'B')],  # Step 1
    [('A', 'B'), ('B', 'C')],  # Step 2
    [('A', 'B'), ('B', 'C'), ('C', 'D')],  # Step 3
    [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')],  # Step 4
    [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F')],  # Step 5
    [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G')],  # Step 6
    [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G'), ('G', 'C')],  # Step 7
    [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G'), ('G', 'C'), ('C', 'A')],  # Step 8
    [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G'), ('G', 'C'), ('C', 'A'), ('A', 'G')],  # Step 9
    [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G'), ('G', 'C'), ('C', 'A'), ('A', 'G'), ('G', 'D')],  # Step 10
    [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G'), ('G', 'C'), ('C', 'A'), ('A', 'G'), ('G', 'D'), ('D', 'F')],  # Step 11
    [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G'), ('G', 'C'), ('C', 'A'), ('A', 'G'), ('G', 'D'), ('D', 'F'), ('F', 'A')]  # Step 12
]

def draw_step(step):
    """
    Menggambar graf hingga step tertentu.
    """
    plt.clf()
    plt.gca().set_aspect('equal')
    plt.axis([-1.5, 1.5, -1.5, 1.5])
    plt.axis('off')
    
    # Gambar node
    for node, pos in nodes.items():
        plt.plot(pos[0], pos[1], 'ko', markersize=15)  # Node
        plt.text(pos[0], pos[1] + 0.1, node, ha='center', fontsize=12)  # Label node
    
    # Gambar edges hingga step saat ini
    if step < len(steps):
        for edge in steps[step]:
            start, end = edge
            start_pos = nodes[start]
            end_pos = nodes[end]
            plt.plot([start_pos[0], end_pos[0]], 
                     [start_pos[1], end_pos[1]], 
                     'b-', linewidth=2)  # Edge warna biru
    
    plt.title(f'Step {step}', fontsize=16)

def run_animation():
    """
    Jalankan animasi langkah-langkah graf.
    """
    fig = plt.figure(figsize=(8, 8))
    
    # Animasi setiap step
    for step in range(len(steps)):
        draw_step(step)
        plt.pause(1)  # Pause 1 detik setiap langkah
    
    plt.show()

if __name__ == "__main__":
    run_animation()