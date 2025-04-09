import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('catan_board_generator.h5', compile=False)

model = load_model()

# Define tile positions and colors
positions = [
    (-1, 2), (0, 2), (1, 2),
    (-1.5, 1), (-0.5, 1), (0.5, 1), (1.5, 1),
    (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
    (-1.5, -1), (-0.5, -1), (0.5, -1), (1.5, -1),
    (-1, -2), (0, -2), (1, -2)
]

COLORS = {
    'brick': '#BF5700',
    'wood': '#009900',
    'ore': '#808080',
    'sheep': '#99FF99',
    'wheat': '#FFFF66',
    'desert': '#D2B48C'
}

# Board generator
def generate_board(model):
    input_data = np.random.rand(1, 19)
    _ = model.predict(input_data)  # dummy prediction to seed randomness

    resources = (
        ['wood'] * 4 +
        ['brick'] * 3 +
        ['sheep'] * 4 +
        ['ore'] * 3 +
        ['wheat'] * 4 +
        ['desert'] * 1
    )
    random.shuffle(resources)

    # Ensure one desert
    desert_count = resources.count('desert')
    if desert_count > 1:
        desert_indices = [i for i, r in enumerate(resources) if r == 'desert']
        for idx in desert_indices[1:]:
            resources[idx] = random.choice(['wheat', 'wood', 'sheep', 'brick', 'ore'])
    elif desert_count == 0:
        non_desert_tiles = [i for i, r in enumerate(resources) if r != 'desert']
        resources[random.choice(non_desert_tiles)] = 'desert'

    numbers = [2, 3, 3, 4, 4, 5, 5, 6, 6,
               8, 8, 9, 9, 10, 10, 11, 11, 12]
    random.shuffle(numbers)

    final_board = []
    for resource in resources:
        if resource == 'desert':
            final_board.append((resource, 0))
        else:
            final_board.append((resource, numbers.pop() if numbers else 0))

    return final_board

# Balanced scoring
def rate_board(final_board):
    score = 50

    desert_index = next(i for i, tile in enumerate(final_board) if tile[0] == 'desert')
    desert_pos = positions[desert_index]

    six_eight_indices = [i for i, tile in enumerate(final_board) if tile[1] in [6, 8]]
    res_on_6_8 = [final_board[i][0] for i in six_eight_indices]

    if desert_pos == (0, 0):
        score -= 10

    for idx in six_eight_indices:
        dx = abs(positions[idx][0] - desert_pos[0])
        dy = abs(positions[idx][1] - desert_pos[1])
        if dx <= 1.1 and dy <= 1.1:
            score -= 6

    repeat_count = len(res_on_6_8) - len(set(res_on_6_8))
    score -= repeat_count * 4

    for i in six_eight_indices:
        for j in six_eight_indices:
            if i != j:
                dx = abs(positions[i][0] - positions[j][0])
                dy = abs(positions[i][1] - positions[j][1])
                if dx <= 1.1 and dy <= 1.1:
                    score -= 5

    if len(res_on_6_8) == len(set(res_on_6_8)) and res_on_6_8:
        score += 12

    for r in res_on_6_8:
        if r in ['wheat', 'wood']:
            score += 3

    all_far = True
    for i in six_eight_indices:
        for j in six_eight_indices:
            if i != j:
                dx = abs(positions[i][0] - positions[j][0])
                dy = abs(positions[i][1] - positions[j][1])
                if dx <= 1.1 and dy <= 1.1:
                    all_far = False
    if all_far and len(six_eight_indices) >= 2:
        score += 8

    return int(max(0, min(score, 100)))

# Visualizer
def visualize_board(final_board, score):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    for i, (x, y) in enumerate(positions):
        resource, number = final_board[i]
        color = COLORS[resource]
        hexagon = patches.RegularPolygon((x, y), numVertices=6, radius=0.5, orientation=np.radians(30),
                                         edgecolor='black', facecolor=color)
        ax.add_patch(hexagon)
        if resource != 'desert' and number != 0:
            color_txt = 'red' if number in [6, 8] else 'black'
            ax.text(x, y, str(number), ha='center', va='center', fontsize=12, color=color_txt)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    plt.title(f"Board Score: {score}", fontsize=14)
    return fig

# Streamlit UI
st.title("🌲 Catan Board Generator & Scorer")
st.markdown("Generate random Settlers of Catan boards and see how well they score based on tile quality.")

if st.button("🎲 Generate Boards"):
    st.info("Generating 100 boards... Finding top 5!")

    top_boards = []

    for _ in range(100):
        board = generate_board(model)
        score = rate_board(board)
        top_boards.append((score, board))

    top_boards.sort(reverse=True, key=lambda x: x[0])

    for i in range(min(5, len(top_boards))):
        score, board = top_boards[i]
        fig = visualize_board(board, score)
        st.pyplot(fig)
        st.markdown(f"**Board #{i+1} — Score: {score}**")
