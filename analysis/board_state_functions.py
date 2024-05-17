#%%##########################
# Creating a dataset evaluating every character/move in a PGN string with board state features.
###########################

from collections import defaultdict
import numpy as np
from tqdm import tqdm, trange
from IPython.display import SVG, display


import chess
import chess.svg
import datasets

# Load datset
dataset = datasets.load_dataset("adamkarvonen/chess_sae_individual_games_filtered")


def test_evaluator_func(dataset, func, kwargs={}, verbose=False):
    print('testing', func.__name__)
    true_count = 0
    max_tries = int(1e4)
    for i in trange(max_tries):
        random_index = np.random.randint(0, len(dataset))
        game_pgn = dataset[random_index]['text']
        board = chess.Board()
        for m in game_pgn.split():
            move_str = m
            # Advance the board
            if "." in m:
                m = m.split(".")[1]
            try:
                board.push_san(m)
                if verbose:
                    print(board)
                    print("-"*50)
            except (chess.InvalidMoveError, chess.IllegalMoveError) as e:
                break
            func_output = func(board, **kwargs)
            if func_output is True:
                print(f'true_count: {true_count}')
                print(f'func: {func.__name__}, value: {func_output}')
                print(f'kwargs: {kwargs}')
                rec_move_color = 'white' if board.turn == chess.BLACK else 'black'
                print(f'color of last move: {rec_move_color}')
                print(f'most recent move: {move_str}')
                # print(board)
                svg = chess.svg.board(board)
                display(SVG(svg))
                print('_'*50)
                true_count += 1
                break
        if true_count >=10:
            break

# Define evaluator functions
def board_to_check_int(board: chess.Board):
    """Return the check state of the board. 0 if not in check, 1 if white is in check, -1 if black is in check."""
    if board.is_check():
        return 1 if board.turn == chess.WHITE else -1
    return 0

def board_to_color(board: chess.Board):
    """Return the color of the move which has been played most recently. 0 if white, 1 if black. (board.turn returns the color of the next move to be played.)"""
    return "w" if board.turn == chess.BLACK else "b"

def check_fork(board, perspective, attacker_piece=chess.KNIGHT, target_pieces=[chess.ROOK, chess.QUEEN, chess.KING]):
    """Given a chess board object, return a 1x1 torch.Tensor.
    The 1x1 array indicates True if a piece is attacking at least two higher value pieces and is not pinned.
    Perspective can be 'mine' or 'other' to specify the player to move, after the most recent move."""

    state = torch.zeros((1, 1), dtype=DEFAULT_DTYPE)
    
    # Determine the color of the knights to check based on the perspective
    if perspective == "mine":
        color = board.turn
    elif perspective == "other":
        color = not board.turn
    else:
        raise ValueError("Perspective must be 'mine' or 'other'")

    # Loop through all pieces to find the knights of the given color
    for square in board.pieces(attacker_piece, color):
        if board.is_pinned(color, square):
            # Skip if the knight is pinned
            continue

        attacks = board.attacks(square)
        high_value_targets = 0

        # Check each attack square to see if it's occupied by a high-value enemy piece
        for attack_square in attacks:
            attacked_piece = board.piece_at(attack_square)
            if attacked_piece and attacked_piece.color != color:
                if attacked_piece.piece_type in target_pieces:
                    high_value_targets += 1
        
    # Check if the knight is attacking at least two high-value pieces
    state[0, 0] = 1 if high_value_targets >= 2 else 0
    return state

def is_pawn_pinned_to_own_king(board, perspective):
    king_square = board.king(board.turn)
    pinned_pawn = False

    for square in chess.SQUARES:
        if board.is_pinned(board.turn, square):
            piece = board.piece_at(square)
            if piece is not None and piece.piece_type == chess.PAWN and board.color_at(square) == board.turn:
                if perspective == "mine":
                    if king_square in board.attacks(square):
                        pinned_pawn = True
                else:
                    if king_square in board.attacks(square):
                        pinned_pawn = True

    return pinned_pawn

#%%
# test_evaluator_func(
#     dataset['train'], 
#     check_fork, 
#     kwargs={'perspective': 'mine', 
#             'attacker_piece': chess.ROOK, 
#             'target_pieces': [chess.QUEEN, chess.KING]}, 
#     verbose=False
# )
test_evaluator_func(
    dataset['train'], 
    is_pawn_pinned_to_own_king, 
    kwargs={'perspective': 'mine'}, 
    verbose=False
)

#%%
board_evaluation_functions = {
    'color': board_to_color,
    'is_check': board_to_check_int,
}




# Wrapper for the board evaluation functions
def game_evaluator(
    game_dict: dict,
    board_evaluation_functions: dict,
    label_per_char: bool = True,
) -> dict:
    """Given a dict containing PGN and game stats, add entries for board state evaluations in place."""
    # Initialize chess board
    board = chess.Board()
    half_moves = game_dict['text'].strip(';').split() # text corresponds to the pgn string
    # TODO Add custom initialization function

    for i, m in enumerate(half_moves):
        if label_per_char is True: 
            # Label every character with the evaluation of the board state after the most recent completed move.
            # For example, if Nf3 checks, the char positions in the brackets <> will be labeled as check: "... Nf3+< 10.Kh1> ..."
            # Exception: Color and move_number correspond to the move which has been played most recently.
            len_move_chars = len(m) + 1 # +1 for the space/semicolon before the move
            if i == 0:
                game_dict['move_number'].extend([None] * len_move_chars)
                for name in board_evaluation_functions.keys():
                    game_dict[name].extend([None] * len_move_chars)
            else:
                move_number = (i-1)//2 + 1
                game_dict['move_number'].extend([move_number] * len_move_chars)
                for name, func in board_evaluation_functions.items():
                    game_dict[name].extend([func(board)] * len_move_chars)

        # Advance the board
        if "." in m:
            m = m.split(".")[1]
        try:
            board.push_san(m)
            # print(board)
            # print("-"*50)
        except chess.InvalidMoveError:
            break
        
        if label_per_char is False:
            # Single label per half move
            move_number = i//2 + 1
            game_dict['move_number'].append(move_number)
            game_dict['move_san'].append(m)
            for name, func in board_evaluation_functions.items():
                game_dict[name].append(func(board))


def create_boardstate_dataset(
    dataset: datasets.arrow_dataset.Dataset,
    num_games: int,
    board_evaluation_functions: dict,
    label_per_char: bool = True,
) -> dict:
    """
    Input: List of dicts each describing a game (containing a PGN string and stats that are constant per game).
    Output: Copy of input with additional board state evaluations per character/move.
    """
    eval_dataset = []
    for i in range(num_games):
        game_dict = defaultdict(list, dataset[i])
        game_evaluator(game_dict, board_evaluation_functions, label_per_char)
        eval_dataset.append(game_dict)
    return eval_dataset


#%%
import pandas as pd

dataset_with_features = create_boardstate_dataset(
    dataset['train'],
    num_games=10, 
    board_evaluation_functions=board_evaluation_functions, 
    label_per_char=False,
)

df = pd.DataFrame(dataset_with_features)
df.head()

# %%
