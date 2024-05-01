###########################
# Creating a dataset evaluating every character/move in a PGN string with board state features.
###########################

from collections import defaultdict

import chess
import datasets


# Define evaluator functions
def board_to_check_int(board: chess.Board):
    """Return the check state of the board. 0 if not in check, 1 if white is in check, -1 if black is in check."""
    if board.is_check():
        return 1 if board.turn == chess.WHITE else -1
    return 0

def board_to_color(board: chess.Board):
    """Return the color of the move which has been played most recently. 0 if white, 1 if black. (board.turn returns the color of the next move to be played.)"""
    return "w" if board.turn == chess.BLACK else "b"

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
            print(board)
            print("-"*50)
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


if __name__ == "__main__":

    dataset = datasets.load_dataset("adamkarvonen/chess_sae_individual_games_filtered")

    dataset_with_features = create_boardstate_dataset(
        dataset['train'],
        num_games=10, 
        board_evaluation_functions=board_evaluation_functions, 
        label_per_char=True,
    )

    # Inspect result
    res = dataset_with_features[-1]
    inspect_keys = ['text'] + [name for name in res.keys() if isinstance(res[name], list)]

    print(inspect_keys)
    for i in range(len(res['move_number'])):
        i_list = [res[name][i] for name in inspect_keys]
        print(i_list)
