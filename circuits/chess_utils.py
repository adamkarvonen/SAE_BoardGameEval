import chess
import pandas as pd
import torch
from torch.nn import functional as F
from typing import Callable, Optional
from dataclasses import dataclass
from jaxtyping import Int, Float, jaxtyped
from beartype import beartype
from torch import Tensor
from enum import Enum
import re

import circuits.othello_utils as othello_utils

# Mapping of chess pieces to integers
PIECE_TO_INT = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}

INT_TO_PIECE = {value: key for key, value in PIECE_TO_INT.items()}
PIECE_TO_ONE_HOT_MAPPING = {
    -6: 0,
    -5: 1,
    -4: 2,
    -3: 3,
    -2: 4,
    -1: 5,
    0: 6,
    1: 7,
    2: 8,
    3: 9,
    4: 10,
    5: 11,
    6: 12,
}
BLANK_INDEX = PIECE_TO_ONE_HOT_MAPPING[0]
ONE_HOT_TO_PIECE_MAPPING = {value: key for key, value in PIECE_TO_ONE_HOT_MAPPING.items()}

DEFAULT_DTYPE = torch.int8


class PlayerColor(Enum):
    WHITE = "White"
    BLACK = "Black"


def board_to_skill_state(board: chess.Board, skill: float) -> torch.Tensor:
    """Given a chess board object, return a 1x1 torch.Tensor.
    The 1x1 array should tell what skill level the player is."""
    state = torch.zeros((1, 1), dtype=DEFAULT_DTYPE)
    state[0][0] = skill

    return state


# import chess.engine

# stockfish_path = "/usr/games/stockfish"
# engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)


def board_to_eval_state(board: chess.Board, skill: Optional[int] = None) -> torch.Tensor:
    """Given a chess board object, return a 1x1 torch.Tensor.
    The 1x1 array should tell which player is winning.
    -1 = Black has > 100 centipawns advantage, 0 = Draw, 1 = White has > 100 centipawns advantage.
    This is horribly inefficient and takes ~0.75 seconds per game. However, I'm just doing exploratory analysis.
    If we wanted efficiency, we could use a bunch of parallel CPU workers to evaluate the board state and store it
    in a lookup table. But, then we couldn't cleanly use this with the existing abstractions.
    To use this function, uncomment the import chess.engine through engine = above, and the internal code below.
    """
    state = torch.zeros((1, 1), dtype=DEFAULT_DTYPE)

    # info = engine.analyse(board, chess.engine.Limit(time=0.01))
    # score = info["score"].white().score(mate_score=10000)

    # # Modify player_one_score based on the score
    # if score < 100:
    #     state[0][0] = -1
    # elif score > 100:
    #     state[0][0] = 1
    # else:
    #     state[0][0] = 0

    return state


def board_to_piece_color_state(board: chess.Board, skill: Optional[int] = None) -> torch.Tensor:
    """Given a chess board object, return a 8x8 torch.Tensor.
    The 8x8 array should tell if each square is black, white, or blank.
    White is 1, black is -1, and blank is 0.
    In the 8x8 array, row 0 is A1-H1 (White), row 1 is A2-H2, etc."""
    state = torch.zeros((8, 8), dtype=DEFAULT_DTYPE)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            # Assign 1 for white pieces and -1 for black pieces
            state[i // 8, i % 8] = 1 if piece.color == chess.WHITE else -1

    return state


def board_to_piece_state(board: chess.Board, skill: Optional[int] = None) -> torch.Tensor:
    """Given a chess board object, return an 8x8 torch.Tensor.
    The 8x8 array should tell what piece is on each square. A white pawn could be 1, a black pawn could be -1, etc.
    Blank squares should be 0.
    In the 8x8 array, row 0 is A1-H1 (White), row 1 is A2-H2, etc."""

    # Because state is initialized to all 0s, we only need to change the values of the pieces
    state = torch.zeros((8, 8), dtype=DEFAULT_DTYPE)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            piece_value = PIECE_TO_INT[piece.piece_type]
            # Multiply by -1 if the piece is black
            if piece.color == chess.BLACK:
                piece_value *= -1
            state[i // 8, i % 8] = piece_value

    return state


def board_to_threat_state(board: chess.Board, skill: Optional[int] = None) -> torch.Tensor:
    """Given a chess board object, return an 8x8 torch.Tensor.
    The 8x8 array should tell if each square is being attacked by the opponent."""

    ATTACKING_COLOR = chess.BLACK
    # Because state is initialized to all 0s, we only need to change the values of the pieces
    state = torch.zeros((8, 8), dtype=DEFAULT_DTYPE)
    for i in range(64):
        if board.is_attacked_by(ATTACKING_COLOR, i):
            state[i // 8, i % 8] = 1

    return state


def board_to_check_state(board: chess.Board, skill: Optional[int] = None) -> torch.Tensor:
    """Given a chess board object, return a 1x1 torch.Tensor.
    The 1x1 array should tell if the current player is in check.
    1 = In check, 0 = Not in check."""
    state = torch.zeros((1, 1), dtype=DEFAULT_DTYPE)
    state[0][0] = 1 if board.is_check() else 0

    return state


def board_to_pin_state(board: chess.Board, skill: Optional[int] = None) -> torch.Tensor:
    """Given a chess board object, return a 1x1 torch.Tensor.
    The 1x1 array indicates if there are any pins on the board (1 = yes, 0 = no)."""

    state = torch.zeros((1, 1), dtype=DEFAULT_DTYPE)

    for color in [chess.WHITE, chess.BLACK]:
        for i in range(64):
            piece = board.piece_at(i)
            if piece and piece.color == color:
                if board.is_pinned(color, i):
                    state[0, 0] = 1
                    return state

    return state


def board_to_prev_state(board: chess.Board, skill: Optional[int] = None) -> torch.Tensor:
    """Given a chess board object, return an 8x8 torch.Tensor.
    The 8x8 array should tell what piece is on each square at a previous board state."""

    PREVIOUS_TURNS = 25
    state = torch.zeros((8, 8), dtype=DEFAULT_DTYPE)

    # If we cannot roll back PREVIOUS_TURNS, return a blank state
    # Predicting blank states is trivial, so be careful and change pos_start to not index into the blank states
    if len(board.move_stack) < PREVIOUS_TURNS:
        return state

    new_board = board.copy()

    for _ in range(PREVIOUS_TURNS):
        new_board.pop()

    for i in range(64):
        piece = new_board.piece_at(i)
        if piece:
            piece_value = PIECE_TO_INT[piece.piece_type]
            # Multiply by -1 if the piece is black
            if piece.color == chess.BLACK:
                piece_value *= -1
            state[i // 8, i % 8] = piece_value

    return state


def board_to_legal_moves_state(board: chess.Board, skill: Optional[int] = None) -> torch.Tensor:
    """Return an 8x8 torch.Tensor indicating squares where White has legal moves.

    Each square in the array is 1 if White can legally move a piece to that square, otherwise 0.
    In the 8x8 array, row 0 corresponds to A1-H1 (from White's perspective), row 1 to A2-H2, etc.
    """
    MOVING_COLOR = chess.WHITE
    # Initialize the state array with all zeros
    state = torch.zeros((8, 8), dtype=DEFAULT_DTYPE)

    # Iterate through all legal moves for White
    for move in board.legal_moves:
        # Check if the move is for a White piece
        if board.color_at(move.from_square) == MOVING_COLOR:
            # Update the state array for the destination square of the move
            to_square = move.to_square
            state[to_square // 8, to_square % 8] = 1

    return state


def board_to_last_self_move_state(board: chess.Board, skill: Optional[int] = None) -> torch.Tensor:
    """Given a chess board object, return an 8x8 torch.Tensor.
    All squares will be 0 except for the square where the last white move was made.
    In the 8x8 array, row 0 is A1-H1 (White), row 1 is A2-H2, etc.
    The purpose of this is to see if the linear probe can determine the next move of the GPT.
    To get next move instead of last move, we offset the state stack by 1 in linear_probe_forward_pass():
    resid_post = resid_post[:, :-1, :]
    state_stack_one_hot = state_stack_one_hot[:, 1:, :, :, :]
    """

    state = torch.zeros((8, 8), dtype=DEFAULT_DTYPE)

    # If offset is 2, we are predicting the LLM's next move
    # If offset is 1, we are predicting the opponent's response to the LLM's next move
    offset = 1

    # If there is no last move (such as beginning of game), return the state as is
    if len(board.move_stack) < offset:
        return state

    last_last_move = board.move_stack[-offset]
    destination_square = last_last_move.to_square
    moved_piece = board.piece_at(destination_square)
    if moved_piece is None:
        raise ValueError("Piece type is None")
    piece_value = PIECE_TO_INT[moved_piece.piece_type]
    if moved_piece.color == chess.BLACK:
        piece_value *= -1
    state[destination_square // 8, destination_square % 8] = piece_value

    return state


@dataclass
class Config:
    min_val: int
    max_val: int
    custom_board_state_function: Callable
    num_rows: int = 8
    num_cols: int = 8


piece_config = Config(
    min_val=-6,
    max_val=6,
    custom_board_state_function=board_to_piece_state,
)

color_config = Config(
    min_val=-1,
    max_val=1,
    custom_board_state_function=board_to_piece_color_state,
)

threat_config = Config(
    min_val=0,
    max_val=1,
    custom_board_state_function=board_to_threat_state,
)

legal_move_config = Config(
    min_val=0,
    max_val=1,
    custom_board_state_function=board_to_legal_moves_state,
)

prev_move_config = Config(
    min_val=-6,
    max_val=6,
    custom_board_state_function=board_to_prev_state,
)


eval_config = Config(
    min_val=-1,
    max_val=1,
    custom_board_state_function=board_to_eval_state,
    num_rows=1,
    num_cols=1,
)

skill_config = Config(
    min_val=-2,
    max_val=20,
    custom_board_state_function=board_to_skill_state,
    num_rows=1,
    num_cols=1,
)

check_config = Config(
    min_val=0,
    max_val=1,
    custom_board_state_function=board_to_check_state,
    num_rows=1,
    num_cols=1,
)

pin_config = Config(
    min_val=0,
    max_val=1,
    custom_board_state_function=board_to_pin_state,
    num_rows=1,
    num_cols=1,
)

# Kind of janky... TODO
othello_config = Config(
    min_val=-1,
    max_val=1,
    custom_board_state_function=othello_utils.games_batch_to_state_stack_BLRRC,
    num_rows=8,
    num_cols=8,
)

othello_mine_yours_config = Config(
    min_val=-1,
    max_val=1,
    custom_board_state_function=othello_utils.games_batch_to_state_stack_mine_yours_BLRRC,
    num_rows=8,
    num_cols=8,
)

othello_no_last_move_config = Config(
    min_val=-1,
    max_val=1,
    custom_board_state_function=othello_utils.games_batch_no_last_move_to_state_stack_BLRRC,
    num_rows=8,
    num_cols=8,
)

all_configs = [
    piece_config,
    color_config,
    threat_config,
    legal_move_config,
    prev_move_config,
    eval_config,
    skill_config,
    check_config,
    pin_config,
    othello_config,
    othello_mine_yours_config,
    othello_no_last_move_config,
]

config_lookup = {config.custom_board_state_function.__name__: config for config in all_configs}


def get_num_classes(config: Config) -> int:
    num_classes = abs(config.min_val) + abs(config.max_val) + 1
    if num_classes == 2:
        num_classes = 1
    return num_classes


def state_stack_to_chess_board(state: torch.Tensor) -> chess.Board:
    """Given a state stack, return a chess.Board object.
    WARNING: The board will not include any information about whose turn it is, castling rights, en passant, etc.
    For this reason, pgn_string_to_board is preferred."""
    board = chess.Board(fen=None)
    for row_idx, row in enumerate(state):
        for col_idx, piece in enumerate(row):
            if piece != 0:
                piece_type = abs(piece)
                color = chess.WHITE if piece > 0 else chess.BLACK
                board.set_piece_at(chess.square(col_idx, row_idx), chess.Piece(piece_type, color))
    return board


def pgn_string_to_board(pgn_string: str, allow_exception: bool = False) -> chess.Board:
    """Convert a PGN string to a chess.Board object.
    We are making an assumption that the PGN string is in this format:
    ;1.e4 e5 2. or ;1.e4 e5 2.Nf3"""
    board = chess.Board()
    for move in pgn_string.split():
        if "." in move:
            move = move.split(".")[1]
        if move == "":
            continue
        try:
            board.push_san(move)
        except:
            if allow_exception:
                break
            else:
                raise ValueError(f"Invalid move: {move}")
    return board


def create_state_stack(
    moves_string: str,
    custom_board_to_state_fns: list[Callable[[chess.Board], torch.Tensor]],
    device: torch.device,
    skill: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Given a string of PGN format moves, create a torch.Tensor for every character in the string.
    Return a dict of func_name to state_stack."""

    board = chess.Board()
    initial_states = {}
    expanded_states = {}
    count = 1

    for custom_fn in custom_board_to_state_fns:
        func_name = custom_fn.__name__
        initial_states[func_name] = []
        expanded_states[func_name] = []
        # Scan 1: Creates states, with length = number of moves in the game
        initial_states[func_name].append(custom_fn(board, skill).to(dtype=DEFAULT_DTYPE))
    # Apply each move to the board
    for move in moves_string.split():
        try:
            count += 1
            # Skip move numbers
            if "." in move:
                board.push_san(move.split(".")[1])
            else:
                board.push_san(move)

            for custom_fn in custom_board_to_state_fns:
                func_name = custom_fn.__name__
                initial_states[func_name].append(custom_fn(board, skill).to(dtype=DEFAULT_DTYPE))
        except:
            # because all games are truncated to a length, often the last move is partial and invalid
            # so we don't need to log this, as it will happen on most games
            break

    # if count % 100 == 0:
    #     pretty_print_state_stack(state)
    #     print("_" * 50)
    #     print(board)

    # Second Scan: Expand states to match the length of moves_string
    # For ;1.e4 e5 2.Nf3, ";1.e4" = idx 0, " e5" = idx 1, " 2.Nf3" = idx 2
    move_index = 0
    for char in moves_string:
        if char == " ":
            move_index += 1
        for func_name in initial_states:
            expanded_states[func_name].append(
                initial_states[func_name][min(move_index, len(initial_states[func_name]) - 1)]
            )

    for func_name in initial_states:
        expanded_states[func_name] = torch.stack(expanded_states[func_name]).to(device=device)
    return expanded_states


def create_state_stacks(
    moves_strings: list[str],
    custom_board_to_state_fns: list[Callable[[chess.Board], torch.Tensor]],
    device: torch.device,
    skill_array: Optional[torch.Tensor] = None,
) -> dict[str, Float[Tensor, "sample_size pgn_str_length rows cols"]]:
    """Given a list of strings of PGN format moves, create a dict of func name to tensor.
    custom_board_to_state is a function that takes a chess.Board object and returns a 8x8 torch.Tensor for
    board state, or 1x1 for centipawn advantage."""
    state_stacks = {}
    skill = None

    for custom_fn in custom_board_to_state_fns:
        func_name = custom_fn.__name__
        state_stacks[func_name] = []

    for idx, pgn_string in enumerate(moves_strings):
        if skill_array is not None:
            skill = skill_array[idx]
        state_stack_dict = create_state_stack(pgn_string, custom_board_to_state_fns, device, skill)

        for func_name in state_stack_dict:
            state_stacks[func_name].append(state_stack_dict[func_name])

    for func_name in state_stacks:
        # Convert the list of tensors to a single tensor
        state_stacks[func_name] = torch.stack(state_stacks[func_name])
    return state_stacks


def state_stack_to_one_hot(
    config: Config,
    device: torch.device,
    state_stack: torch.Tensor,
    user_mapping: Optional[dict[int, int]] = None,  # Only used for skill mapping
) -> Int[Tensor, "sample_size num_white_moves rows cols one_hot_range"]:
    """Input shape: assert(state_stacks_all_chars.shape) == (sample_size, game_length, rows, cols)
    Output shape: assert(state_stacks_one_hot.shape) == (sample_size, game_length, rows, cols, one_hot_range)
    """
    range_size = get_num_classes(config)

    # This will return binary values as scalar, not one-hot
    if range_size <= 2:
        return state_stack.unsqueeze(-1)

    mapping = {}
    if user_mapping:
        mapping = user_mapping
        min_val = min(mapping.values())
        max_val = max(mapping.values())
        range_size = max_val - min_val + 1
    else:
        for val in range(config.min_val, config.max_val + 1):
            mapping[val] = val - config.min_val

    # Initialize the one-hot tensor
    one_hot = torch.zeros(
        state_stack.shape[0],  # num games
        state_stack.shape[1],  # num moves
        config.num_rows,
        config.num_cols,
        range_size,
        device=device,
        dtype=DEFAULT_DTYPE,
    )

    for val in mapping:
        one_hot[..., mapping[val]] = state_stack == val

    return one_hot


def one_hot_to_state_stack(one_hot: torch.Tensor, min_val: int) -> torch.Tensor:
    """Input shape: assert(probe_out.shape) == (sample_size, num_white_moves, rows, cols, one_hot_range)
    Output shape: assert(state_stacks_probe_outputs.shape) == (sample_size, num_white_moves, rows, cols)
    """
    indices = torch.argmax(one_hot, dim=-1)
    state_stack = indices + min_val
    return state_stack


def square_to_coordinate(square: chess.Square) -> tuple[int, int]:
    row = chess.square_rank(square)
    column = chess.square_file(square)
    return (row, column)


def find_dots_indices(moves_string: str) -> list[int]:
    """Returns a list of ints of indices of every '.' in the string.
    This will hopefully provide a reasonable starting point for training a linear probe.
    """
    indices = [index for index, char in enumerate(moves_string) if char == "."]
    return indices


def find_spaces_indices(moves_string: str) -> list[int]:
    """Returns a list of ints of indices of every ' ' in the string."""
    indices = [index for index, char in enumerate(moves_string) if char == " "]
    return indices


def get_othello_even_list_indices(tokens_list: list[int]) -> list[int]:
    """"""
    max_len = len(tokens_list)
    return [i for i in range(max_len) if i % 2 == 0]


def get_all_white_piece_prev_pos_indices(
    moves_string: str, board: chess.Board, move_san: chess.Move
) -> list[int]:
    white_pos_indices = get_all_white_pos_indices(moves_string)
    new_board = board.copy()
    count = count_turns_with_piece_at_square(new_board, move_san) // 2

    assert moves_string[-1] == ".", f"Last char in moves_string is {moves_string[-1]}"
    # Because e.g. " 3." has not been counted as a move, but is a sublist in white_pos_indices
    count += 1
    # Because we want to include the turn that the move was made on
    count += 1

    # Flatten the list of lists of ints into a single list of ints
    correct_indices = [idx for sublist in white_pos_indices[-count:] for idx in sublist]

    return correct_indices


def get_all_black_piece_prev_pos_indices(
    moves_string: str, board: chess.Board, move_san: chess.Move
) -> list[int]:
    black_pos_indices = get_all_black_pos_indices(moves_string)
    new_board = board.copy()
    count = (count_turns_with_piece_at_square(new_board, move_san) + 1) // 2

    assert moves_string[-1] == ".", f"Last char in moves_string is {moves_string[-1]}"

    # Flatten the list of lists of ints into a single list of ints
    correct_indices = [idx for sublist in black_pos_indices[-count:] for idx in sublist]

    return correct_indices


def count_turns_with_piece_at_square(board: chess.Board, move_san: chess.Move) -> int:
    source_square = move_san.from_square
    moved_piece = board.piece_at(source_square)
    count = 0
    for _ in range(len(board.move_stack)):
        board.pop()
        if board.piece_at(source_square) == moved_piece:
            count += 1
        else:
            break
    return count


def get_all_white_pos_indices(moves_string: str) -> list[list[int]]:
    """From this pgn string: ;1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Qxd4 a6 5.Bc4 Nc6 6.Qd1...
    Return a list of lists of indices that correspond to the chars in parentheses:
    (;1.e4)< c5>( 2.Nf3)< d6>( 3.d4)< cxd4>( 4.Qxd4)< a6>( 5.Bc4)< Nc6>( 6.Qd1)"""
    space_indices = find_spaces_indices(moves_string)
    white_move_indices: list[list[int]] = []
    start_index = 0

    if len(space_indices) == 0:
        return [list(range(0, len(moves_string)))]

    for i, space in enumerate(space_indices):
        if i % 2 == 1:
            start_index = space
            if i == len(space_indices) - 1:
                white_move_indices.append(list(range(start_index, len(moves_string))))
                break
            continue
        white_move_indices.append(list(range(start_index, space)))
    return white_move_indices


def get_all_black_pos_indices(moves_string: str) -> list[list[int]]:
    """From this pgn string: ;1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Qxd4 a6 5.Bc4 Nc6 6.Qd1...
    Return a list of lists of indices that correspond to the chars in brackets:
    (;1.e4)< c5>( 2.Nf3)< d6>( 3.d4)< cxd4>( 4.Qxd4)< a6>( 5.Bc4)< Nc6>( 6.Qd1)"""
    space_indices = find_spaces_indices(moves_string)
    black_move_indices: list[list[int]] = []

    if len(space_indices) == 0:
        return []

    start_index = space_indices[0]

    for i, space in enumerate(space_indices):
        if i % 2 == 0:
            start_index = space
            if i == len(space_indices) - 1:
                black_move_indices.append(list(range(start_index, len(moves_string))))
                break
            continue
        black_move_indices.append(list(range(start_index, space)))
    return black_move_indices


def find_num_indices(moves_string: str) -> list[int]:
    """Returns a list of ints of indices of every `counting number.` in the PGN string.
    In this case, it would be the characters in angle brackets: ;<1.>e4 e5 <2.>Nf3
    Example: ;1.e4 e5 2. would be [1,2,9,10]. nums in moves like e4 / e5 are not counted.
    """

    # Regex pattern to match move numbers (e.g., "1.", "2.") immediately following optional whitespace or start of string
    pattern = r"(?<=;|\s)(\d+)(\.|$)"

    indices = []

    for match in re.finditer(pattern, moves_string):
        start_index = match.start(1)  # Start index of the number itself
        end_index = match.end() - 1  # End index of the number before the dot
        if start_index == end_index:
            indices.extend([start_index])
        else:
            indices.extend([start_index, end_index])

    return indices


def find_odd_spaces_indices(moves_string: str) -> list[int]:
    """Returns a list of ints of odd indices of every ' ' in the string.
    There is some duplicated logic but it simplifies using the Callable function."""
    indices = [index for index, char in enumerate(moves_string) if char == " "]
    # Select only the odd indices: start from index 1, go till the end, step by 2
    odd_indices = indices[1::2]
    return odd_indices


def find_even_spaces_indices(moves_string: str) -> list[int]:
    """Returns a list of ints of even indices of every ' ' in the string.
    There is some duplicated logic but it simplifies using the Callable function."""
    indices = [index for index, char in enumerate(moves_string) if char == " "]
    # Select only the even indices: start from index 0, go till the end, step by 2
    even_indices = indices[::2]
    return even_indices


def find_dots_indices_offset_one(moves_string: str) -> list[int]:
    """Returns a list of ints of indices of every '.' in the string.
    This will hopefully provide a reasonable starting point for training a linear probe.
    """
    indices = [index for index, char in enumerate(moves_string) if char == "."]

    incremented_indices = [index + 1 for index in indices if index + 1 < len(moves_string)]

    return incremented_indices


def find_even_indices_offset_one(moves_string: str) -> list[int]:
    """
    Returns a list of ints of even indices of every ' ' in the string, each incremented by one.
    If the incremented index would be greater than the length of the string, it is not included.
    """
    indices = [index for index, char in enumerate(moves_string) if char == " "]
    even_indices = indices[::2]

    # Increment each even index by one, ensuring it doesn't exceed the string length
    incremented_indices = [index + 1 for index in even_indices if index + 1 < len(moves_string)]

    return incremented_indices


def find_odd_indices_offset_one(moves_string: str) -> list[int]:
    """
    Returns a list of ints of odd indices of every ' ' in the string, each incremented by one.
    If the incremented index would be greater than the length of the string, it is not included.
    """
    indices = [index for index, char in enumerate(moves_string) if char == " "]
    odd_indices = indices[1::2]

    # Increment each odd index by one, ensuring it doesn't exceed the string length
    incremented_indices = [index + 1 for index in odd_indices if index + 1 < len(moves_string)]

    return incremented_indices


def find_custom_indices(
    custom_indexing_fn: Callable[[str], list[int]], df: pd.DataFrame
) -> torch.Tensor:
    indices_series = df["transcript"].apply(custom_indexing_fn)
    shortest_length = indices_series.apply(len).min()
    print("Shortest length:", shortest_length)

    indices_series = indices_series.apply(lambda x: x[:shortest_length])
    assert all(
        len(lst) == shortest_length for lst in indices_series
    ), "Not all lists have the same length"

    indices = torch.tensor(indices_series.apply(list).tolist(), dtype=torch.int32)
    return indices


supported_indexing_functions = {
    find_dots_indices.__name__: find_dots_indices,
    get_othello_even_list_indices.__name__: get_othello_even_list_indices,
}


def encode_string(meta: dict, s: str) -> list[int]:
    """Encode a string into a list of integers."""
    stoi = meta["stoi"]
    return [stoi[c] for c in s]


def decode_list(meta: dict, l: list[int]) -> str:
    """Decode a list of integers into a string."""
    itos = meta["itos"]
    return "".join([itos[i] for i in l])


def chess_boards_to_state_stack(
    chess_boards: list[chess.Board],
    device: torch.device,
    config: Config,
    skill: Optional[torch.Tensor] = None,
) -> Int[Tensor, "batch_size num_rows num_cols num_options"]:
    one_hot_list = []

    for board in chess_boards:
        state_stack = config.custom_board_state_function(board, skill)
        state_stack = state_stack.view(1, 1, config.num_rows, config.num_cols)
        one_hot = state_stack_to_one_hot(config, device, state_stack)
        one_hot_list.append(one_hot)
    stacked_one_hot = torch.stack(one_hot_list, dim=0)
    return stacked_one_hot


def mask_initial_board_states(
    one_hot_list: Int[Tensor, "batch_size num_rows num_cols num_options"],
    device: torch.device,
    config: Config,
    skill: Optional[torch.Tensor] = None,
) -> Int[Tensor, "batch_size num_rows num_cols num_options"]:
    """Mask off all board states that are shared with the initial board state.
    Otherwise the initial board state will dominate the common states."""
    initial_board = chess.Board()
    initial_state = config.custom_board_state_function(initial_board, skill)
    initial_state = initial_state.view(1, 1, config.num_rows, config.num_cols)
    initial_one_hot = state_stack_to_one_hot(config, device, initial_state)

    mask = (initial_one_hot == 1) & (one_hot_list == 1)
    one_hot_list[mask] = 0
    return one_hot_list


def get_averaged_states(
    one_hot_stack: Int[Tensor, "batch_size num_rows num_cols num_options"]
) -> Int[Tensor, "num_rows num_cols num_options"]:
    summed_one_hot = torch.sum(one_hot_stack, dim=0)
    averaged_one_hot = summed_one_hot / one_hot_stack.shape[0]
    averaged_one_hot = averaged_one_hot
    return averaged_one_hot


def find_common_states(
    averaged_one_hot: Int[Tensor, "num_rows num_cols num_options"], threshold: float
) -> tuple[torch.Tensor, ...]:
    greater_than_threshold = averaged_one_hot >= threshold
    indices = torch.nonzero(greater_than_threshold, as_tuple=True)
    return indices
