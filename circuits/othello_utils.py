import torch as t
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset

from circuits.othello_engine_utils import (
    OthelloBoardState,
    stoi,
    itos,
    to_board_label,
    stoi_indices,
)


def hf_othello_dataset_to_generator(
    dataset_name="taufeeque/othellogpt", split="train", streaming=True, token_mapping=None
):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    def gen():
        for x in iter(dataset):
            tokens = x["tokens"]
            if token_mapping is not None:
                tokens = [token_mapping[token] for token in tokens]
            yield tokens

    return gen()


def board_state_to_RRC(board_state, flip: int = 1) -> t.Tensor:
    board_state = t.tensor(board_state, dtype=t.int8)
    board_state *= flip
    one_hot = t.zeros((8, 8, 3), dtype=t.int8)
    one_hot[..., 0] = (board_state == -1).int()
    one_hot[..., 1] = (board_state == 0).int()
    one_hot[..., 2] = (board_state == 1).int()
    return one_hot


# TODO Remove duplicated logic from these functions
def games_batch_to_state_stack_BLRRC(batch_str_moves: list[int]) -> t.Tensor:
    """Sequences of moves (dataset format) to state stack (one-hot) of shape (seq_len, 8, 8, 3)"""
    iterable = tqdm(batch_str_moves) if len(batch_str_moves) > 50 else batch_str_moves

    game_stack = []
    for game in iterable:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        for move in game:
            board.umpire(move)
            one_hot = board_state_to_RRC(board.state)
            states.append(one_hot)
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


def games_batch_to_valid_moves_BLRRC(batch_str_moves: list[int]) -> t.Tensor:
    """Sequences of moves (dataset format) to state stack of valid moves"""
    iterable = tqdm(batch_str_moves) if len(batch_str_moves) > 50 else batch_str_moves

    game_stack = []
    for game in iterable:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        for i, move in enumerate(game):
            moves_board = t.zeros(8, 8, 1, dtype=t.int8)
            board.umpire(move)
            valid_moves_list = board.get_valid_moves()
            for move in valid_moves_list:
                moves_board[move // 8, move % 8] = 1
            states.append(moves_board)
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


def games_batch_to_state_stack_mine_yours_BLRRC(batch_str_moves: list[int]) -> t.Tensor:
    """Sequences of moves (dataset format) to state stack (one-hot) of shape (seq_len, 8, 8, 3)"""
    iterable = tqdm(batch_str_moves) if len(batch_str_moves) > 50 else batch_str_moves

    game_stack = []
    for game in iterable:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        for i, move in enumerate(game):
            flip = 1
            if i % 2 == 1:
                flip = -1
            board.umpire(move)
            one_hot = board_state_to_RRC(board.state, flip)
            states.append(one_hot)
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


def games_batch_to_state_stack_mine_yours_blank_mask_BLRRC(batch_str_moves: list[int]) -> t.Tensor:
    """Sequences of moves (dataset format) to state stack (one-hot) of shape (seq_len, 8, 8, 3)"""
    iterable = tqdm(batch_str_moves) if len(batch_str_moves) > 50 else batch_str_moves

    game_stack = []
    for game in iterable:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        for i, move in enumerate(game):
            flip = 1
            if i % 2 == 1:
                flip = -1
            board.umpire(move)
            one_hot = board_state_to_RRC(board.state, flip)
            one_hot[..., 1] = 0
            states.append(one_hot)
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


def board_state_to_lines_RRC(board_state_RR, flip: int) -> t.Tensor:
    board_state_RR = t.tensor(board_state_RR, dtype=t.int8)
    board_state_RR *= flip  # Flip the board to standardize the player's perspective

    lines_board_RRC = t.zeros(8, 8, 9, dtype=t.int8)

    # Directions for movement in the format [dx, dy]
    eights = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

    for r in range(8):
        for c in range(8):
            if board_state_RR[r, c] != 0:
                continue

            # Check each direction from 'eights'
            for direction_idx, (dx, dy) in enumerate(eights):
                direction_idx += 1  # 0 is reserved for the blank space
                x, y = r + dx, c + dy
                found_opponent = False
                while 0 <= x < 8 and 0 <= y < 8 and board_state_RR[x, y] == 1:
                    found_opponent = True
                    x += dx
                    y += dy

                # Check if the line ends with the player's piece (-1)
                if 0 <= x < 8 and 0 <= y < 8 and board_state_RR[x, y] == -1 and found_opponent:
                    lines_board_RRC[r, c, direction_idx] = 1

    return lines_board_RRC


def games_batch_to_state_stack_lines_mine_BLRCC(batch_str_moves: list[list[int]]) -> t.Tensor:

    iterable = tqdm(batch_str_moves) if len(batch_str_moves) > 50 else batch_str_moves

    game_stack = []
    for game in iterable:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        for i, move in enumerate(game):
            flip = 1
            if i % 2 == 1:
                flip = -1
            board.umpire(move)
            one_hot = board_state_to_lines_RRC(board.state, flip)
            one_hot[..., 0] = 0
            states.append(one_hot)
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


def games_batch_to_state_stack_lines_yours_BLRCC(batch_str_moves: list[int]) -> t.Tensor:
    """Difference is in `if i % 2 == 0:` instead of `if i % 2 == 1:`"""

    game_stack = []
    for game in batch_str_moves:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        for i, move in enumerate(game):
            flip = 1
            if i % 2 == 0:
                flip = -1
            board.umpire(move)
            one_hot = board_state_to_lines_RRC(board.state, flip)
            one_hot[..., 0] = 0
            states.append(one_hot)
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


othello_functions = [
    games_batch_to_state_stack_BLRRC.__name__,
    games_batch_to_state_stack_mine_yours_BLRRC.__name__,
    games_batch_to_state_stack_mine_yours_blank_mask_BLRRC.__name__,
    games_batch_to_valid_moves_BLRRC.__name__,
    games_batch_to_state_stack_lines_mine_BLRCC.__name__,
    games_batch_to_state_stack_lines_yours_BLRCC.__name__,
]
