import torch as t
from datasets import load_dataset
from circuits.othello_engine_utils import OthelloBoardState, stoi, itos


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


def board_state_to_RRC(board_state):
    board_state = t.tensor(board_state, dtype=t.int8)
    one_hot = t.zeros((8, 8, 3), dtype=t.int8)
    one_hot[..., 0] = (board_state == -1).int()
    one_hot[..., 1] = (board_state == 0).int()
    one_hot[..., 2] = (board_state == 1).int()
    return one_hot


def games_batch_to_state_stack_BLRRC(batch_str_moves):
    """Sequences of moves (dataset format) to state stack (one-hot) of shape (seq_len, 8, 8, 3)"""
    game_stack = []
    for game in batch_str_moves:
        if isinstance(str_moves, t.Tensor):
            str_moves = str_moves.flatten()

        board = OthelloBoardState()
        states = []
        for move in game:
            board.umpire(move)
            one_hot = board_state_to_RRC(board.state)
            states.append(one_hot)
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


othello_functions = [games_batch_to_state_stack_BLRRC.__name__]
