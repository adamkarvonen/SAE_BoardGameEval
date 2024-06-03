import circuits.othello_utils as othello_utils
import circuits.othello_engine_utils as othello_engine_utils
import torch

device = "cpu"


def test_decoding():
    input = [
        20,
        21,
        34,
        19,
        13,
        40,
        47,
        28,
        12,
        41,
        35,
        5,
        10,
    ]

    expected_output = [19, 20, 37, 18, 12, 43, 50, 29, 11, 44, 38, 4, 9]

    output = othello_engine_utils.to_string(input)
    assert output == expected_output


def test_board_to_state():
    input = [[19, 20, 37, 18, 12, 43, 50, 29, 11, 44, 38, 4, 9]]

    boards_state_stack = othello_utils.games_batch_to_state_stack_BLRRC(input)

    move_of_interest = 0
    r = input[0][move_of_interest] // 8
    c = input[0][move_of_interest] % 8

    # Black just placed a piece on r, c
    assert torch.equal(
        boards_state_stack[0][0][r][c], torch.tensor([0, 0, 1], dtype=torch.int8, device=device)
    )

    new_move_of_interest = 5
    masked_board = boards_state_stack.clone()
    masked_board[0, 5, :, :, 1] = 0
    assert masked_board[0][5].sum().item() == new_move_of_interest + 5


def test_board_to_mine_yours_state():
    input = [[19, 20, 37, 18, 12, 43, 50, 29, 11, 44, 38, 4, 9]]

    boards_state_stack = othello_utils.games_batch_to_state_stack_mine_yours_BLRRC(input)

    move_of_interest = 0
    r = input[0][move_of_interest] // 8
    c = input[0][move_of_interest] % 8

    # For visualization
    print(torch.argmax(boards_state_stack[0][0], dim=-1) - 1)

    # I just placed a piece on r, c
    assert torch.equal(
        boards_state_stack[0][0][r][c], torch.tensor([0, 0, 1], dtype=torch.int8, device=device)
    )

    # That piece wasn't flipped, so now it becomes your piece from the current perspective
    assert torch.equal(
        boards_state_stack[0][1][r][c], torch.tensor([1, 0, 0], dtype=torch.int8, device=device)
    )

    new_move_of_interest = 5
    masked_board = boards_state_stack.clone()
    masked_board[0, 5, :, :, 1] = 0
    assert masked_board[0][5].sum().item() == new_move_of_interest + 5


def test_board_to_valid_moves_state():
    input = [[19, 20, 37, 18, 12, 43, 50, 29, 11, 44, 38, 4, 9]]

    boards_state_stack = othello_utils.games_batch_to_valid_moves_BLRRC(input)

    move_of_interest = 0

    expected_legal_moves = [18, 20, 34]

    for i in range(8):
        for j in range(8):
            if i * 8 + j in expected_legal_moves:
                assert boards_state_stack[0][move_of_interest][i][j].item() == 1
            else:
                assert boards_state_stack[0][move_of_interest][i][j].item() == 0


def test_board_to_mine_lines_state():
    input = [[19]]

    boards_state_stack = othello_utils.games_batch_to_state_stack_lines_mine_BLRCC(input)

    move_of_interest = 0

    print(torch.argmax(boards_state_stack[0][move_of_interest], dim=0))

    expected_lines = [(2, 2, 3), (2, 4, 4), (4, 2, 2)]

    for expected_line in expected_lines:
        r, c, v = expected_line
        assert boards_state_stack[0][move_of_interest][r][c][v].item() == 1
    assert boards_state_stack[0][move_of_interest].sum().item() == len(expected_lines)


def test_board_to_yours_lines_state():
    input = [[19]]

    boards_state_stack = othello_utils.games_batch_to_state_stack_lines_yours_BLRCC(input)

    move_of_interest = 0

    # print(torch.argmax(boards_state_stack[0][move_of_interest], dim=0))
    # print(boards_state_stack[0][move_of_interest])

    expected_lines = [(4, 5, 6), (5, 4, 0), (5, 5, 7)]

    for expected_line in expected_lines:
        r, c, v = expected_line
        assert boards_state_stack[0][move_of_interest][r][c][v].item() == 1
    assert boards_state_stack[0][move_of_interest].sum().item() == len(expected_lines)
