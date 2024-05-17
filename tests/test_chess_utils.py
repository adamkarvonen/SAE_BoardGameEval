import circuits.chess_utils as chess_utils
import chess
import torch

device = "cpu"


def test_white_pos_indices():
    test1 = ";1.e4 c5 2.Nf3 d6 3"
    test2 = ";1.e4 c5 2.Nf3 d"
    test3 = ";1."

    ans1 = [[0, 1, 2, 3, 4], [8, 9, 10, 11, 12, 13], [17, 18]]
    ans2 = [[0, 1, 2, 3, 4], [8, 9, 10, 11, 12, 13]]
    ans3 = [[0, 1, 2]]

    assert chess_utils.get_all_white_pos_indices(test1) == ans1
    assert chess_utils.get_all_white_pos_indices(test2) == ans2
    assert chess_utils.get_all_white_pos_indices(test3) == ans3


def test_black_pos_indices():
    test1 = ";1.e4 c5 2.Nf3 d6 3"
    test2 = ";1.e4 c5 2.Nf3 d"
    test3 = ";1."

    ans1 = [[5, 6, 7], [14, 15, 16]]
    ans2 = [[5, 6, 7], [14, 15]]
    ans3 = []

    assert chess_utils.get_all_black_pos_indices(test1) == ans1
    assert chess_utils.get_all_black_pos_indices(test2) == ans2
    assert chess_utils.get_all_black_pos_indices(test3) == ans3


def test_board_to_piece_state():

    test_str = ";1.e4 e5 2.Nf3"
    board = chess_utils.pgn_string_to_board(test_str)
    state = chess_utils.board_to_piece_state(board)

    expected_state = torch.tensor(
        [
            [4, 2, 3, 5, 6, 3, 0, 4],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [-1, -1, -1, -1, 0, -1, -1, -1],
            [-4, -2, -3, -5, -6, -3, -2, -4],
        ],
        dtype=torch.int8,
    )

    assert torch.equal(state, expected_state)


def test_create_state_stacks():

    test_strs = [";1.e4 e5 2.Nf3 ", ";1.e4 e5 2.Nf3 "]

    state_stacks_dict_BLRR = chess_utils.create_state_stacks(
        test_strs, [chess_utils.board_to_piece_state, chess_utils.board_to_pin_state], device
    )

    expected_state = torch.tensor(
        [
            [4, 2, 3, 5, 6, 3, 0, 4],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [-1, -1, -1, -1, 0, -1, -1, -1],
            [-4, -2, -3, -5, -6, -3, -2, -4],
        ],
        dtype=torch.int8,
    )

    assert torch.equal(
        state_stacks_dict_BLRR[chess_utils.board_to_piece_state.__name__][0][-1], expected_state
    )

    assert torch.equal(
        state_stacks_dict_BLRR[chess_utils.board_to_pin_state.__name__][0][-1],
        torch.tensor([[0]], dtype=torch.int8),
    )


def test_state_stack_to_one_hot():
    test_strs = [";1.e4 e5 2.Nf3 ", ";1.e4 e5 2.Nf3 "]

    white_knight_index = 8
    white_rook_index = 10

    custom_functions = [chess_utils.board_to_piece_state, chess_utils.board_to_pin_state]

    state_stacks_dict_BLRR = chess_utils.create_state_stacks(test_strs, custom_functions, device)
    one_hot_dict = {}

    for custom_function in custom_functions:
        func_name = custom_function.__name__
        config = chess_utils.config_lookup[func_name]
        one_hot_dict[func_name] = chess_utils.state_stack_to_one_hot(
            config, device, state_stacks_dict_BLRR[func_name]
        )

    expected_one_hot_A1_rook = torch.tensor(1)
    expected_one_hot_A1_knight = torch.tensor(0)

    assert torch.equal(
        one_hot_dict[chess_utils.board_to_piece_state.__name__][0][-1][0][0][white_rook_index],
        expected_one_hot_A1_rook,
    )

    assert torch.equal(
        one_hot_dict[chess_utils.board_to_piece_state.__name__][0][-1][0][0][white_knight_index],
        expected_one_hot_A1_knight,
    )


def test_board_to_pin_state():
    initial_board = chess.Board()
    expected_answer = torch.tensor([[0]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_pin_state(initial_board), expected_answer)

    test_str = "1. e4 e5 2. d3 f6 3. Nd2 Bb4"
    board = chess_utils.typical_pgn_string_to_board(test_str)
    expected_answer = torch.tensor([[1]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_pin_state(board), expected_answer)


def test_board_to_has_castling_rights():
    initial_board = chess.Board()
    expected_answer = torch.tensor([[1]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_has_castling_rights(initial_board), expected_answer)

    test_str = "1. e4 d5 2. Nc3 dxe4 3. Nxe4 Nf6 4. d3 e6 5. Be3 h6 6. Qf3 Be7 7. O-O-O Nh5"
    board = chess_utils.typical_pgn_string_to_board(test_str)
    expected_answer = torch.tensor([[0]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_has_castling_rights(board), expected_answer)


def test_board_to_has_queenside_castling_rights():
    initial_board = chess.Board()
    expected_answer = torch.tensor([[1]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_has_queenside_castling_rights(initial_board), expected_answer)

    test_str = "1. e4 d5 2. Nc3 dxe4 3. Nxe4 Nf6 4. d3 e6 5. Be3 h6 6. Qf3 Be7 7. O-O-O Nh5"
    board = chess_utils.typical_pgn_string_to_board(test_str)
    expected_answer = torch.tensor([[0]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_has_queenside_castling_rights(board), expected_answer)


def test_board_to_has_kingside_castling_rights():
    initial_board = chess.Board()
    test_str = "1. e4 d5 2. Nc3 dxe4 3. Nxe4 Nf6 4. d3 e6 5. Be3 h6 6. Qf3 Be7 7. O-O-O"
    board = chess_utils.typical_pgn_string_to_board(test_str)
    
    expected_answer = torch.tensor([[1]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_has_kingside_castling_rights(board), expected_answer)
    
    test_str = "1. e4 d5 2. Nc3 dxe4 3. Nxe4 Nf6 4. d3 e6 5. Be3 h6 6. Qf3 Be7 7. O-O-O Nh5"
    board = chess_utils.typical_pgn_string_to_board(test_str)
    expected_answer = torch.tensor([[0]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_has_kingside_castling_rights(board), expected_answer)


def test_board_to_has_legal_en_passant():
    initial_board = chess.Board()
    expected_answer = torch.tensor([[0]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_has_legal_en_passant(initial_board), expected_answer)

    test_str = "1. e4 Nc6 2. e5 f5"
    board = chess_utils.typical_pgn_string_to_board(test_str)
    expected_answer = torch.tensor([[1]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_has_legal_en_passant(board), expected_answer)
    

def test_board_to_can_claim_draw():
    initial_board = chess.Board()
    expected_answer = torch.tensor([[0]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_can_claim_draw(initial_board), expected_answer)

    test_str = "1. Nf3 Nc6 2. Ng1 Nb8 3. Nf3 Nc6 4. Ng1 Nb8 5. Nf3 Nc6 6. Ng1 Nb8"
    board = chess_utils.typical_pgn_string_to_board(test_str)
    expected_answer = torch.tensor([[1]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_can_claim_draw(board), expected_answer)
    

def test_board_to_is_stalemate():
    initial_board = chess.Board()
    expected_answer = torch.tensor([[0]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_is_stalemate(initial_board), expected_answer)

    test_str = "1. e3 a5 2. Qh5 Ra6 3. Qxa5 h5 5. h4 Rah6 5. Qxc7 f6 6. Qxd7+ Kf7 7. Qxb7 Qd3 8. Qxb8 Qh7 9. Qxc8 Kg6 10. Qe6" 
    board = chess_utils.typical_pgn_string_to_board(test_str)
    expected_answer = torch.tensor([[1]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_is_stalemate(board), expected_answer)


def test_board_to_can_check_next():
    initial_board = chess.Board()
    expected_answer = torch.tensor([[0]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_can_check_next(initial_board), expected_answer)

    test_str = "1. e4 f5" 
    board = chess_utils.typical_pgn_string_to_board(test_str)
    expected_answer = torch.tensor([[1]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_can_check_next(board), expected_answer)


def test_board_to_material():
    initial_board = chess.Board()
    expected_answer = torch.tensor([[39]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_material(initial_board), expected_answer)

    test_str = "1. e4 d5 2. exd5" 
    board = chess_utils.typical_pgn_string_to_board(test_str)
    expected_answer = torch.tensor([[38]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_material(board), expected_answer)


def test_board_to_number_of_pieces():
    initial_board = chess.Board()
    expected_answer = torch.tensor([[15]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_number_of_pieces(initial_board), expected_answer)

    test_str = "1. e4 d5 2. exd5" 
    board = chess_utils.typical_pgn_string_to_board(test_str)
    expected_answer = torch.tensor([[14]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_number_of_pieces(board), expected_answer)


def test_board_to_has_bishop_pair():
    initial_board = chess.Board()
    expected_answer = torch.tensor([[1]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_has_bishop_pair(initial_board), expected_answer)

    test_str = "1. e4 d5 2. Bc4 dxc4" 
    board = chess_utils.typical_pgn_string_to_board(test_str)
    expected_answer = torch.tensor([[0]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_has_bishop_pair(board), expected_answer)


def test_board_to_has_mate_threat():
    initial_board = chess.Board()
    expected_answer = torch.tensor([[0]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_has_mate_threat(initial_board), expected_answer)

    test_str = "1. e4 f6 2. Nc3 g5 3.d3" 
    board = chess_utils.typical_pgn_string_to_board(test_str)
    expected_answer = torch.tensor([[1]], dtype=torch.int8)
    #print("mate thres", chess_utils.board_to_has_mate_threat(board))
    assert torch.equal(chess_utils.board_to_has_mate_threat(board), expected_answer)


def test_board_to_can_capture_queen():
    initial_board = chess.Board()
    expected_answer = torch.tensor([[0]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_can_capture_queen(initial_board), expected_answer)

    test_str = "1. e4 g6 2. Qh5" 
    board = chess_utils.typical_pgn_string_to_board(test_str)
    expected_answer = torch.tensor([[1]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_can_capture_queen(board), expected_answer)


def test_board_to_has_queen():
    initial_board = chess.Board()
    expected_answer = torch.tensor([[1]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_has_queen(initial_board), expected_answer)

    test_str = "1. e4 g6 2. Qh5 gxh5" 
    board = chess_utils.typical_pgn_string_to_board(test_str)
    expected_answer = torch.tensor([[0]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_has_queen(board), expected_answer)


def test_board_to_has_connected_rooks():
    initial_board = chess.Board()
    expected_answer = torch.tensor([[0]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_has_connected_rooks(initial_board), expected_answer)

    test_str = "1. a4 d6 2. Ra3 e6 3. h4 f6 4. Rhh3 c6" 
    board = chess_utils.typical_pgn_string_to_board(test_str)
    expected_answer = torch.tensor([[1]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_has_connected_rooks(board), expected_answer)
  
  
def test_board_to_has_ambiguous_moves():
    initial_board = chess.Board()
    expected_answer = torch.tensor([[0]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_ambiguous_moves(initial_board), expected_answer)

    test_str = "1. a4 d6 2. Ra3 e6 3. h4 f6 4. Rhh3 c6" 
    board = chess_utils.typical_pgn_string_to_board(test_str)
    expected_answer = torch.tensor([[1]], dtype=torch.int8)
    assert torch.equal(chess_utils.board_to_ambiguous_moves(board), expected_answer)
