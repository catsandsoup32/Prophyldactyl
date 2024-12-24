import numpy as np
from torch import tensor


def fen_str_to_array(fen_string):
    field_array = fen_string.split(" ")
    piece_placement = field_array.pop(0).split('/')
    new_piece_placement = []

    # Fill in spaces, ex. [R, 4, b, 2] becomes [R, 0, 0, 0, 0, b, 0, 0]
    for idx, rank in enumerate(piece_placement):
        rank = list(rank)
        new_rank = []

        if ('K') in rank:
            white_king_rank = idx 
        if ('k') in rank:
            black_king_rank = idx 
        
        for tile in rank:
            if tile.isdigit():
                num_spaces = int(tile)
                for _ in range(num_spaces):
                    new_rank.append('0')
                continue
            new_rank.append(tile)
        new_piece_placement.append(new_rank)
           
    return field_array, np.array(new_piece_placement), white_king_rank, black_king_rank


def fen_to_768(fen_string):
    _, piece_placement, _, _ = fen_str_to_array(fen_string)
    return_vec = np.zeros((64, 6, 2), dtype=np.float32)

    white_piece_dict = {'R': 0, 'N': 1, 'B': 2, 'Q': 3, 'P': 4, 'K': 5}
    black_piece_dict = {'r': 0, 'n': 1, 'b': 2, 'q': 3, 'p': 4, 'k': 5}

    for i, rank in enumerate(piece_placement):
        for j, tile in enumerate(rank):

            if tile in white_piece_dict:
                piece_type = white_piece_dict[tile]
                return_vec[ (i*8 + j), piece_type, 0] = 1

            elif tile in black_piece_dict: 
                piece_type = black_piece_dict[tile]
                return_vec[ (i*8 + j), piece_type, 1] = 1

    return_vec = tensor(return_vec)
    return_vec = return_vec.view(768)
    return return_vec   


def fen_to_halfKP(fen_string):
    # Need to create two tuples (our_king_square, piece_square, piece_type, piece_color)
    # If one wanted to, it would technically be more efficient to cache these tensors rather than perform this transform on-the-fly

    # Using index notation of a1 = 0, ..., h8 = 63 for white and a8 = 0, ..., h1 = 63 for black
    # See https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md HalfKP section for a more thorough explanation
    white_tensor = black_tensor = np.zeros((64, 64, 5, 2), dtype=np.float32)  
    field_array, piece_placement, w_king_rank, b_king_rank = fen_str_to_array(fen_string)
    
    # Find king squares
    for idx, tile in enumerate(piece_placement[w_king_rank]):
        if tile == 'K':
            w_king_square = ((7-w_king_rank) * 8) + idx
            break
    for idx, tile in enumerate(piece_placement[b_king_rank]):
        if tile == 'k':
            b_king_square = (b_king_rank * 8) + idx
            break

    # Find other pieces and assign placements to the array 
    # This sparseness is what enables efficient updating
    # Ordered as [rook, knight, bishop, queen, pawn] in the third dimension
    # Index 0 is own side and index 1 is opposing side in the fourth dim
    white_piece_dict = {'R': 0, 'N': 1, 'B': 2, 'Q': 3, 'P': 4}
    black_piece_dict = {'r': 0, 'n': 1, 'b': 2, 'q': 3, 'p': 4}
    for i, rank in enumerate(piece_placement):
        for j, tile in enumerate(rank):

            if tile in white_piece_dict:
                piece_type = white_piece_dict[tile]
                white_tensor[w_king_square, (7-i)*8 + j, piece_type, 0] = 1
                black_tensor[b_king_square, i*8 + j, piece_type, 1] = 1

            elif tile in black_piece_dict: 
                piece_type = black_piece_dict[tile]
                white_tensor[w_king_square, (7-i)*8 + j, piece_type, 1] = 1
                black_tensor[b_king_square, i*8 + j, piece_type, 0] = 1

    # Return if white is the side to move
    stm = 1 if field_array[0] == 'w' else 0

    return tensor(white_tensor), tensor(black_tensor), tensor(stm)

   

    