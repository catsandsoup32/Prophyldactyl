import torch
from torch import nn

# Original network from StockFish docs 
class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()
        NUM_FEATURES = 40960 # 64*64*5*2, given there are 5 pieces except for king, and two perspectives
        M, N, K = 4, 8, 1

        self.ft = nn.Linear(NUM_FEATURES, M)
        self.l1 = nn.Linear(2 * M, N)
        self.l2 = nn.Linear(N, K)

    # The inputs are a whole batch!
    # `stm` indicates whether white is the side to move. 1 = true, 0 = false.
    # Features are tuples (our_king_square, piece_square, piece_type, piece_color)
    def forward(self, white_features, black_features, stm):
        w = self.ft(white_features) # white's perspective
        b = self.ft(black_features) # black's perspective

        # Remember that we order the accumulators for 2 perspectives based on who is to move.
        # So we blend two possible orderings by interpolating between `stm` and `1-stm` tensors.
        accumulator = (stm * torch.cat([w, b], dim=1)) + ((1 - stm) * torch.cat([b, w], dim=1))

        # Run the linear layers and use clamp_ as ClippedReLU
        l1_x = torch.clamp(accumulator, 0.0, 1.0)
        l2_x = torch.clamp(self.l1(l1_x), 0.0, 1.0)
        return self.l2(l2_x)
    