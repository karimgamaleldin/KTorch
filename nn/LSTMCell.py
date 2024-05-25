from autograd.engine import Tensor
from core import KTorch
from nn.Module import Module
from nn.init import simpleUniformInitialization


class LSTMCell(Module):
    """
    A class that represents a single LSTM cell
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        """
        Initialize the LSTM cell
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Initialize the weight matrices
        self.w_ii = simpleUniformInitialization((input_size, hidden_size))
        self.w_if = simpleUniformInitialization((input_size, hidden_size))
        self.w_ig = simpleUniformInitialization((input_size, hidden_size))
        self.w_io = simpleUniformInitialization((input_size, hidden_size))
        self.w_hi = simpleUniformInitialization((hidden_size, hidden_size))
        self.w_hf = simpleUniformInitialization((hidden_size, hidden_size))
        self.w_hg = simpleUniformInitialization((hidden_size, hidden_size))
        self.w_ho = simpleUniformInitialization((hidden_size, hidden_size))

        # Initialize the biases
        if bias:
            self.b_ii = simpleUniformInitialization((hidden_size, 1)).squeeze(-1)
            self.b_if = simpleUniformInitialization((hidden_size, 1)).squeeze(-1)
            self.b_ig = simpleUniformInitialization((hidden_size, 1)).squeeze(-1)
            self.b_io = simpleUniformInitialization((hidden_size, 1)).squeeze(-1)
            self.b_hi = simpleUniformInitialization((hidden_size, 1)).squeeze(-1)
            self.b_hf = simpleUniformInitialization((hidden_size, 1)).squeeze(-1)
            self.b_hg = simpleUniformInitialization((hidden_size, 1)).squeeze(-1)
            self.b_ho = simpleUniformInitialization((hidden_size, 1)).squeeze(-1)

    def forward(self, x: Tensor, h_c: tuple) -> Tensor:
        """
        Forward pass

        Parameters:
          x (Tensor): The input tensor
          h_c (tuple): The hidden state and cell state
        """
        h, c = h_c  # Get the hidden state and cell state

        # The forget gate
        forget = KTorch.matmul(x, self.w_if) + (self.b_if if self.bias else 0)
        forget += KTorch.matmul(h, self.w_hf) + (self.b_hf if self.bias else 0)
        forget = forget.sigmoid()

        # The input gate
        input = KTorch.matmul(x, self.w_ii) + (self.b_ii if self.bias else 0)
        input += KTorch.matmul(h, self.w_hi) + (self.b_hi if self.bias else 0)
        input = input.sigmoid()

        # Output gate
        output = KTorch.matmul(x, self.w_io) + (self.b_io if self.bias else 0)
        output += KTorch.matmul(h, self.w_ho) + (self.b_ho if self.bias else 0)
        output = output.sigmoid()

        # Current cell state
        current_state = KTorch.matmul(x, self.w_ig) + (self.b_ig if self.bias else 0)
        current_state += KTorch.matmul(h, self.w_hg) + (self.b_hg if self.bias else 0)
        current_state = current_state.tanh()

        # New cell state
        new_c = forget * c + input * current_state

        # New hidden state
        new_h = output * new_c.tanh()
        return new_h, new_c

    def parameters(self):
        out = [
            self.w_ii,
            self.w_if,
            self.w_ig,
            self.w_io,
            self.w_hi,
            self.w_hf,
            self.w_hg,
            self.w_ho,
        ]
        if self.bias:
            out.extend(
                [
                    self.b_ii,
                    self.b_if,
                    self.b_ig,
                    self.b_io,
                    self.b_hi,
                    self.b_hf,
                    self.b_hg,
                    self.b_ho,
                ]
            )
        return out
