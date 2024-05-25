from autograd.engine import Tensor
from core import KTorch
from nn.RNNBase import RNNBase
from nn.LSTMCell import LSTMCell
from nn.Dropout import Dropout


class LSTM(RNNBase):
    """
    A class that represents an LSTM
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        """
        Initialize the LSTM
        """
        super().__init__(
            "LSTM", input_size, hidden_size, num_layers, bias, dropout, bidirectional
        )
        self.layers = [
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias)
            for i in range(num_layers)
        ]
        self.dropout_layers = [Dropout(dropout) for _ in range(num_layers)]
        self.bi_layers = (
            [
                LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias)
                for i in range(num_layers)
            ]
            if bidirectional
            else None
        )

    # Methods for the LSTM class are defined in RNNBase.py as they are shared with RNN and GRU
