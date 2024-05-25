from autograd.engine import Tensor
from core import KTorch
from nn.RNNBase import RNNBase
from nn.RNNCell import RNNCell
from nn.Dropout import Dropout


class RNN(RNNBase):
    """
    A class that represents a RNN
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        nonlinearity: str = "tanh",
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        """
        Initialize the RNN

        Parameters:
          input_size (int): The size of the input
          hidden_size (int): The size of the hidden state
          num_layers (int): The number of layers
          nonlinearity (str): The nonlinearity function to use (default is 'tanh')
          bias (bool): Whether to use bias (default is True)
          dropout (float): The dropout probability (default is 0.0)
          bidirectional (bool): Whether to use bidirectional RNN (default is False)
        """
        super().__init__(
            "RNN", input_size, hidden_size, num_layers, bias, dropout, bidirectional
        )
        self.nonlinearity = nonlinearity
        self.layers = [
            RNNCell(
                input_size if i == 0 else hidden_size, hidden_size, bias, nonlinearity
            )
            for i in range(num_layers)
        ]
        self.dropout_layers = [Dropout(dropout) for _ in range(num_layers)]
        self.bi_layers = (
            [
                RNNCell(
                    input_size if i == 0 else hidden_size,
                    hidden_size,
                    bias,
                    nonlinearity,
                )
                for i in range(num_layers)
            ]
            if bidirectional
            else None
        )


# Methods for the RNN class are defined in RNNBase.py as they are shared with LSTM and GRU
