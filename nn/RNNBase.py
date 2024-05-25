from autograd.engine import Tensor
from core import KTorch
from nn.Module import Module


class RNNBase(Module):
    """
    A class that represents a base class for RNNs (RNN, LSTM, GRU)
    """

    def __init__(
        self,
        mode: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        """
        Initialize the base RNN class
        """
        super().__init__()
        self.mode = mode
        assert mode in [
            "RNN",
            "LSTM",
            "GRU",
        ], "The mode must be one of the following: RNN, LSTM, GRU"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.hidden = None
        self.cell = None
        self.dropout_layers = None
        self.layers = None
        self.bi_layers = None

    def forward(self, x: Tensor, h_0: Tensor = None, h_0_back: Tensor = None) -> Tensor:
        """
        Perform the forward pass of the RNN
        """
        assert (
            self.layers is not None
            or self.bi_layers is not None
            or self.dropout_layers is not None
        ), "Please use a subclass of RNNBase, such as RNN, LSTM, or GRU"
        if self.bidirectional:
            out, hidden = self._forward_bidirectional(x, h_0, h_0_back)
        else:
            out, hidden = self._forward_single_direction(x, h_0)

        return out, hidden

    def _forward_bidirectional(
        self, x: Tensor, h_0: Tensor, h_0_back: Tensor
    ) -> Tensor:
        """
        Perform the forward pass of the bidirectional RNN
        """
        forward_out, forward_hidden = self._forward_single_direction(x, h_0)
        backward_out, backward_hidden = self._forward_single_direction(
            x.flip(1), h_0_back, isBi=True
        )
        out = [
            KTorch.cat((forward, backward), axis=-1)
            for forward, backward in zip(forward_out, backward_out[::-1])
        ]
        hidden = [
            KTorch.cat((forward, backward), axis=-1)
            for forward, backward in zip(forward_hidden, backward_hidden)
        ]
        return out, hidden

    def _forward_single_direction(
        self, x: Tensor, h_0: Tensor, isBi: bool = False
    ) -> Tensor:
        """
        Perform the forward pass of the single direction RNN
        """
        batch_size, seq_len, _ = x.shape
        if h_0 is None:
            if self.mode == "LSTM":
                h_0 = [
                    (
                        KTorch.zeros((batch_size, self.hidden_size)),
                        KTorch.zeros((batch_size, self.hidden_size)),
                    )
                    for _ in range(self.num_layers)
                ]
            else:
                h_0 = [
                    KTorch.zeros((batch_size, self.hidden_size))
                    for _ in range(self.num_layers)
                ]

        h_prev = h_0
        output = []

        for i in range(seq_len):  # iterate over the sequence length
            x_i = x[:, i, :]
            for j in range(self.num_layers):  # iterate over the number of layers
                if self.mode == "LSTM":
                    hidden, c = (
                        self.layers[j](x_i, h_prev[j])
                        if not isBi
                        else self.bi_layers[j](x_i, h_prev[j])
                    )
                else:
                    hidden = (
                        self.layers[j](x_i, h_prev[j])
                        if not isBi
                        else self.bi_layers[j](x_i, h_prev[j])
                    )
                hidden = self.dropout_layers[j](hidden)
                h_prev[j] = (hidden, c) if self.mode == "LSTM" else hidden
                x_i = hidden

            output.append(
                x_i
            )  # append the output of the current time step to the output list

        if self.mode == "LSTM":
            h_prev = [h[0] for h in h_prev]

        return (
            output,
            h_prev,
        )  # return the output of each time step and the hidden states of the last time step

    def parameters(self):
        out = [param for layer in self.layers for param in layer.parameters()]
        if self.bidirectional:
            out += [param for layer in self.bi_layers for param in layer.parameters()]
        return out
