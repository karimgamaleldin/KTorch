from autograd.engine import Tensor


class Module:
    """
    A class that represents a module in a neural network
    """

    def __init__(self):
        """
        Initialize the module
        """
        self.training = True

    def __call__(self, *args) -> Tensor:
        """
        Perform the forward pass of the module
        """
        return self.forward(*args)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform the forward pass of the module
        """
        raise NotImplementedError

    def parameters(self):
        """
        Return the parameters of the module
        """
        return []
