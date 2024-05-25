from autograd.engine import Tensor


class Optim:
    """
    The base optimizer class
    """

    def __init__(self, params: Tensor):
        """
        Initialize the optimizer with the parameters
        """
        self.params = params

    def step(self):
        """
        Update the parameters using the gradients
        """
        raise NotImplementedError

    def zero_grad(self):
        """
        Zero the gradients of the parameters
        """
        for param in self.params:
            param.zero_grad()
