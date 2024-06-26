import numpy as np
from scipy.special import erf


class Tensor:
    """
    Stores a numpy array and its gradient
    """

    def __init__(self, data, _prev=(), _op="", label="", requires_grad=True):
        """
        Initialize the tensor with the data
        params:
          data: numpy array: the data of the tensor
          _prev: tuple: the previous tensors that were used to compute the current tensor
          op: string: the operation that was used to compute the current tensor
        """
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.grad = np.zeros_like(data, dtype=np.float32)
        self._prev = _prev
        self._op = _op
        self.label = label
        self._backward = lambda: None
        self.requires_grad = requires_grad
        self.ndim = self.data.ndim

    def check_broadcastable(self, other: "Tensor"):
        """
        Checks if the tensors are broadcastable according to numpy rules
        """
        if self.data.shape == other.data.shape:
            return True

        self_shape = np.array(self.data.shape)
        other_shape = np.array(other.data.shape)

        # Pad the shapes with ones
        max_dim = max(len(self_shape), len(other_shape))
        self_shape = np.pad(
            self_shape, (max_dim - len(self_shape), 0), "constant", constant_values=1
        )
        other_shape = np.pad(
            other_shape, (max_dim - len(other_shape), 0), "constant", constant_values=1
        )

        # Check if the shapes are broadcastable
        compatible = (
            (self_shape == other_shape) | (self_shape == 1) | (other_shape == 1)
        )

        return compatible.all()

    def __len__(self):
        """
        Get the length of the tensor
        """
        return len(self.data)

    def __add__(self, other):
        """
        Add the data of the tensor with another tensor
        """
        if isinstance(other, (Tensor, np.ndarray, int, float)):
            other = Tensor(other) if isinstance(other, (np.ndarray)) else other
            other = (
                Tensor(np.array([other])) if isinstance(other, (int, float)) else other
            )  # Convert to tensor if it is a number
            if not self.check_broadcastable(other):
                raise ValueError("The shapes of the tensors are not broadcastable")
            out = Tensor(
                self.data + other.data,
                _prev=(self, other),
                _op="add",
                label=f"{self.label} + {other.label}",
            )
        else:
            raise TypeError("The input must be a tensor or a number")

        def _backward():
            if self.data.shape == other.data.shape:
                self.grad += out.grad
                other.grad += out.grad
            else:
                # Check for self and out
                # Pad the shapes with ones
                max_dim = max(len(self.data.shape), len(out.data.shape))
                self_shape = np.pad(
                    self.data.shape,
                    (max_dim - len(self.data.shape), 0),
                    "constant",
                    constant_values=1,
                )
                out_shape = np.pad(
                    out.data.shape,
                    (max_dim - len(out.data.shape), 0),
                    "constant",
                    constant_values=1,
                )

                # Get the axes where the shapes will be broadcasted
                self_axes = [
                    axis
                    for axis, (s, os) in enumerate(zip(self_shape, out_shape))
                    if s == 1 and os != 1
                ]
                new_grad = np.sum(out.grad, axis=tuple(self_axes), keepdims=True)
                self.grad += new_grad.reshape(self.data.shape)

                # Check for other and out
                # Pad the shapes with ones
                max_dim = max(len(other.data.shape), len(out.data.shape))
                other_shape = np.pad(
                    other.data.shape,
                    (max_dim - len(other.data.shape), 0),
                    "constant",
                    constant_values=1,
                )
                out_shape = np.pad(
                    out.data.shape,
                    (max_dim - len(out.data.shape), 0),
                    "constant",
                    constant_values=1,
                )

                # Get the axes where the shapes will be broadcasted
                other_axes = [
                    axis
                    for axis, (s, os) in enumerate(zip(other_shape, out_shape))
                    if s == 1 and os != 1
                ]
                new_grad = np.sum(out.grad, axis=tuple(other_axes), keepdims=True)
                other.grad += new_grad.reshape(other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        """
        Add the data of the tensor with another tensor or number
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtract the data of the tensor with another tensor
        """
        return self.__add__(-1 * other)

    def __rsub__(self, other):
        """
        Subtract the data of the tensor from another tensor or number
        """
        return (-1 * self).__add__(other)

    def __mul__(self, other):
        """
        Multiply the data of the tensor with another tensor
        """
        if isinstance(other, (Tensor, np.ndarray, int, float)):
            other = (
                Tensor(other) if isinstance(other, (np.ndarray, int, float)) else other
            )
            if not self.check_broadcastable(other):
                raise ValueError("The shapes of the tensors are not broadcastable")
            out = Tensor(
                self.data * other.data,
                _prev=(self, other),
                _op="mul",
                label=f"{self.label} * {other.label}",
            )
        else:
            raise TypeError("The tensor must be a tensor or a number")

        def _backward():
            if self.data.shape == other.data.shape:
                self.grad += out.grad * other.data
                other.grad += out.grad * self.data
            else:
                # Broadcasting
                max_dim = max(len(self.data.shape), len(other.data.shape))
                self_shape = np.pad(
                    self.data.shape,
                    (max_dim - len(self.data.shape), 0),
                    "constant",
                    constant_values=1,
                )
                other_shape = np.pad(
                    other.data.shape,
                    (max_dim - len(other.data.shape), 0),
                    "constant",
                    constant_values=1,
                )
                out_shape = np.pad(
                    out.data.shape,
                    (max_dim - len(out.data.shape), 0),
                    "constant",
                    constant_values=1,
                )

                # Get the axes where the shapes will be broadcasted
                self_axes = [
                    axis
                    for axis, (s, os) in enumerate(zip(self_shape, out_shape))
                    if s == 1 and os != 1
                ]
                other_axes = [
                    axis
                    for axis, (s, os) in enumerate(zip(other_shape, out_shape))
                    if s == 1 and os != 1
                ]

                new_grad = np.sum(
                    out.grad * other.data, axis=tuple(self_axes), keepdims=True
                )
                self.grad += new_grad.reshape(self.data.shape)
                new_grad = np.sum(
                    out.grad * self.data, axis=tuple(other_axes), keepdims=True
                )
                other.grad += new_grad.reshape(other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        """
        Multiply the data of the tensor with another tensor or number
        """
        return self * other

    def __pow__(self, other):
        """
        Raise the data of the tensor to the power of a number
        """
        assert isinstance(other, (int, float)), "The exponent must be a number"
        out = Tensor(
            self.data**other, _prev=(self,), _op="pow", label=f"{self.label} ** {other}"
        )

        def _backward():
            self.grad += out.grad * other * np.power(self.data, other - 1)

        out._backward = _backward
        return out

    def square(self):
        """
        Square the data of the tensor
        """
        return self.__pow__(2)

    def __repr__(self):
        return f"tensor: {self.data}"

    def __str__(self):
        return f"tensor: {self.data}"

    def ReLU(self):
        """
        Apply the ReLU function to the tensor
        """
        out = Tensor(
            np.maximum(self.data, 0),
            _prev=(self,),
            _op="ReLU",
            label=f"ReLU({self.label})",
        )

        def _backward():
            self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out

    def sigmoid(self):
        """
        Apply the sigmoid function to the tensor

        This version is a stable version of the sigmoid function as:
        - for x > 0, sigmoid(x) = 1 / (1 + exp(-x)) - for big negative values it is unstable
        - for x < 0, sigmoid(x) = exp(x) / (1 + exp(x)) - for big positive values it is unstable

        """
        pos_mask = self.data >= 0
        t = np.zeros_like(self.data)
        t[pos_mask] = 1 / (1 + np.exp(-self.data[pos_mask]))
        t[~pos_mask] = np.exp(self.data[~pos_mask]) / (1 + np.exp(self.data[~pos_mask]))
        out = Tensor(t, _prev=(self,), _op="sigmoid", label=f"sigmoid({self.label})")

        def _backward():
            self.grad += out.grad * t * (1 - t)

        out._backward = _backward
        return out

    def tanh(self):
        """
        Apply the tanh function to the tensor
        """
        t = np.tanh(self.data)
        out = Tensor(t, _prev=(self,), _op="tanh", label=f"tanh({self.label})")

        def _backward():
            self.grad += out.grad * (1 - t**2)

        out._backward = _backward
        return out

    def __neg__(self):
        """
        Negate the data of the tensor
        """
        return self.__mul__(-1)

    def __truediv__(self, other):
        """
        Divide the data of the tensor by another tensor
        """
        return self * (other**-1)

    def exp(self):
        """
        Apply the exponential function to the tensor
        """
        t = np.exp(self.data)
        out = Tensor(t, _prev=(self,), _op="exp", label=f"exp({self.label})")

        def _backward():
            self.grad += out.grad * t

        out._backward = _backward
        return out

    def log(self):
        """
        Apply the logarithm function to the tensor
        """
        # Prevent log(0) and handle very small values to avoid overflow in division
        epsilon = 1e-10
        safe_data = np.where(self.data <= 0, epsilon, self.data)
        t = np.log(safe_data)
        out = Tensor(t, _prev=(self,), _op="log", label=f"log({self.label})")

        def _backward():
            self.grad += out.grad * (1 / safe_data)

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        """
        Compute the sum of the tensor
        """
        t = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(t, _prev=(self,), _op="sum", label=f"sum({self.label})")

        def _backward():
            if axis is None:
                self.grad += out.grad * np.ones_like(self.data)
            else:
                output_shape = np.array(out.data.shape)
                self_shape = np.array(self.data.shape)
                axes = np.atleast_1d(axis)

                if not keepdims:  # Return the shape to the original shape
                    output_shape = np.insert(output_shape, axes, 1)

                self.grad += np.broadcast_to(
                    out.grad, self_shape
                )  # Broadcast the gradient to the original shape

        out._backward = _backward
        return out

    def backward(self):
        """
        Compute the gradient for all the previous tensors using topological sort
        """
        topo = []
        visited = set()

        def dfs(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    dfs(child)
                topo.append(v)

        dfs(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            if isinstance(node, Tensor):
                node._backward()
            elif isinstance(node, [tuple, list]):
                for n in node:
                    n._backward()

    def __matmul__(self, other):
        """
        Compute the matrix multiplication of 2 tensors
        """
        if self.data.ndim == 1:
            self.data = self.data.reshape(1, -1)
        if other.data.ndim == 1:
            other.data = other.data.reshape(1, -1)
        t = np.matmul(self.data, other.data)
        out = Tensor(
            t, _prev=(self, other), _op="matmul", label=f"{self.label} @ {other.label}"
        )

        def _backward():
            """
            self, self.grad - matrix of shape (..., n, m)
            other, other.grad - tensor of shape (..., m, p)
            out, out.grad - tensor of shape (n, p)
            """
            other_t = np.swapaxes(other.data, -1, -2)
            self.grad += np.matmul(
                out.grad, other_t
            )  # (..., n, p) * (..., p, m) = (..., n, m)
            self_t = np.swapaxes(self.data, -1, -2)
            other.grad += np.matmul(
                self_t, out.grad
            )  # (..., m, n) * (..., n, p) = (..., m, p)

        out._backward = _backward
        return out

    def _zero_grad(self):
        """
        Zero the gradient of the tensor
        """
        self.grad = np.zeros_like(self.data)

    def zero_grad(self):
        self._zero_grad()

    def flatten(self, start_dim=None, end_dim=-1):
        """
        Flatten the tensor
        """
        if start_dim is None:
            start_dim = self.data.shape[0]
        t = self.data.reshape(start_dim, end_dim)
        out = Tensor(t, _prev=(self,), _op="flatten", label=f"flatten({self.label})")

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def sqrt(self):
        """
        Compute the square root of the tensor
        """
        t = np.sqrt(self.data)
        out = Tensor(t, _prev=(self,), _op="sqrt", label=f"sqrt({self.label})")

        def _backward():
            self.grad += out.grad * 0.5 * np.power(self.data, -0.5)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """
        Compute the mean of the tensor
        """
        t = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(t, _prev=(self,), _op="mean", label=f"mean({self.label})")

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad / np.prod(self.data.shape)

        out._backward = _backward
        return out

    def var(self, axis=None, keepdims=False, unbiased=False):
        """
        Compute the variance of the tensor
        """
        t = np.var(self.data, axis=axis, keepdims=keepdims, ddof=1 if unbiased else 0)
        out = Tensor(t, _prev=(self,), _op="var", label=f"var({self.label})")

        def _backward():
            if unbiased:
                self.grad += (
                    out.grad
                    * 2
                    * (self.data - self.mean(axis=axis, keepdims=keepdims).data)
                    / (self.data.size - 1)
                )
            else:
                self.grad += (
                    out.grad
                    * 2
                    * (self.data - self.mean(axis=axis, keepdims=keepdims).data)
                    / self.data.size
                )

        out._backward = _backward
        return out

        # '''
        # Compute the variance of the tensor
        # '''
        # t = np.var(self.data, axis=axis, keepdims=keepdims)
        # out = Tensor(t, _prev=(self,), _op='var', label=f"var({self.label})")
        # def _backward():
        #   self.grad += 2 * (self.data - self.mean(axis=axis, keepdims=keepdims).data) * out.grad / self.data.size

        # out._backward = _backward
        # return out

    def __getitem__(self, idx):
        """
        Get the item at the specified index
        """
        t = self.data[idx]
        out = Tensor(t, _prev=(self,), _op="getitem", label=f"{self.label}[{idx}]")

        def _backward():
            self.grad[idx] += out.grad

        out._backward = _backward
        return out

    def view(self, *shape):
        """
        Reshape the tensor
        """
        t = self.data.reshape(*shape)
        out = Tensor(t, _prev=(self,), _op="view", label=f"view({self.label})")

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        """
        Compute the maximum of the tensor
        """
        t = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(t, _prev=(self,), _op="max", label=f"max({self.label})")

        def _backward():
            self.grad += (self.data == out.data) * out.grad

        out._backward = _backward
        return out

    def min(self, axis=None, keepdims=False):
        """
        Compute the minimum of the tensor
        """
        t = np.min(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(t, _prev=(self,), _op="min", label=f"min({self.label})")

        def _backward():
            self.grad += (self.data == out.data) * out.grad

        out._backward = _backward
        return out

    def cos(self):
        """
        Compute the cosine of the tensor
        """
        t = np.cos(self.data)
        out = Tensor(t, _prev=(self,), _op="cos", label=f"cos({self.label})")

        def _backward():
            self.grad += -np.sin(self.data) * out.grad

        out._backward = _backward
        return out

    def sin(self):
        """
        Compute the sine of the tensor
        """
        t = np.sin(self.data)
        out = Tensor(t, _prev=(self,), _op="sin", label=f"sin({self.label})")

        def _backward():
            self.grad += np.cos(self.data) * out.grad

        out._backward = _backward
        return out

    def __gt__(self, other):
        """
        Compare the data of the tensor with another tensor
        """
        if isinstance(other, Tensor):
            t = self.data > other.data
            out = Tensor(
                t, _prev=(self, other), _op="gt", label=f"{self.label} > {other.label}"
            )
        elif isinstance(other, (int, float)):
            t = self.data > other
            out = Tensor(t, _prev=(self,), _op="gt", label=f"{self.label} > {other}")
        else:
            raise TypeError("The input must be a tensor or a number")

        def _backward():  # Not differentiable
            pass

        out._backward = _backward
        return out

    def numpy(self):
        """
        Convert the tensor to a numpy array
        """
        return self.data

    def phi(self):
        """
        Compute the cumulative distribution function of the tensor
        """
        t = 0.5 * (1 + erf(self.data / np.sqrt(2)))
        out = Tensor(t, _prev=(self,), _op="phi", label=f"phi({self.label})")

        def _backward():
            self.grad += (
                out.grad * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * self.data**2)
            )

        out._backward = _backward
        return out

    def split(self, num_splits, dim=-1):
        """
        Split the tensor
        """
        assert (
            self.data.shape[dim] % num_splits == 0
        ), "The dimension to split must be divisible by the number of splits"
        t = np.split(self.data, num_splits, axis=dim)
        out = [
            Tensor(x, _prev=(self,), _op="split", label=f"split({self.label})")
            for x in t
        ]

        for i, o in enumerate(out):

            def _backward():
                slices = [slice(None)] * self.data.ndim
                start = i * o.shape[dim]
                end = (i + 1) * o.shape[dim]
                slices[dim] = slice(start, end)
                self.grad[tuple(slices)] += o.grad

            o._backward = _backward
        return out

    def masked_fill(self, mask, value):
        """
        Fill the tensor using a mask
        """
        t = np.copy(self.data)
        t[mask] = value
        out = Tensor(
            t, _prev=(self,), _op="masked_fill", label=f"masked_fill({self.label})"
        )

        def _backward():
            self.grad += out.grad

        out._backward = _backward
        return out

    def softmax(self, axis=None):
        """
        Compute the softmax of the tensor
        """
        t = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True)).astype(
            np.float32
        )  # Numerically stable
        t = t / np.sum(t, axis=axis, keepdims=True)
        out = Tensor(t, _prev=(self,), _op="softmax", label=f"softmax({self.label})")

        def _backward():  # Softmax derivative
            classes = self.data.shape[-1]
            diag_p = np.eye(classes).astype(np.float32) * t.copy().reshape(
                -1, classes, 1
            )  # (classes, classes) * (batch, classes, 1) - (batch, classes, classes)
            p_p = t.copy().reshape(-1, classes, 1) * t.copy().reshape(
                -1, 1, classes
            )  # (batch, classes, 1) * (batch, 1, classes)
            total_p = diag_p - p_p
            self.grad += out.grad * np.sum(total_p, axis=-1)

        out._backward = _backward
        return out

    def unsqueeze(self, dim):
        """
        Unsqueeze the tensor
        """
        t = np.expand_dims(self.data, axis=dim)
        out = Tensor(
            t, _prev=(self,), _op="unsqueeze", label=f"unsqueeze({self.label})"
        )

        def _backward():
            self.grad += np.sum(out.grad, axis=dim)

        out._backward = _backward
        return out

    def squeeze(self, dim):
        """
        Squeeze the tensor
        """
        t = np.squeeze(self.data, axis=dim)
        out = Tensor(t, _prev=(self,), _op="squeeze", label=f"squeeze({self.label})")

        def _backward():
            self.grad += np.expand_dims(out.grad, axis=dim)

        out._backward = _backward
        return out

    def transpose(self, dim0, dim1):
        """
        Transpose the tensor
        """
        t = np.swapaxes(self.data, dim0, dim1)
        out = Tensor(
            t, _prev=(self,), _op="transpose", label=f"transpose({self.label})"
        )

        def _backward():
            self.grad += np.transpose(out.grad, (dim1, dim0))

        out._backward = _backward
        return out

    def conv2d(self, weight, stride=1):
        """
        Perform the 2D convolution operation
        """
        N, C, H, W = (
            self.data.shape
        )  # N - batch size, C - number of channels, H - height, W - width
        F, _, Kh, Kw = (
            weight.data.shape
        )  # F - number of filters, Kh - kernel height, Kw - kernel width
        H_out = (
            H - Kh
        ) // stride + 1  # Padding is implicit as the input is already padded
        W_out = (
            W - Kw
        ) // stride + 1  # Padding is implicit as the input is already padded
        t = np.zeros((N, F, H_out, W_out))
        # print(self.data.shape, weight.data.shape)
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                h_end = h_start + Kh
                w_start = j * stride
                w_end = w_start + Kw
                x_slice = self.data[:, :, h_start:h_end, w_start:w_end]
                for f in range(F):
                    # print(x_slice.shape, weight.data[f].shape)
                    t[:, f, i, j] = np.sum(x_slice * weight.data[f], axis=(1, 2, 3))

        out = Tensor(
            t,
            _prev=(self, weight),
            _op="conv2d",
            label=f"conv2d({self.label}, {weight.label})",
        )

        def _backward():
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + Kh
                    w_start = j * stride
                    w_end = w_start + Kw
                    inp_region = self.data[:, :, h_start:h_end, w_start:w_end]
                    for f in range(F):
                        weight.grad[f] += np.sum(
                            inp_region * out.grad[:, f, i, j][:, None, None, None],
                            axis=0,
                        )  # Get the gradient of the weight for the current filter for all elements in the batch
                        self.grad[:, :, h_start:h_end, w_start:w_end] += (
                            weight.data[f] * out.grad[:, f, i, j][:, None, None, None]
                        )  # Get the gradient of the input for the current filter for all elements in the batch

        out._backward = _backward
        return out

    def pad(self, pad_width, mode="zeros"):
        """
        Pad the tensor
        """
        if mode == "zeros":
            t = np.pad(self.data, pad_width, mode="constant", constant_values=0)
        elif mode == "reflect":
            t = np.pad(self.data, pad_width, mode="reflect")
        elif mode == "replicate":
            t = np.pad(self.data, pad_width, mode="edge")
        elif mode == "circular":
            t = np.pad(self.data, pad_width, mode="wrap")

        out = Tensor(t, _prev=(self,), _op="pad", label=f"pad({self.label})")

        def _backward():
            self.grad += out.grad[
                tuple([slice(p[0], -p[1] if p[1] != 0 else None) for p in pad_width])
            ]

        out._backward = _backward
        return out

    def max_pool2d(self, kernel_size, stride):
        """
        Perform the 2D max pooling operation
        """
        N, C, H, W = (
            self.data.shape
        )  # N - batch size, C - number of channels, H - height, W - width
        Kh, Kw = kernel_size, kernel_size  # Kh - kernel height, Kw - kernel width
        H_out = (H - Kh) // stride + 1
        W_out = (W - Kw) // stride + 1
        t = np.zeros((N, C, H_out, W_out))  # Initialize the output tensor

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                h_end = h_start + Kh
                w_start = j * stride
                w_end = w_start + Kw
                x_slice = self.data[:, :, h_start:h_end, w_start:w_end]
                t[:, :, i, j] = np.max(x_slice, axis=(2, 3))
        out = Tensor(
            t, _prev=(self,), _op="max_pool2d", label=f"max_pool2d({self.label})"
        )

        def _backward():
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + Kh
                    w_start = j * stride
                    w_end = w_start + Kw
                    x_slice = self.data[:, :, h_start:h_end, w_start:w_end]
                    mask = x_slice == np.max(x_slice, axis=(2, 3), keepdims=True)
                    self.grad[:, :, h_start:h_end, w_start:w_end] += (
                        mask * out.grad[:, :, i, j][:, :, None, None]
                    )

        out._backward = _backward
        return out

    def avg_pool2d(self, kernel_size, stride):
        """
        Perform the 2D average pooling operation
        """
        N, C, H, W = self.data.shape
        Kh, Kw = kernel_size, kernel_size
        H_out = (H - Kh) // stride + 1
        W_out = (W - Kw) // stride + 1
        t = np.zeros((N, C, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                h_end = h_start + Kh
                w_start = j * stride
                w_end = w_start + Kw
                x_slice = self.data[:, :, h_start:h_end, w_start:w_end]
                t[:, :, i, j] = np.mean(x_slice, axis=(2, 3))
        out = Tensor(
            t, _prev=(self,), _op="avg_pool2d", label=f"avg_pool2d({self.label})"
        )

        def _backward():
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + Kh
                    w_start = j * stride
                    w_end = w_start + Kw
                    self.grad[:, :, h_start:h_end, w_start:w_end] += out.grad[
                        :, :, i, j
                    ][:, :, None, None] / (Kh * Kw)

        out._backward = _backward
        return out

    def flip(self, axis):
        """
        Flip the tensor
        """
        t = np.flip(self.data, axis=axis)
        out = Tensor(t, _prev=(self,), _op="flip", label=f"flip({self.label})")

        def _backward():
            self.grad += np.flip(out.grad, axis=axis)

        out._backward = _backward
        return out

    def cat(tensors, axis=0):
        """
        Concatenate tensors
        """
        t = np.concatenate([x.data for x in tensors], axis=axis)
        out = Tensor(
            t, _prev=tensors, _op="cat", label=f"cat({[x.label for x in tensors]})"
        )

        def _backward():
            start = 0
            for x in tensors:  # Iterate over the tensors
                end = start + x.data.shape[axis]  # Get the range of the current tensor
                x.grad += out.grad.take(
                    range(start, end), axis=axis
                )  # Add the gradient of the current tensor
                start = end  # Update the start index for the next tensor

        out._backward = _backward
        return out

    def clamp(self, min_val, max_val):
        """
        Clamp the tensor
        """
        assert (
            min_val <= max_val
        ), "The minimum value must be less than or equal to the maximum value"
        t = np.clip(self.data, min_val, max_val)
        out = Tensor(t, _prev=(self,), _op="clamp", label=f"clamp({self.label})")

        def _backward():
            self.grad += (
                (min_val <= self.data) * (self.data <= max_val) * out.grad
            )  # The gradient is back propagated only if the value is within the range.

        out._backward = _backward
        return out

    def one_hot(self, num_classes):
        """
        Perform one hot encoding
        """
        t = np.eye(num_classes)[
            self.data.astype(np.int32).flatten()
        ]  # Flatten the tensor to prevent an extra dimension to appear
        out = Tensor(t, _prev=(self,), _op="one_hot", label=f"one_hot({self.label})")

        def _backward():
            self.grad += out.grad

        out._backward = _backward
        return out

    def abs(self):
        """
        Compute the absolute value of the tensor
        """
        t = np.abs(self.data)
        out = Tensor(t, _prev=(self,), _op="abs", label=f"abs({self.label})")

        def _backward():
            self.grad += np.sign(self.data) * out.grad

        out._backward = _backward
        return out

    def detach(self):
        """
        Detach the tensor
        """
        out = Tensor(self.data, _prev=(), _op="detach", label=f"detach({self.label})")

        def _backward():
            pass

        out._backward = _backward
        return out

    def maximum(self, value: float):
        """
        Compute the maximum of the tensor and a value
        """
        t = np.maximum(self.data, value)
        out = Tensor(
            t, _prev=(self,), _op="maximum", label=f"maximum({self.label}, {value})"
        )

        def _backward():
            self.grad += (self.data >= value) * out.grad

        out._backward = _backward
        return out
