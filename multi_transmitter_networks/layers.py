import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTransmitterFeedForwardLayer(nn.Module):
    """
    Implements a Multi-Trasmitter Feedforward Layer, which is the extension of a regular linear dense layer.

    In this layer, each neuron computes ``num_neurotransmitters`` linear transformations, each with their own weight matrix. Each linear transformation can be thought of the resulting expression of a neurotransmitter given a certain stimuli (input).

    Each neurotransmitter expression, or linear transformation, is collapsed into a single output value by another linear transformation.

    This linear transformation is the neuromodulator mechanism, which attributes weights for each neurotransmitter expression, given the input. This can be thought of as neuromodulators that inhibit or excite the receptor response. In this case, it excites or inhibits the presence of each neurotransmitter that the ensuing layers will receive.

    In fact, the previous can be loosely thought of as a releasing mechanism, where a single-release neuron only releases one neurotransmitter at a time, while co-release happens when two or more neurotransmitter can be released at each time.

    Currently, this layer contemplates co-release only.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        num_neurotransmitters: number of neurotransmitters composing each neuron response
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        co_release: If set to ``True``, then multiple neurotransmitters can compose the output signal, otherwise only one neurotransmitter. Only co-release (``True``) is implemented.
            Default: ``True``.

    Shape:
        - Input: (*, in_features), where * means any number of dimensions including none.
        - Output: (*, out_features), where * means any number of dimensions including none.

    Attributes:
        weight: learnable weights of the neurotransmitter response. Shape is (num_neurotransmitters, out_features, in_features).
        bias: learnable bias of the neurotransmitter response. Shape is (num_neurotransmitters, out_features).
        modulators_weight: learnable weights of the neuromodulator system. Shape is (out_features, num_neurotransmitters, in_features).
        modulators_bias: learnable bias of the neuromodulator system. Shape is (out_features, num_neurotransmitters).

    Examples::

        >>> m = MultiTransmitterFeedForwardLayer(20, 30, 3)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_neurotransmitters: int,
        bias: bool = True,
        co_release: bool = True,
    ) -> None:
        super().__init__()

        if not co_release:
            raise NotImplementedError("Only co-release is implemented at this moment.")

        self.in_features = in_features
        self.out_features = out_features
        self.num_neurotransmitters = num_neurotransmitters
        self.co_release = co_release

        self.weight = nn.Parameter(
            torch.empty((num_neurotransmitters, out_features, in_features))
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(num_neurotransmitters, out_features))
        else:
            self.register_parameter("bias", None)

        # If the signal is only composed by 1 neurotransmitter
        # Then no need to have a modulating system
        if num_neurotransmitters > 1:
            self.modulators_weight = nn.Parameter(
                torch.empty((out_features, num_neurotransmitters, in_features))
            )
        else:
            self.register_parameter("modulators_weight", None)

        if num_neurotransmitters > 1 and bias:
            self.modulators_bias = nn.Parameter(
                torch.empty(out_features, num_neurotransmitters)
            )
        else:
            self.register_parameter("modulators_bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.modulators_weight is not None:
            nn.init.kaiming_uniform_(self.modulators_weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if self.modulators_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.modulators_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.modulators_bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # only calculate modulators if actually needed
        if self.modulators_weight is not None:
            # Given tensor x with shape (i,j) where i = batch size and j = input size
            # and the weights for the neuromodulators which is a tensor (l,k,j)
            # where l = out features, k = number of neurotransmitters and j = input size
            # calculate the forward pass through a tensor contraction, iteratively multiplying
            # and summing over the j dimension
            modulators = torch.einsum("ij,lkj->ilk", x, self.modulators_weight)
            if self.modulators_bias is not None:
                modulators += self.modulators_bias
            modulators = F.softmax(modulators, dim=-1)

        # The same tensor contraction operation here, given tensor x, same shape as before
        # and the neurotransmitter weights, a tensor with shape (l,k,j) where l = number of neurotransmitters
        # k = out features and j = in features
        # iteratively multiply and sum over j dimension.
        x = torch.einsum("ij,lkj->ilk", x, self.weight)
        if self.bias is not None:
            x += self.bias

        if self.modulators_weight is None:
            # if we only have 1 neuromodulator, then we don't need to multiply by the weights
            # given by the neuromodulators, so we reshape x which has a shape of
            # (batch_size, 1, out_features) to (batch_size, out_features)
            x = x.reshape((x.size(0), x.size(-1)))
        else:
            # or we multiply and sum the neuromdulator weights with the neurotransmitter expression
            # such that we obtain 1 output per neuron
            # so we go from (batch_size, number_of_neurotransmitters, out_features)
            # to (batch_size, out_features)
            x = torch.einsum("ijk,ikj->ij", modulators, x)

        return x

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, num_neurotransmitters={}, bias={}, co_release={}".format(
            self.in_features,
            self.out_features,
            self.num_neurotransmitters,
            self.bias is not None,
            self.co_release,
        )
