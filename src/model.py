# model.py
# neural network models and other algorithms
# Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
from collections import OrderedDict
from numpy.typing import ArrayLike, NDArray
from typing import *

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3,
                 input_ch_views=3, output_ch=4,
                 skips=None, use_viewdirs=False):
        """
        """
        super(NeRF, self).__init__()
        if skips is None:
            skips = [D//2] #[range(0, D-2, 2)] #[4]
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs: # False
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=3.):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                nn.init.uniform_(self.linear.weight, a=-1 / self.in_features, b=1 / self.in_features)
            else:
                nn.init.uniform_(self.linear.weight, a=-np.sqrt(6 / self.in_features) / self.omega_0,
                                 b=np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=3, hidden_omega_0=3.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                nn.init.uniform_(final_linear.weight, a=-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                 b=np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

class nerf2siren(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3,
                 input_ch_views=3, output_ch=4,
                 skips=None, use_viewdirs=False):
        super(nerf2siren, self).__init__()
        self.nerf = NeRF(D=D, W=W, input_ch=input_ch,
                         input_ch_views=input_ch_views,
                         output_ch=output_ch,
                         skips=skips, use_viewdirs=use_viewdirs)
        self.siren = Siren(in_features = input_ch, 
                           hidden_features = W, 
                           hidden_layers = D, 
                           out_features = output_ch, 
                           outermost_linear = True)
    def forward(self, inputs):
        outnerf = self.nerf.forward(inputs)
        outsiren = self.siren.forward(inputs)
        maxv = torch.max(outnerf) 
        mask = (outnerf < 0.5 * maxv)
        outnerf[mask] = outsiren[mask]
        return outnerf
        
    
class MonotonicDense(nn.Linear):
    """
    Monotonic counterpart of the regular Dense Layer of torch.nn
    Args:
        in_dim: Positive integer, dimensionality of the input space.
        out_dim: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
        indicator_vector: Vector to indicate which of the inputs are monotonically
            increasing or monotonically decreasing or non-monotonic. Has value 1 for
            monotonically increasing and -1 for monotonically decreasing and 0 for
            non-monotonic variables.
        convexity_indicator: If the value is 0 or 1, then all elements of the
            activation selector will be 0 or 1, respectively. If None, epsilon will be
            used to determine the number of 0 and 1 in the activation selector.
        epsilon: Percentage of elements with value 1 in the activation vector if
            convexity_indicator is None, ignored otherwise.
    """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 activation: Optional[Union[str, Callable]] = None,
                 indicator_vector: Optional[NDArray[np.int_]] = None,
                 convexity_indicator: Optional[int] = None,
                 epsilon: float = 0.5,
                 **kwargs):
        super(MonotonicDense, self).__init__(in_dim, out_dim, **kwargs)
        self.org_activation = activation
        self.out_dim = out_dim
        self.indicator_vector = (
            np.array(indicator_vector, dtype="int32")
            if indicator_vector is not None
            else None
        )
        self.convexity_indicator = convexity_indicator
        self.epsilon = epsilon
        if indicator_vector is not None:
            self.indicator_vector = torch.from_numpy(np.asarray(indicator_vector)).reshape(-1, 1)
        if convexity_indicator is not None:
            self.convexity_indicator = torch.from_numpy(np.asarray(convexity_indicator))

    def get_activation(self, name):
        if name == 'relu':
            return F.relu
        elif name == 'elu':
            return F.elu
        else:
            raise NotImplementedError

    def get_activation_selector(self, inputs) -> torch.Tensor:
        n_ones = int(round(self.out_dim * self.epsilon))
        n_zeros = self.out_dim - n_ones
        activation_selector = torch.cat((torch.ones((inputs.shape[0], n_ones), dtype=torch.bool),
                                         torch.zeros((inputs.shape[0], n_zeros), dtype=torch.bool)),
                                         dim=1)
        return activation_selector

    def replace_weight(self):
        """Replace kernel with absolute values of weights based on indicator vector
           This is implemented as context manager to ensure rollback functionality
           after the calculation is finished.
        """
        weight_params = nn.Parameter(self.weight)
        if self.indicator_vector is not None:
            abs_weight = torch.abs(weight_params)
            weight_1 = torch.where(self.indicator_vector == 1,
                                   abs_weight,
                                   weight_params)
            weight_updated = torch.where(self.indicator_vector == -1, -abs_weight, weight_1)
        else:
            weight_updated = torch.abs(weight_params)
        return weight_updated

    def forward(self, inputs):
        """
        calculate linear forward pass with weights set to either positive or
        negative values based on the value of indicator_vector
        """
        weight_updated = self.replace_weight()
        bias_params = nn.Parameter(self.bias)
        y = torch.mm(inputs, weight_updated.t()) + bias_params
        if self.org_activation is not None:
            activation = self.get_activation(self.org_activation)
            if self.convexity_indicator == 1:
                y = activation(y)
            elif self.convexity_indicator == -1:
                y = -activation(-y)
            else:
                activation_selector = self.get_activation_selector(inputs)
                y1 = activation(y)
                y2 = -activation(-y)
                y = torch.where(activation_selector, y1, y2)
        return y

def build_monotonic_type1_model(in_dim: int = 4,
                                mid_dim: int = 8,
                                out_dim: int = 1,
                                activation: str = 'relu',
                                n_layers: int = 2,
                                is_classification: bool = True,
                                indicator_vector = None,
                                epsilon: float = 0.5,
                                ):
    model = nn.ModuleList()
    model.append(MonotonicDense(in_dim=in_dim,
                                out_dim=mid_dim,
                                activation=activation,
                                indicator_vector=indicator_vector,
                                epsilon = epsilon))

    for _ in range(n_layers - 1):
        model.append(MonotonicDense(in_dim=mid_dim,
                                    out_dim=mid_dim,
                                    activation=activation,
                                    epsilon = epsilon))

    model.append(MonotonicDense(in_dim=mid_dim,
                                out_dim=out_dim,
                                activation=None,
                                epsilon = epsilon))

    if is_classification:
        model.append(nn.Sigmoid())
    return model

class build_monotonic_type2_model(nn.Module):
    """
    in_mono_dim: dimension for monotonic variables.
    cat_dim: dimension after concatenating monotonic & non-monotonic variables.
    out_dim: output dimension.
    activation: activation function, default = relu.
    n_layers: number of layers.
    is_classification: if true, then add a sigmoid function at the end of the network.
    indicator_vector: indicate which variable is monotonically increasing/decreasing.
    convexity_indicator: indicate if a function is convex.
    model2: neural network for non-monotonic variables.
    epsilon: fractions of ones in 's' (see paper).
    """
    def __init__(self,
                 in_mono_dim: int = 4,
                 cat_dim: int = 8,
                 out_dim: int = 1,
                 activation: str = "relu",
                 n_layers: int = 2,
                 is_classification: bool = False,
                 indicator_vector=None,
                 convexity_indicator=None,
                 model2=None,
                 epsilon: float = 0.5,
                 ):
        super().__init__()
        self.mono_dense = MonotonicDense(in_dim=in_mono_dim,
                                         out_dim=in_mono_dim,
                                         activation=activation,
                                         indicator_vector=indicator_vector,
                                         convexity_indicator=convexity_indicator,
                                         epsilon = epsilon)

        self.mono_dense_cat = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.mono_dense_cat.append(MonotonicDense(in_dim=cat_dim,
                                                      out_dim=cat_dim,
                                                      activation=activation,
                                                      epsilon = epsilon))

        self.mono_dense_cat.append(MonotonicDense(in_dim=cat_dim,
                                                  out_dim=out_dim,
                                                  activation=None,
                                                  epsilon = epsilon))

        self.is_classification = is_classification
        self.in_mono_dim = in_mono_dim
        self.model2 = model2
    def forward(self, x, z):
        out1 = self.mono_dense.forward(x)
        out2 = self.model2(z)
        out3 = torch.cat((out1, out2), dim=1)
        for i in range(len(self.mono_dense_cat)):
            out3 = self.mono_dense_cat[i].forward(out3)
        if self.is_classification:
            out3 = out3.sigmoid()
        return out3

if __name__ == "__main__":
    # model = NeRF(D=8, W=256, input_ch=60, output_ch=1, input_ch_views=0, use_viewdirs=False)
    # model = Siren(in_features = 3, hidden_features = 256, hidden_layers = 8, out_features = 1, outermost_linear = True)
    model2 = nn.Linear(6, 4)
    model = build_monotonic_type2_model(in_mono_dim=4,
                                        cat_dim=8,
                                        out_dim=1,
                                        indicator_vector = np.asarray([1,-1,1,-1]),
                                        model2=model2)
    x = torch.randn(10, 4)
    z = torch.randn(10, 6)
    y = model.forward(x, z)
    print('y: ', y)
    ytrue = torch.ones(10, 1)
    criterion = nn.MSELoss()
    loss = criterion(y, ytrue)
    print('loss: ', loss.item())
    loss.backward()
    # summary(model, (1, 10))
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Linear-1            [-1, 1, 1, 256]          15,616
#             Linear-2            [-1, 1, 1, 256]          65,792
#             Linear-3            [-1, 1, 1, 256]          65,792
#             Linear-4            [-1, 1, 1, 256]          65,792
#             Linear-5            [-1, 1, 1, 256]          65,792
#             Linear-6            [-1, 1, 1, 256]          81,152
#             Linear-7            [-1, 1, 1, 256]          65,792
#             Linear-8            [-1, 1, 1, 256]          65,792
#             Linear-9              [-1, 1, 1, 1]             257
# ================================================================
# Total params: 491,777
# Trainable params: 491,777
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.00
# Forward/backward pass size (MB): 0.02
# Params size (MB): 1.88
# Estimated Total Size (MB): 1.89
# ----------------------------------------------------------------
#
# Process finished with exit code 0
