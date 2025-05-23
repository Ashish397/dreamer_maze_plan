import numpy as np
from typing import Dict, List
import gymnasium as gym
import copy 

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import (
    normc_initializer,
    same_padding,
    SlimConv2d,
    SlimFC,
)
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

torch, nn = try_import_torch()


class VisionNetwork(TorchModelV2, nn.Module):
    """Generic vision network."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        if not model_config.get("conv_filters"):
            model_config["conv_filters"] = get_filter_config(obs_space.shape)

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        activation = self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"

        # Post FC net config.
        post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        post_fcnet_activation = get_activation_fn(
            model_config.get("post_fcnet_activation"), framework="torch"
        )

        no_final_linear = self.model_config.get("no_final_linear")
        vf_share_layers = self.model_config.get("vf_share_layers")

        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        self.last_layer_is_flattened = False
        self._logits = None

        layers = []
        (w, h, in_channels) = obs_space.shape

        in_size = [w, h]
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = same_padding(in_size, kernel, stride)
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    activation_fn=activation,
                )
            )
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = filters[-1]

        # No final linear: Last layer has activation function and exits with
        # num_outputs nodes (this could be a 1x1 conv or a FC layer, depending
        # on `post_fcnet_...` settings).
        if no_final_linear and num_outputs:
            out_channels = out_channels if post_fcnet_hiddens else num_outputs
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    None,  # padding=valid
                    activation_fn=activation,
                )
            )

            # Add (optional) post-fc-stack after last Conv2D layer.
            layer_sizes = post_fcnet_hiddens[:-1] + (
                [num_outputs] if post_fcnet_hiddens else []
            )
            for i, out_size in enumerate(layer_sizes):
                layers.append(
                    SlimFC(
                        in_size=out_channels,
                        out_size=out_size,
                        activation_fn=post_fcnet_activation,
                        initializer=normc_initializer(1.0),
                    )
                )
                out_channels = out_size

        # Finish network normally (w/o overriding last layer size with
        # `num_outputs`), then add another linear one of size `num_outputs`.
        else:
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    None,  # padding=valid
                    activation_fn=activation,
                )
            )

            # num_outputs defined. Use that to create an exact
            # `num_output`-sized (1,1)-Conv2D.
            if num_outputs:
                in_size = [
                    np.ceil((in_size[0] - kernel[0]) / stride),
                    np.ceil((in_size[1] - kernel[1]) / stride),
                ]
                padding, _ = same_padding(in_size, [1, 1], [1, 1])
                if post_fcnet_hiddens:
                    layers.append(nn.Flatten())
                    in_size = out_channels
                    # Add (optional) post-fc-stack after last Conv2D layer.
                    for i, out_size in enumerate(post_fcnet_hiddens + [num_outputs]):
                        layers.append(
                            SlimFC(
                                in_size=in_size,
                                out_size=out_size,
                                activation_fn=post_fcnet_activation
                                if i < len(post_fcnet_hiddens) - 1
                                else None,
                                initializer=normc_initializer(1.0),
                            )
                        )
                        in_size = out_size
                    # Last layer is logits layer.
                    self._logits = layers.pop()

                else:
                    self._logits = SlimConv2d(
                        out_channels,
                        num_outputs,
                        [1, 1],
                        1,
                        padding,
                        activation_fn=None,
                    )

            # num_outputs not known -> Flatten, then set self.num_outputs
            # to the resulting number of nodes.
            else:
                self.last_layer_is_flattened = True
                layers.append(nn.Flatten())

        self._convs_left = nn.Sequential(*layers)
        self._convs_right = nn.Sequential(*copy.deepcopy(layers))  # Now distinct layers

        # If our num_outputs still unknown, we need to do a test pass to
        # figure out the output dimensions. This could be the case, if we have
        # the Flatten layer at the end.
        if self.num_outputs is None:
            # Create a B=1 dummy sample and push it through out conv-net.
            dummy_in = (
                torch.from_numpy(self.obs_space.sample())
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
            )
            W = dummy_in.shape[-1]
            left = dummy_in[:, :, :, :W // 2]
            right = dummy_in[:, :, :, W // 2:]

            print(f"[DEBUG] Dummy input shape: {dummy_in.shape}")          # e.g., [1, 3, 64, 128]
            print(f"[DEBUG] Dummy left shape: {left.shape}, right: {right.shape}")

            left_out = self._convs_left(left)
            right_out = self._convs_right(right)

            print(f"[DEBUG] Left conv output: {left_out.shape}, Right conv output: {right_out.shape}")

            dummy_out = left_out + right_out
            print(f"[DEBUG] Combined dummy conv output shape: {dummy_out.shape}")

            self.num_outputs = dummy_out.shape[1]

        # Build the value layers
        self._value_branch_separate = self._value_branch = None
        if vf_share_layers:
            self._value_branch = SlimFC(
                out_channels, 1, initializer=normc_initializer(0.01), activation_fn=None
            )
        else:
            vf_layers_left = []
            vf_layers_right = []

            (w, h, in_channels) = obs_space.shape
            in_size = [w, h // 2]  # half-width since we split left/right

            for out_channels, kernel, stride in filters[:-1]:
                padding, out_size = same_padding(in_size, kernel, stride)

                # Left stream
                vf_layers_left.append(
                    SlimConv2d(
                        in_channels,
                        out_channels,
                        kernel,
                        stride,
                        padding,
                        activation_fn=activation,
                    )
                )

                # Right stream
                vf_layers_right.append(
                    SlimConv2d(
                        in_channels,
                        out_channels,
                        kernel,
                        stride,
                        padding,
                        activation_fn=activation,
                    )
                )

                in_channels = out_channels
                in_size = out_size

            # Final conv layer (last one in filter stack)
            out_channels, kernel, stride = filters[-1]

            vf_layers_left.append(
                SlimConv2d(in_channels, out_channels, kernel, stride, None, activation_fn=activation)
            )
            vf_layers_right.append(
                SlimConv2d(in_channels, out_channels, kernel, stride, None, activation_fn=activation)
            )

            # Final 1x1 conv to output scalar value
            vf_value_head = SlimConv2d(
                in_channels=out_channels,
                out_channels=1,
                kernel=1,
                stride=1,
                padding=None,
                activation_fn=None,
            )

            # === Register modules ===
            self._vf_convs_left = nn.Sequential(*vf_layers_left)
            self._vf_convs_right = nn.Sequential(*vf_layers_right)
            self._vf_value_head = vf_value_head
            self._value_branch_separate = True  # just to signal use

        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self._features = input_dict["obs"].float()
        # Permuate b/c data comes in as [B, dim, dim, channels]:
        self._features = self._features.permute(0, 3, 1, 2)

        W = self._features.shape[-1]
        left = self._features[:, :, :, :W // 2].detach()
        right = self._features[:, :, :, W // 2:].detach()
        conv_out = self._convs_left(left) + self._convs_right(right)
        # Store features to save forward pass when getting value_function out.
        if not self._value_branch_separate:
            self._features = conv_out

        if not self.last_layer_is_flattened:
            if self._logits:
                conv_out = self._logits(conv_out)
            if len(conv_out.shape) == 4:
                if conv_out.shape[2] != 1 or conv_out.shape[3] != 1:
                    raise ValueError(
                        "Given `conv_filters` ({}) do not result in a [B, {} "
                        "(`num_outputs`), 1, 1] shape (but in {})! Please "
                        "adjust your Conv2D stack such that the last 2 dims "
                        "are both 1.".format(
                            self.model_config["conv_filters"],
                            self.num_outputs,
                            list(conv_out.shape),
                        )
                    )
                logits = conv_out.squeeze(3)
                logits = logits.squeeze(2)
            else:
                logits = conv_out
            del left, right, conv_out
            torch.cuda.empty_cache()
            return logits, state
        else:
            del left, right
            torch.cuda.empty_cache()
            return conv_out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            W = self._features.shape[-1]
            left = self._features[:, :, :, :W // 2].detach()
            right = self._features[:, :, :, W // 2:].detach()

            left_val = self._vf_convs_left(left)
            right_val = self._vf_convs_right(right)

            combined = 0.5 * (left_val + right_val)
            value = self._vf_value_head(combined)

            while value.dim() > 1:
                value = value.squeeze(-1)

            del left, right, left_val, right_val, combined
            torch.cuda.empty_cache()
            return value

        else:
            value = self._value_branch(self._features)
            while value.dim() > 1:
                value = value.squeeze(-1)
            return value

    def _hidden_layers(self, obs: TensorType) -> TensorType:
        obs = obs.permute(0, 3, 1, 2)
        W = obs.shape[-1]
        left = obs[:, :, :, :W // 2]
        right = obs[:, :, :, W // 2:]
        res = self._convs_left(left) + self._convs_right(right)
        res = res.squeeze(3)
        res = res.squeeze(2)
        return res
