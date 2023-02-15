
import torch

from whisper.model import MultiHeadAttention

n_state = 512
n_head = 8

model = MultiHeadAttention(n_state, n_head)
input = torch.zeros(1, n_head, n_state)

torch.onnx.export(
    model.cpu(),  # --dynamic only compatible with cpu
    input.cpu(),
    'MultiHeadAttention.onnx',
    verbose=False,
    opset_version=12,
    do_constant_folding=True,
    input_names=['mel'],
    output_names=['wv', 'qk'],
    dynamic_axes=None)

