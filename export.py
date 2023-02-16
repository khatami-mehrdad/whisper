
import torch

from whisper.model import MultiHeadAttention

n_state = 512
n_head = 8

model = MultiHeadAttention(n_state, n_head)
input = torch.zeros(1, n_head, n_state)

f = 'MultiHeadAttention_Sim.onnx'

torch.onnx.export(
    model.cpu(),  # --dynamic only compatible with cpu
    input.cpu(),
    f,
    verbose=False,
    opset_version=12,
    do_constant_folding=True,
    input_names=['mel'],
    output_names=['wv', 'qk'],
    dynamic_axes=None)

# Checks
import onnx
model_onnx = onnx.load(f)  # load onnx model
onnx.checker.check_model(model_onnx)  # check onnx model

# simplify
import onnxsim

model_onnx, check = onnxsim.simplify(model_onnx)
assert check, 'assert check failed'
onnx.save(model_onnx, f)

# onnxruntime
import onnxruntime as rt

sess_options = rt.SessionOptions()

# Set graph optimization level
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

# To enable model serialization after graph optimization set this
sess_options.optimized_model_filepath = "MultiHeadAttention_Sim_Opt.onnx"

session = rt.InferenceSession(f, sess_options)
