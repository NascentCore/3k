import onnxruntime
import numpy as np
import torch
from model.basic_transformer import *

def test_inference():
    session = onnxruntime.InferenceSession('./train_out/model.onnx')

    trace_input = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = trace_input.shape[1]
    trace_input_mask = torch.ones(1, 1, max_len)


    start_symbol=0
    ys = torch.zeros(1, 4).fill_(start_symbol).type_as(trace_input.data)
    ys_mask = subsequent_mask(ys.size(1)).type_as(trace_input.data)[0,-1].reshape((ys.shape[0],1,ys.shape[1]))

    trace_input = trace_input.numpy()
    trace_input_mask = trace_input_mask.numpy()
    ys = ys.numpy()
    ys_mask = ys_mask.numpy()

    output = session.run(None, {'x':trace_input, 'y': ys, 'x_mask': trace_input_mask, 'y_mask': ys_mask})
    print(output)

test_inference()