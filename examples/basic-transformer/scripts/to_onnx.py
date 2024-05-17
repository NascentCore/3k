from model.basic_transformer import *
from utils.train_utils import *
from utils.model_config import model_config

def to_onnx():
    V = model_config['vocab_size']
    model = make_model(V,V,N=model_config['num_layers'],
                       d_model=model_config['d_model'], 
                       d_ff=model_config['d_ff'], 
                       h=model_config['n_head'], 
                       dropout=model_config['dropout'])
    # model.load_state_dict('./train_out/model.pth')

    model.eval()

    trace_input = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = trace_input.shape[1]
    trace_input_mask = torch.ones(1, 1, max_len)


    start_symbol=0
    ys = torch.zeros(1, 4).fill_(start_symbol).type_as(trace_input.data)
    ys_mask = subsequent_mask(ys.size(1)).type_as(trace_input.data)[0,-1].reshape((ys.shape[0],1,ys.shape[1]))
    print(trace_input.shape)
    print(trace_input_mask.shape)
    print(ys.shape)
    print(ys_mask.shape)
    
    # print(model.forward(trace_input, ys, trace_input_mask, ys_mask)[0].shape)
    # print(model.forward(trace_input, ys, trace_input_mask, ys_mask)[1].shape)
    out = model.forward(trace_input, ys, trace_input_mask, ys_mask)
    print(out.shape)
    torch.onnx.export(model, (trace_input, ys, trace_input_mask, ys_mask), './train_out/model.onnx', 
                      input_names=['x','y','x_mask','y_mask'],
                      dynamic_axes={'x':{0:'batch_size',1:'sentence_length'}, 'x_mask':{0:'batch_size', 2:'sentence_length'},
                                    'y':{0:'batch_size',1:'output_length'}, 'y_mask':{0:'batch_size', 2:'output_length'}})
    torch.onnx.export(model.generator, (out), './train_out/generator.onnx', input_names=['out'], dynamic_axes={'out':{0:'batch_size',1:'output_length'}})

to_onnx()