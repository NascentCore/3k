from model.basic_transformer import *
from utils.train_utils import *
from utils.model_config import model_config

def train_model():
    V = model_config['vocab_size']
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = make_model(V,V,N=model_config['num_layers'],
                       d_model=model_config['d_model'], 
                       d_ff=model_config['d_ff'], 
                       h=model_config['n_head'], 
                       dropout=model_config['dropout']).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        )
    )
    batch_size = 4
    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, model_config['train_length'], 20, device),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, model_config['train_length'], 5, device),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]
    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).to(device=device)
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len).to(device=device)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))

    # save model
    torch.save(model.state_dict(), './train_out/model.pth')
    torch.save(optimizer.state_dict(), './train_out/optimizer.pth')



if __name__ == '__main__':
    train_model()