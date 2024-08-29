import os
from dotenv import load_dotenv
import time
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from gru import Seq2SeqGRU
from dataset import WMT_Dataset_en2zh
from loss import MaskedSoftmaxCELoss

run_name = 'test_mac'

en_seq_len = 30
zh_seq_len = 45

epochs = 10
batch_size = 12
lr = 0.002


def main():
    if torch.cuda.is_available():
        cuda = "cuda:2"
        device = torch.device(cuda)
        print(f'Use device: {cuda}')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f'Use device: mps')
    else:
        print("GPU or MPS not found, exit.")
        exit()
        
    load_dotenv()

    writer = SummaryWriter(f'runs/{run_name}')
    dataset = WMT_Dataset_en2zh(os.getenv('CSV_PATH', ''), zh_target_len=zh_seq_len, en_target_len=en_seq_len, batch_size=batch_size)
    params = {
        'source_vocab_size': len(dataset.en_vocab), 
        'target_vocab_size': len(dataset.zh_vocab), 
        'source_embed_size': 300, 
        'target_embed_size': 200, 
        'source_seq_len': en_seq_len, 
        'target_seq_len': zh_seq_len, 
        'num_hiddens': 256, 
        'num_layers': 2
    }

    net = Seq2SeqGRU(**params).to(device)
    loss_fn = MaskedSoftmaxCELoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr) # type: ignore

    net.train()

    for epoch in range(epochs):
        run_loss = 0.0
        step = 0
        last_time = None
        
        for zh_input, en_input, zh_target, zh_valid_lens, en_valid_lens, zh_tokens, en_tokens in dataset:
            zh_input = zh_input.to(device)
            en_input = en_input.to(device)
            zh_target = zh_target.to(device)
            zh_valid_lens = zh_valid_lens.to(device)
            en_valid_lens = en_valid_lens.to(device)
            
            optimizer.zero_grad()
            Y = net(en_input, zh_input)
            # Y.shape = (batch_size, seq_len, target_vocab_size)
            
            loss = loss_fn(Y, zh_target, zh_valid_lens)
            loss.sum().backward()
            optimizer.step()
            
            run_loss += loss.mean().item()

            if step % 100 == 0:
                if last_time is not None:
                    token_num = 100 * batch_size * (zh_seq_len + en_seq_len)
                    writer.add_scalar(f'tokens / sec [epoch{epoch}]', token_num / (time.time() - last_time), step)
                last_time = time.time()
                
                Y_idx = Y.argmax(dim=2)
                # Y_idx.shape = (batch_size, seq_len)
                
                writer.add_scalar(f'loss [epoch{epoch}]', loss.mean().item(), step)
                writer.add_text(f'encoder_input [epoch{epoch}]', dataset.en_vocab.idx_to_str(en_input[0].tolist()), step)
                writer.add_text(f'decoder_input [epoch{epoch}]', dataset.zh_vocab.idx_to_str(zh_input[0].tolist()), step)
                writer.add_text(f'target [epoch{epoch}]', dataset.zh_vocab.idx_to_str(zh_target[0].tolist()), step)
                writer.add_text(f'predict [epoch{epoch}]', dataset.zh_vocab.idx_to_str(Y_idx[0].tolist()), step)
                writer.flush()

            step += 1

if __name__ == '__main__':
    main()
