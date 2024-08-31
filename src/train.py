import time
import logging
import torch
import torch.optim as optim
import signal
import os
import multiprocessing
from tensorboardX import SummaryWriter

from gru import Seq2SeqGRU
from loss import MaskedSoftmaxCELoss
from config import run_name, en_seq_len, zh_seq_len, epochs, batch_size, lr
from dataset import WMT_Dataset_en2zh
from init_params import get_params
from log import init_main_logging, log_queue


def terminate(signal_number, frame):
    print(f"Main process {os.getpid()} received signal {signal_number}, terminating child processes...")
    for p in multiprocessing.active_children():
        p.terminate()
    print("All child processes terminated.")
    exit(0)


def main():
    global run_name, en_seq_len, zh_seq_len, epochs, batch_size, lr
    
    if torch.cuda.is_available():
        cuda = "cuda:1"
        device = torch.device(cuda)
        print(f'Use device: {cuda}')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f'Use device: mps')
    else:
        print("GPU or MPS not found, exit.")
        exit()
    
    init_main_logging(log_queue)
    logger = logging.getLogger('main')
    
    writer = SummaryWriter(f'runs/{run_name}')

    data_params, net_params = get_params()
    dataset = WMT_Dataset_en2zh(**data_params)
    net = Seq2SeqGRU(**net_params).to(device)
    loss_fn = MaskedSoftmaxCELoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr) # type: ignore

    net.train()

    try:
        for epoch in range(epochs):
            run_loss = 0.0
            step = 0
            last_time = None
            logger.info(f'Epoch {epoch} start')
            
            for zh_input, en_input, zh_target, zh_valid_lens, en_valid_lens in dataset:
                zh_input = zh_input.to(device)
                en_input = en_input.to(device)
                zh_target = zh_target.to(device)
                zh_valid_lens = zh_valid_lens.to(device)
                
                optimizer.zero_grad()
                Y = net(en_input, zh_input)
                # Y.shape = (batch_size, seq_len, target_vocab_size)
                
                loss = loss_fn(Y, zh_target, zh_valid_lens)
                loss.sum().backward()
                optimizer.step()
                
                run_loss += loss.mean().item()

                if step % 50 == 0:
                    if last_time is not None:
                        token_num = 200 * batch_size * (zh_seq_len + en_seq_len)
                        writer.add_scalar(f'tokens/s_epoch{epoch}', token_num / (time.time() - last_time), step)
                    last_time = time.time()
                    
                    Y_idx = Y.argmax(dim=2)
                    # Y_idx.shape = (batch_size, seq_len)
                    
                    writer.add_scalar(f'loss_epoch{epoch}', loss.mean().item(), step)
                    writer.add_text(f'encoder_input [epoch{epoch}]', dataset.en_vocab.idx_to_str(en_input[0].tolist()), step)
                    # writer.add_text(f'decoder_input [epoch{epoch}]', dataset.zh_vocab.idx_to_str(zh_input[0].tolist()), step)
                    writer.add_text(f'target [epoch{epoch}]', dataset.zh_vocab.idx_to_str(zh_target[0].tolist()), step)
                    writer.add_text(f'predict [epoch{epoch}]', dataset.zh_vocab.idx_to_str(Y_idx[0].tolist()), step)
                    writer.flush()

                step += 1
            
            logger.info(f'Epoch {epoch} finished, loss: {run_loss / step}')
            
            torch.save(net.state_dict(), f'runs/{run_name}/epoch{epoch}.pth')
            logger.info(f'Epoch {epoch} model saved')
            
            writer.add_scalar(f'epoch loss', run_loss / step, epoch)
            writer.add_scalar(f'epoch lr', lr, epoch)
            writer.flush()
            
            lr = lr * 0.9
            
        logger.info('Finished training')
        
        logger.info('waiting for dataset to join...')
        dataset.join()
        logger.info('main process finished')
    except Exception as e:
        logger.error(e)
    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt')
    finally:
        terminate(signal.SIGINT, None)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)
    main()
