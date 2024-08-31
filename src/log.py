import logging
from logging.handlers import QueueHandler, QueueListener
import multiprocessing
from pathlib import Path

from config import run_name

log_queue = multiprocessing.Queue(-1)

def init_subprocess_logging(queue):
    print(f'Init subprocess logging')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d] - %(process)d',
        handlers=[
            QueueHandler(queue),
        ]
    )

def init_main_logging(queue):
    print(f'Init main logging')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d] - %(process)d',
        handlers=[
            QueueHandler(queue),
        ]
    )
    
    # create log folder
    Path(f'runs/{run_name}').mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(f'runs/{run_name}/logging.log')
    listener = QueueListener(log_queue, file_handler)
    listener.start()
