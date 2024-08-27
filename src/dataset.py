import pandas as pd
from dotenv import load_dotenv
import os
from collections import Counter
import re
from tqdm import tqdm
import html
from multiprocessing import Pool


def process_chunk(chunk):
    zh_counter = Counter()
    en_counter = Counter()
    
    zh = ' '.join(chunk.iloc[:, 0].astype(str).tolist())
    en = ' '.join(chunk.iloc[:, 1].astype(str).tolist())
    
    zh = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]|[，。！？、；：“”（）《》〈〉—…‘’“”""\'\-—_/\[\]{}<>`~@#$%^&*+=|\\]', zh)
    zh_counter.update(zh)
    
    en = html.unescape(en)
    en = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|[0-9]+|[.,!?;:\"\'\-(){}[\]<>`~#$%^&*+=|\\/_]", en)
    en_counter.update(en)
    
    return zh_counter, en_counter, chunk.shape[0]


def main():
    load_dotenv()

    total_lines = 2.5e7
    chunk_size = 100000
    data = pd.read_csv(os.getenv('CSV_PATH', ''), chunksize=chunk_size)

    line_num = 0
    zh_counter = Counter()
    en_counter = Counter()
    
    max_workers = 4

    with Pool(processes=max_workers) as pool:
        jobs = []
        
        with tqdm(total=total_lines) as pbar:
            
            def update_res(res):
                zh, en, num = res
                zh_counter.update(zh)
                en_counter.update(en)
                nonlocal line_num
                line_num += num
                pbar.update(num)
                
            for chunk in data:
                if len(jobs) > max_workers * 2:
                    for job in jobs:
                        if job.ready():
                            update_res(job.get())
                            jobs.remove(job)
                    
                jobs.append(pool.apply_async(process_chunk, (chunk,)))
                
            for job in jobs:
                update_res(job.get())

    print(f'line_num: {line_num}')
    print('================== zh ==================')
    print(len(zh_counter))
    print(zh_counter.most_common(100))
    print('================== en ==================')
    print(len(en_counter))
    print(en_counter.most_common(100))


if __name__ == '__main__':
    main()
