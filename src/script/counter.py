from dotenv import load_dotenv
import os
from collections import Counter
from tqdm import tqdm
import html
from multiprocessing import Pool
import pickle
from dataset import zh_to_token, en_to_token, WMT_DatasetChunk


def process_chunk(data):
    zh_lines, en_lines = data
    zh = zh_to_token(' '.join(zh_lines))
    en = en_to_token(html.unescape(' '.join(en_lines)))
    
    return Counter(zh), Counter(en), len(zh_lines)


def main():
    batch_size = 100000
    dataset = WMT_DatasetChunk(os.getenv('CSV_PATH', ''), batch_size)

    line_num = 0
    zh_counter = Counter()
    en_counter = Counter()
    
    max_workers = 4

    with Pool(processes=max_workers) as pool:
        jobs = []
        
        with tqdm(total=dataset.tot_lines) as pbar:
            
            def update_res(res):
                zh, en, num = res
                zh_counter.update(zh)
                en_counter.update(en)
                nonlocal line_num
                line_num += num
                pbar.update(num)
                
            for data in dataset:
                if len(jobs) > max_workers * 2:
                    for job in jobs:
                        if job.ready():
                            update_res(job.get())
                            jobs.remove(job)
                    
                jobs.append(pool.apply_async(process_chunk, (data,)))
                
            for job in jobs:
                update_res(job.get())

    print(f'line_num: {line_num}')
    print('================== zh ==================')
    print(len(zh_counter))
    print(zh_counter.most_common(100))
    print('================== en ==================')
    print(len(en_counter))
    print(en_counter.most_common(100))
    
    data_dict = {
        'zh_counter': zh_counter,
        'en_counter': en_counter,
        'line_num': line_num
    }
    with open('data.pkl', 'wb') as f:
        pickle.dump(data_dict, f)


if __name__ == '__main__':
    load_dotenv()
    main()
