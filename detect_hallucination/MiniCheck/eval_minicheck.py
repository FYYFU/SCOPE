from minicheck.minicheck import MiniCheck
import os
import json
import ipdb
import nltk
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=str)
    parser.add_argument('--dataset', default='gov_report', type=str)
    parser.add_argument('--method', default='SnapKV', type=str)
    parser.add_argument('--kv_size', default=64, type=int)
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    scorer = MiniCheck(model_name='Bespoke-MiniCheck-7B', tensor_parallel_size=1, cache_dir='./ckpts', enable_prefix_caching=False)
    data_path = f'/home/greenland-user/SCOPE/results/llama-3.1-8b-instruct_{args.kv_size}/{args.dataset}/{args.method}.json'

    total_probs = []
    total_data = []
    with open(data_path, 'r') as f:
        for line in f.readlines():
            d = json.loads(line.strip())
            total_data.append(d)
    index = 0

    for d in tqdm(total_data, total=len(total_data), desc='Eval'):
        claims = nltk.sent_tokenize(d['pred'])
        doc = d['context']
        pred_label, raw_prob, _, _ = scorer.score(docs=[doc for i in range(len(claims))], claims=claims)
        final_prob = sum(raw_prob) / len(raw_prob)
        total_probs.append(final_prob)
        print(f'index: {index}. Score: {final_prob}')
        index += 1

    avg_prob = sum(total_probs) / len(total_probs)
    print(f'data path: {data_path}')
    print(f'average prob: {avg_prob}')

    # # Alternatively, you can use our Bespoke-MiniCheck-7B model (7B) for evaluation. 
    # # Bespoke-MiniCheck-7B is the most performant fact-checking model 
    # # in the MiniCheck series AND is the current SOTA regardless of size.
    # # It's also commercially useable! 
    # # For commercial licensing, please contact company@bespokelabs.ai
    # scorer = MiniCheck(model_name='Bespoke-MiniCheck-7B', enable_prefix_caching=False, cache_dir='./ckpts')
    # pred_label, raw_prob, _, _ = scorer.score(docs=[doc, doc], claims=[claim_1, claim_2])

    # print(pred_label) # [1, 0]
    # print(raw_prob)   # [0.9840446675150499, 0.010986349594852094]