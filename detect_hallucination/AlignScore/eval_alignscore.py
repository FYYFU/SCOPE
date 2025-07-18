import os
import json
import ipdb
import nltk
from tqdm import tqdm
import argparse
from alignscore import AlignScore


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=str)
    parser.add_argument('--dataset', default='gov_report', type=str)
    parser.add_argument('--method', default='SnapKV', type=str)
    parser.add_argument('--kv_size', default=64, type=int)
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device="cuda:" + args.gpu
    scorer = AlignScore(model='roberta-large', batch_size=128, device=device, ckpt_path='./ckpts/AlignScore-large.ckpt', evaluation_mode='nli_sp')
    total_probs = []
    total_data = []

    data_path = f'/home/greenland-user/SCOPE/results/llama-3.1-8b-instruct_{args.kv_size}/{args.dataset}/{args.method}.json'

    with open(data_path, 'r') as f:
        for line in f.readlines():
            d = json.loads(line.strip())
            total_data.append(d)
    index = 0

    for d in tqdm(total_data, total=len(total_data), desc='Eval'):
        # claims = nltk.sent_tokenize(d['answers'][0])
        claims = nltk.sent_tokenize(d['pred'])
        doc = d['context']
        try:
            raw_prob = scorer.score(contexts=[doc for _ in range(len(claims))], claims=claims)
            final_prob = sum(raw_prob) / len(raw_prob)
            total_probs.append(final_prob)
            print(f'index: {index}. Score: {final_prob}')
            index += 1
        except:
            print(f'Error occurs: {index}')
            continue

    avg_prob = sum(total_probs) / len(total_probs)
    print(f'data path: {data_path}')
    print(f'average prob: {avg_prob}')
