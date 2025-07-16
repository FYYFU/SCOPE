python3 eval_minicheck.py --gpu 0 --dataset gov_report --method SnapKV --kv_size 64 
python3 eval_minicheck.py --gpu 0 --dataset multi_news --method SnapKV --kv_size 64 
python3 eval_minicheck.py --gpu 0 --dataset qmsum --method SnapKV --kv_size 64 

python3 eval_minicheck.py --gpu 0 --dataset gov_report --method FullKV --kv_size 64 
python3 eval_minicheck.py --gpu 0 --dataset multi_news --method FullKV --kv_size 64 
python3 eval_minicheck.py --gpu 0 --dataset qmsum --method FullKV --kv_size 64 

python3 eval_minicheck.py --gpu 0 --dataset gov_report --method pyramidkv --kv_size 64 
python3 eval_minicheck.py --gpu 0 --dataset multi_news --method pyramidkv --kv_size 64 
python3 eval_minicheck.py --gpu 0 --dataset qmsum --method pyramidkv --kv_size 64 

python3 eval_minicheck.py --gpu 0 --dataset gov_report --method SnapKV_256-128 --kv_size 64 
python3 eval_minicheck.py --gpu 0 --dataset multi_news --method SnapKV_256-128 --kv_size 64 
python3 eval_minicheck.py --gpu 0 --dataset qmsum --method SnapKV_256-128 --kv_size 64 
