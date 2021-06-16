# utt data and 20% ted talk trained concurrently model test
# CUDA_VISIBLE_DEVICES=1,2 python src/test.py --pretrained-model=bert-base-uncased --lstm-dim=-1 --data-path=data/test --weight-path=out/mixed_utt_mini_en_data_sliding_window_seed_1 --sequence-length=256 --save-path=out/mixed_utt_mini_en_data_sliding_window_seed_1/test_without_window --sliding-window=False --stride_size=0.5
# CUDA_VISIBLE_DEVICES=1,2 python src/test.py --pretrained-model=bert-base-multilingual-cased --lstm-dim=-1 --data-path=data/test --weight-path=out/mixed_utt_mini_en_data --sequence-length=256 --save-path=out/mixed_utt_mini_en_data_sliding_window_seed_1
# CUDA_VISIBLE_DEVICES=1,2 python src/test.py --pretrained-model=xlm-roberta-base --lstm-dim=-1 --data-path=data/test --weight-path=out/mixed_utt_mini_en_data_sliding_window_seed_1 --sequence-length=256 --save-path=out/mixed_utt_mini_en_data_sliding_window_seed_1/test_without_window --sliding-window=False --stride_size=0.5
# CUDA_VISIBLE_DEVICES=1,2 python src/test.py --pretrained-model=bert-base-multilingual-uncased --lstm-dim=-1 --data-path=data/test --weight-path=out/mixed_utt_mini_en_data_sliding_window_seed_1 --sequence-length=256 --save-path=out/mixed_utt_mini_en_data_sliding_window_seed_1/test_without_window --sliding-window=False
# # 
#
# utt data and 20% LJ Data trained concurrently model test
CUDA_VISIBLE_DEVICES=2 python src/test.py --pretrained-model=bert-base-uncased --lstm-dim=-1 --data-path=data/test --weight-path=out/mixed_utt_mini_LJ_data_sliding_window_seed_1 --sequence-length=256 --save-path=out/mixed_utt_mini_LJ_data_sliding_window_seed_1/test_without_window --sliding-window=False --stride_size=-0.5
CUDA_VISIBLE_DEVICES=2 python src/test.py --pretrained-model=bert-base-multilingual-uncased --lstm-dim=-1 --data-path=data/test --weight-path=out/mixed_utt_mini_LJ_data_sliding_window_seed_1 --sequence-length=256 --save-path=out/mixed_utt_mini_LJ_data_sliding_window_seed_1/test_without_window --sliding-window=False
CUDA_VISIBLE_DEVICES=2 python src/test.py --pretrained-model=xlm-roberta-base --lstm-dim=-1 --data-path=data/test --weight-path=out/mixed_utt_mini_LJ_data_sliding_window_seed_1 --sequence-length=256 --save-path=out/mixed_utt_mini_LJ_data_sliding_window_seed_1/test_without_window --sliding-window=False
#
# 
# utt data and 20% ted talk trained concurrently (with lower case)
# CUDA_VISIBLE_DEVICES=1,2 python src/train.py --cuda=True --pretrained-model=bert-base-uncased --freeze-bert=False --lstm-dim=-1 --language=mini-english --seed=1 --lr=5e-6 --epoch=10 --augment-type=all  --augment-rate=0.15 --alpha-sub=0.4 --alpha-del=0.4 --data-path=data --save-path=out/mixed_utt_mini_en_data_sliding_window_seed_1
# CUDA_VISIBLE_DEVICES=1,2 python src/train.py --cuda=True --pretrained-model=bert-base-multilingual-uncased --freeze-bert=False --lstm-dim=-1 --language=mini-english --seed=1 --lr=5e-6 --epoch=10 --augment-type=all  --augment-rate=0.15 --alpha-sub=0.4 --alpha-del=0.4 --data-path=data --save-path=out/mixed_utt_mini_en_data_sliding_window_seed_1
# CUDA_VISIBLE_DEVICES=1,2 python src/train.py --cuda=True --pretrained-model=xlm-roberta-base --freeze-bert=False --lstm-dim=-1 --language=mini-english --seed=1 --lr=5e-6 --epoch=10 --augment-type=all  --augment-rate=0.15 --alpha-sub=0.4 --alpha-del=0.4 --data-path=data --save-path=out/mixed_utt_mini_en_data_sliding_window_seed_1
#
#
# utt and 20% mixed-LJ-speech trained concurrently (with lower case)
# CUDA_VISIBLE_DEVICES=1 python src/train.py --cuda=True --pretrained-model=bert-base-uncased --freeze-bert=False --lstm-dim=-1 --language=mixed-LJ-speech --seed=1 --lr=5e-6 --epoch=10 --augment-type=all  --augment-rate=0.15 --alpha-sub=0.4 --alpha-del=0.4 --data-path=data --save-path=out/mixed_utt_mini_LJ_data_sliding_window_seed_1 --sliding-window=True --stride_size=0.5
# CUDA_VISIBLE_DEVICES=1 python src/train.py --cuda=True --pretrained-model=bert-base-multilingual-uncased --freeze-bert=False --lstm-dim=-1 --language=mixed-LJ-speech --seed=1 --lr=5e-6 --epoch=10 --augment-type=all  --augment-rate=0.15 --alpha-sub=0.4 --alpha-del=0.4 --data-path=data --save-path=out/mixed_utt_mini_LJ_data_sliding_window_seed_1 --sliding-window=True --stride_size=0.5
# CUDA_VISIBLE_DEVICES=1 python src/train.py --cuda=True --pretrained-model=xlm-roberta-base --freeze-bert=False --lstm-dim=-1 --language=mixed-LJ-speech --seed=1 --lr=5e-6 --epoch=10 --augment-type=all  --augment-rate=0.15 --alpha-sub=0.4 --alpha-del=0.4 --data-path=data --save-path=out/mixed_utt_mini_LJ_data_sliding_window_seed_1 --sliding-window=True --stride_size=0.5
#
# test
# python src/train.py --cuda=True --pretrained-model=bert-base-uncased --freeze-bert=False --lstm-dim=-1 --language=test --sequence-length=20 --seed=1 --lr=5e-6 --epoch=10 --augment-type=all  --augment-rate=0 --alpha-sub=0.4 --alpha-del=0.4 --data-path=data --save-path=out/test --sliding-window=True