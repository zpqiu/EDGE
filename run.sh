# train
CUDA_VISIBLE_DEVICES=0 fairseq-train data --task dg_task --arch dg --user-dir . --batch-size 128 --optimizer adam --lr 0.0005 \   
     --max_positions 500 --max_tgt_positions 15 --max_q_positions 17 --max_ans_positions 15  \   
     --encoder-embed-dim 300 --decoder-embed-dim 300 --encoder-hidden-size 150 --decoder-hidden-size 300 --encoder-bidirectional \
     --decoder-out-embed-dim 300 --encoder-embed-path glove.840B.300d.bin.npy --decoder-embed-path glove.840B.300d.bin.npy  --fp16 --share-all-embeddings


# generate for test data
CUDA_VISIBLE_DEVICES=1 fairseq-generate data --task dg_task --user-dir . --batch-size 32 --path checkpoints/checkpoint_40.pt \
    --max_positions 500 --max_tgt_positions 15 --max_q_positions 17 --max_ans_positions 15  \
    --nbest 50 --beam 50 --fp16 > results.txt

# evaluation
python eval_from_generate.py --test_path data/race_test.json --gen_path results.txt