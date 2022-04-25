export  CUDA_VISIBLE_DEVICES=0
python train_reqa_base.py \
  --data_file data/squad.json \
  --data_rate 1 \
  --max_question_len 36 \
  --max_answer_len 150 \
  --epoch 30 \
  --batch_size 24 \
  --encoder_type bert \
  --pooler_type att \
  --temperature 20 \
  --plm_path 'bert' \
  --save_model output/squad/bert_shuffle_lr2e-5 \
  --learning_rate 2e-5 \
  --warmup_proportion 0.1 \
  --weight_decay 0.0 \
  --theta 0.1 \
  --seed 82 \
  --no-shuffle \
  # --mixed_training
  #  --plm_path /workspace/wcr/models/english/bert-base-uncased \

