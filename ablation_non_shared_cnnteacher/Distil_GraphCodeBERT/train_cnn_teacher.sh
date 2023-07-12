python textcnn_main.py \
    --output_dir=./saved_models \
    --model_name=t1.bin \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --train_data_file=../../data/big_vul/g1/g1_train.csv \
    --eval_data_file=../../data/big_vul/g1/g1_val.csv \
    --test_data_file=../../data/big_vul/g1/g1_test.csv \
    --do_train \
    --do_test \
    --block_size 512 \
    --epochs 50 \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --learning_rate 5e-3 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train_teacher1.log

python textcnn_main.py \
    --output_dir=./saved_models \
    --model_name=t2.bin \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --train_data_file=../../data/big_vul/g2/g2_train.csv \
    --eval_data_file=../../data/big_vul/g2/g2_val.csv \
    --test_data_file=../../data/big_vul/g2/g2_test.csv \
    --do_train \
    --do_test \
    --block_size 512 \
    --epochs 50 \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --learning_rate 5e-3 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train_teacher2.log

python textcnn_main.py \
    --output_dir=./saved_models \
    --model_name=t3.bin \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --train_data_file=../../data/big_vul/g3/g3_train.csv \
    --eval_data_file=../../data/big_vul/g3/g3_val.csv \
    --test_data_file=../../data/big_vul/g3/g3_test.csv \
    --do_train \
    --do_test \
    --block_size 512 \
    --epochs 50 \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --learning_rate 5e-3 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train_teacher3.log

python textcnn_main.py \
    --output_dir=./saved_models \
    --model_name=t4.bin \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --train_data_file=../../data/big_vul/g4/g4_train.csv \
    --eval_data_file=../../data/big_vul/g4/g4_val.csv \
    --test_data_file=../../data/big_vul/g4/g4_test.csv \
    --do_train \
    --do_test \
    --block_size 512 \
    --epochs 50 \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --learning_rate 5e-3 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train_teacher4.log

python textcnn_main.py \
    --output_dir=./saved_models \
    --model_name=t5.bin \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --train_data_file=../../data/big_vul/g5/g5_train.csv \
    --eval_data_file=../../data/big_vul/g5/g5_val.csv \
    --test_data_file=../../data/big_vul/g5/g5_test.csv \
    --do_train \
    --do_test \
    --block_size 512 \
    --epochs 50 \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --learning_rate 5e-3 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train_teacher5.log