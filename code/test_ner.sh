#--model_name_or_path dmis-lab/biobert-base-cased-v1.1 \
export SAVE_DIR=./output_CP
export DATA_DIR=../datasets/NER

export MAX_LENGTH=192
export BATCH_SIZE=3
export NUM_EPOCHS=10
export SAVE_STEPS=1000
export ENTITY=NCBI-disease/labeled_part
export SEED=1


python run_ner.py \
    --data_dir ${DATA_DIR}/${ENTITY}/ \
    --labels ${DATA_DIR}/${ENTITY}/labels.txt \
    --model_name_or_path output_CP/ \
    --tokenizer_name dmis-lab/biobert-base-cased-v1.1\
    --output_dir ${SAVE_DIR}/${ENTITY} \
    --max_seq_length ${MAX_LENGTH} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --save_steps ${SAVE_STEPS} \
    --seed ${SEED} \
    --do_train \
    --do_predict \
    --overwrite_output_dir \
    --auto_find_batch_size 
    --do_eval \
