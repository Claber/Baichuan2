model_dir="../../"
model_name="Baichuan2-13B-Chat"
model_output_dir="output"
when_start=`date "+%Y-%m-%d-%H-%M-%S"`

train_data_path="data/belle_chat_ramdon_10k.json"

#note that CUDA_VISIBLE_DEVICES can’t be used with DeepSpeed to control which devices should be used.
use_devices="4,5,6,7"

echo "Finetuning model = "$model_name", with data = "${train_data_path}", at "${when_start}
echo "target dir = ["{$model_output_dir}"/"${when_start}

# check directories
if [ -d "${model_output_dir}" ]; then
    echo "${model_output_dir} exists"
else
    mkdir -p ${model_output_dir}
    echo "${model_output_dir}  created successfully."
fi

if [ -d "${model_output_dir}/${when_start}" ]; then
    echo "${model_output_dir}/${when_start} already exists, exit to avoid overwritting!"
    exit 1
else
    mkdir -p "${model_output_dir}/${when_start}"
    echo "${model_output_dir}/${when_start} created successfully."
fi


# sinlge host with multiple GPUs
hostfile=""
deepspeed --hostfile=$hostfile --include="localhost:"${use_devices}  fine-tune.py  \
    --report_to "none" \
    --data_path ${train_data_path} \
    --model_name_or_path ${model_dir}${model_name} \
    --output_dir ${model_output_dir} \
    --model_max_length 4096 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed ds_config.json \
    --bf16 True \
    --tf32 True \
    --use_lora True

when_done=`date "+%Y-%m-%d-%H-%M-%S"`
echo "done, at "${when_done}