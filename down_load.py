from modelscope.hub.snapshot_download import snapshot_download

embeddings_model = "iic/multi-modal_clip-vit-base-patch16_zh"
chat_model_7b_int8 = "qwen/Qwen-7B-Chat-Int8" 
chat_model_7b_int4 = "qwen/Qwen-7B-Chat-Int4"
chat_model_15_7b_int4_GPTQ = "qwen/Qwen1.5-7B-Chat-GPTQ-Int4"
chat_model_14b_int4 =  "qwen/Qwen-14B-Chat-Int4"

embeddings_model_dir = snapshot_download(embeddings_model, cache_dir='/root/autodl-tmp')
print("embeddings_model_dir:",embeddings_model_dir)

chat_model_dir_8 = snapshot_download(chat_model_7b_int8, cache_dir='/root/autodl-tmp')
print("chat_model_7b_int8:",chat_model_dir_8)

chat_model_dir_4 = snapshot_download(chat_model_7b_int4, cache_dir='/root/autodl-tmp')
print("chat_model_7b_int4:",chat_model_dir_4)

chat_model_dir_15_int4 = snapshot_download(chat_model_15_7b_int4_GPTQ, cache_dir='/root/autodl-tmp')
print("chat_model_7b_int4:",chat_model_dir_15_int4)

chat_model_dir_14 = snapshot_download(chat_model_14b_int4, cache_dir='/root/autodl-tmp')
print("chat_model_14b_int4:",chat_model_dir_14)

