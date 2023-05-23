# finetune_CLIP_LLM_with_fashion_products_dataset
Infimind project to finetune Ch_CLIP model in the fashion field

## THANKS TO THE OPEN-SOURCE PROJECT ---- CHINESE-CLIP
you can get the repository [https://github.com/OFA-Sys/Chinese-CLIP]

## pipeline to finetune ch_CLIP model:
1.get the image and text/query dataset which can refer to the link [https://github.com/DengRay/pipeline_crawl_vip_product_dataset]
2.form the lmdb dataset for train/valid/test,running build_test.py,utilize the parameter --data_dir --splits
3.use the script sh file to finetune the Ch_CLIP model, running muge_finetune_vit-b-16_rbt-base.sh


