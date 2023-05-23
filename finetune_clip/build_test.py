import argparse
import os
import lmdb
from tqdm import tqdm
import json
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",type = str,required = True,help="the directory which stores the image tsvfiles and the text jsonl annotations"
    )
    parser.add_argument(
        "--splits",type = str,required = True,help = "specify the dataset splits which this script processes,concatenated by comma \
            (e.g. train,valid,text)"
    )
    parser.add_argument(
        "--lmdb_dir",type = str,default=None,help="specify the directory which stores the output lmdb files. \
            if set to None,the lmdb_dir will be set to {args.data_dir/lmdb}"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    assert os.path.isdir(args.data_dir),"the data_dir does not exist! Please check the input args..."
    print(args.data_dir)
    print(args.splits)
    print(args.lmdb_dir)
    
    specified_splits = list(set(args.splits.strip().split(",")))
    print(specified_splits)
    print("Dataset splits to be processed:{}".format(",".join(specified_splits)))
    
    #build LMDB data files
    if args.lmdb_dir is None:
        args.lmdb_dir = os.path.join(args.data_dir,"lmdb")
        print(args.lmdb_dir)
    for split in specified_splits:
        lmdb_split_dir = os.path.join(args.lmdb_dir,split)
        if os.path.isdir(lmdb_split_dir):
            print("We will overwrit an exsting LMDB file{}".format(lmdb_split_dir))
        os.makedirs(lmdb_split_dir,exist_ok = True)
        lmdb_img = os.path.join(lmdb_split_dir,"imgs")
        env_img = lmdb.open(lmdb_img,map_size=1024**4)
        txn_img = env_img.begin(write=True)
        lmdb_pairs = os.path.join(lmdb_split_dir,"pairs")
        env_pairs = lmdb.open(lmdb_pairs,map_size=1024**4)
        txn_pairs = env_pairs.begin(write=True)
        
        # write LMDB file storing(imge_id,text_id,text) pairs
        pairs_annotation_path = os.path.join(args.data_dir,"{}_queries.jsonl".format(split))
        #print(pairs_annotation_path)
        with open(pairs_annotation_path,"r",encoding="utf-8") as fin_pairs:
            write_idx = 0
            for line in tqdm(fin_pairs):
                line = line.strip()
                obj = json.loads(line)
            
                if split == "test":
                    dump = pickle.dumps(([], obj['query_id'], obj['query_text']))
                    txn_pairs.put(key="{}".format(write_idx).encode('utf-8'),value=dump)
                    write_idx += 1
                    if write_idx % 5000 == 0:
                        txn_pairs.commit()
                        txn_pairs = env_pairs.begin(write=True)
                else: 
                    for field in ("query_id","query_text","item_ids"):
                        assert field in obj,"Field {} does not exist in line {}. \
                            please check the integrity of the text annotation jsonl file".format(field,obj)
                    for item_id in obj["item_ids"]:  #此处的id是字典类型的
                        dump = pickle.dumps((item_id, obj['query_id'], obj['query_text']))#在test集合中没有图片id
                        txn_pairs.put(key="{}".format(write_idx).encode('utf-8'),value=dump)
                        write_idx += 1
                        if write_idx % 5000 == 0:
                            txn_pairs.commit()
                            txn_pairs = env_pairs.begin(write=True)
            txn_pairs.put(key=b'num_samples',
                          value="{}".format(write_idx).encode('utf-8'))
            txn_pairs.commit()
            env_pairs.close()
        print("Finished serializing {} {} split pairs into{}".format(write_idx,split,lmdb_pairs))
        
        #write LMDB file storing image base64 strings
        base64_path = os.path.join(args.data_dir,"{}_imgs.tsv".format(split))
        with open(base64_path,"r",encoding="utf-8") as fin_imgs:
            write_idx = 0
            for line in tqdm(fin_imgs):
                line = line.strip()
                image_id,b64 = line.split("\t")
                txn_img.put(key="{}".format(image_id).encode('utf-8'),value=b64.encode('utf-8'))
                write_idx += 1
                if write_idx % 1000 == 0:
                    txn_img.commit()
                    txn_img = env_img.begin(write=True)
            txn_img.put(key=b'num_images',
                        value="{}".format(write_idx).encode('utf-8'))
            txn_img.commit()
            env_img.close()
        print("finished serializing {} {} split images into {}.".format(write_idx,split,lmdb_img))
        
    print("all done")
                
                        
                    