import argparse
import json
import os
import random
import csv
from typing import List

from module.tester import Tester
from module.finetuner import Finetuner

import numpy as np
import torch

# Reproducibility setting
def set_random_seed(seed: int):
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpus', default=[1])
    parser.add_argument('--num_process', default=int(os.cpu_count() * 0.8), type=int)
    parser.add_argument('--num_worker', default=6, type=int)
    parser.add_argument('--port', default=1234, type=int)

    parser.add_argument('--model_name', default="j-hartmann/emotion-english-distilroberta-base")
    
    # Training Environment
    parser.add_argument('--epoch', default=50, type=int)            # When fine-tuning is required
    parser.add_argument('--dropout', default=0.5, type=float)       # When fine-tuning is required
    parser.add_argument('--learning_rate', default=5e-5)            # When fine-tuning is required
    parser.add_argument('--batch_size', default=10, type=int)        # When fine-tuning is required
    
    parser.add_argument('--mode', default='fine-tuning') # fine-tuning or test
    parser.add_argument('--model_type', default=4) # fine-tuning additional model의 형태를 결정
    parser.add_argument('--fine_tuned_model', default=None)
    
    parser.add_argument('--taxonomy', default=False)  # pretrained emotion classification model to test has different emotion taxonomy as DailyDialog
    parser.add_argument('--n_emotion', default=7)  # pretrained emotion classification model to test has different emotion taxonomy as DailyDialog
    
    parser.add_argument('--train_data', default="data/data_fold/data_0/dailydialog_test.json")
    parser.add_argument('--test_data', default="data/data_fold/data_0/dailydialog_test.json")
    parser.add_argument('--data_label', default='dailydialog_fold_0')
    parser.add_argument('--model_label', default='j-hartmann_emotion-english-distilroberta-base')
    
    # logging 관련
    parser.add_argument('--log_directory', default='log-testing_model-type4-weights(5e-5)', type=str)
    parser.add_argument('--use_wandb', default=False)

    parser.add_argument('--max_seq_len', help='the max length of each tokenized utterance', default=75, type=int)
    return parser.parse_args()

def test_preconditions(args: argparse.Namespace):
    assert args.model_name is not None, "For test, load emotion classification model."

def main():
    args = parse_args()

    test_preconditions(args)
    set_random_seed(77)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(_) for _ in args.gpus])
    
    # model_name = ["j-hartmann/emotion-english-distilroberta-base", "j-hartmann/emotion-english-roberta-large"]
    # model_label = ['j-hartmann distill roberta base', 'j-hartmann roberta large']
    # model_name = ["j-hartmann/emotion-english-roberta-large"]
    # model_label = ['j-hartmann_roberta_large']
    model_name = ["j-hartmann/emotion-english-distilroberta-base"]
    model_label = ['j-hartmann_distill_roberta_base']
    
    # model_name = ["j-hartmann/emotion-english-roberta-large",
    #               "nateraw/bert-base-uncased-emotion",
    #               "gokuls/distilroberta-emotion-intent",
    #               "bergum/xtremedistil-l6-h384-go-emotion",
    #               "gokuls/distilroberta-emotion-intent"]
    # model_label = ['j-hartmann roberta large',
    #                "nateraw_bert-base-uncased-emotion",
    #                "gokuls_distilroberta-emotion-intent",
    #                "bergum_xtremedistil-l6-h384-go-emotion",
    #                "gokuls_distilroberta-emotion-intent"]
    
    if args.mode == 'fine-tuning':
        # train_datas = ['data_fold/data_0/dailydialog_train.json', * 
        #             [f'data_fold/data_{fold_}/data_{fold_}_train.json' for fold_ in range(1, 5)]]
        # test_datas = ['data_fold/data_0/dailydialog_test.json', * 
        #             [f'data_fold/data_{fold_}/data_{fold_}_test.json' for fold_ in range(1, 5)]]
        # data_labels = ['-original_dd', *[f'-data_{fold_}_DailyDialog' for fold_ in range(1, 5)]]
        
        train_datas = ['data_fold/data_0/dailydialog_train.json']
        test_datas = ['data_fold/data_0/dailydialog_test.json']
        data_labels = ['-original_dd']
        
        print("Fine-tuning")
        
        for mn, ml in zip(model_name, model_label):
            args.model_name = mn
            args.model_label = ml
            
            print("model: ", args.model_name)
            for tr, te, dl in zip(train_datas, test_datas, data_labels):
                args.train_data, args.test_data, args.data_label = tr, te, dl
                
                finetuner = Finetuner(**vars(args))
                finetuner.run(**vars(args))
                
                del finetuner
                
    # elif args.mode == 'test':
    #     test_datas = ['data_fold/data_0/dailydialog_test.json', * 
    #                 [f'data_fold/data_{fold_}/data_{fold_}_test.json' for fold_ in range(1, 5)]]
    #     data_labels = ['-original_dd', *[f'-data_{fold_}_DailyDialog' for fold_ in range(1, 5)]]
    #     print("Testing fine-tuned")
        
    #     for mn, ml in zip(model_name, model_label):
    #         args.model_name = mn
    #         args.model_label = ml
            
    #         for te, dl in zip(test_datas, data_labels):
    #             args.test_data, args.data_label = te, dl
                
    #             model_file = args.fine_tuned_model
    #             tester = Finetuner(**vars(args))
    #             tester.test_b_model(model_file)
                
    #             del tester
    
    
    # 이거는 tester.py 를 사용한 테스트 (taxonomy가 같은 모델)
    # elif args.mode == 'test':
    #     test_datas = ['data_fold/data_0/dailydialog_test.json', * 
    #                 [f'data_fold/data_{fold_}/data_{fold_}_test.json' for fold_ in range(1, 5)]]
    #     data_labels = ['-original_dd', *[f'-data_{fold_}_DailyDialog' for fold_ in range(1, 5)]]
    #     print("Testing")
        
    #     for mn, ml in zip(model_name, model_label):
    #         tester = Tester(mn, ml, test_datas, data_labels)
    #         tester.run()

    #         del tester
            
        

if __name__ == '__main__':
    main()