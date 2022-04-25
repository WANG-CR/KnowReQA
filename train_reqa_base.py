'''
This script handling the training process.
'''
import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import math
import time
import logging
import json
import random
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import torch

from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, BertTokenizer

from utils_data import ReqaDataset, SentDataset


logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def train_epoch(model, train_loader, optimizer, scheduler, scaler, epoch_i, args):

    model.train()
    total_tr_loss = 0.0
    total_train_batch = 0
    total_acc = 0.0

    for step, batch in enumerate(tqdm(train_loader, desc='  -(Training)', leave=False)):
        # prepare data
        if 'know' not in args.encoder_type:
            src_seq, src_mask, tgt_seq, tgt_mask = map(lambda x: x.to(args.device), batch)

            # forward
            if args.mixed_training:
                with autocast():
                    tr_loss, tr_acc = model(src_seq, src_mask, tgt_seq, tgt_mask)
            else:
                tr_loss, tr_acc = model(src_seq, src_mask, tgt_seq, tgt_mask)
    
        elif 'know' in args.encoder_type:
            q_input_ids, q_entity_ids, q_entity_position_ids, q_attention_mask, q_entity_attention_mask, a_input_ids, a_entity_ids, a_entity_position_ids, a_attention_mask, a_entity_attention_mask = map(lambda x: x.to(args.device), batch)

            if args.mixed_training:
                with autocast():
                    tr_loss, tr_acc = model(q_input_ids, q_entity_ids, q_entity_position_ids, q_attention_mask, q_entity_attention_mask, a_input_ids, a_entity_ids, a_entity_position_ids, a_attention_mask, a_entity_attention_mask)
            else:
                tr_loss, tr_acc = model(q_input_ids, q_entity_ids, q_entity_position_ids, q_attention_mask, q_entity_attention_mask, a_input_ids, a_entity_ids, a_entity_position_ids, a_attention_mask, a_entity_attention_mask)
        
        # backward
        if args.mixed_training:
            scaler.scale(tr_loss).backward()
        else:
            tr_loss.backward()

        # record
        total_acc += tr_acc
        total_tr_loss += tr_loss.item()
        total_train_batch += 1

        # update parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if args.mixed_training:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        model.zero_grad()

    return total_tr_loss / total_train_batch, total_acc / total_train_batch

def dev_eval(model, dev_loader, args):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    src_embeddings = []
    tgt_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc='  -(Validation)', leave=False):
            if 'know' not in args.encoder_type:
                # prepare data
                src_seq, src_mask, tgt_seq, tgt_mask = map(lambda x: x.to(args.device), batch)

                # forward
                src_embedding, tgt_embedding = model(src_seq, src_mask, tgt_seq, tgt_mask)

            elif 'know' in args.encoder_type:
                q_input_ids, q_entity_ids, q_entity_position_ids, q_attention_mask, q_entity_attention_mask, a_input_ids, a_entity_ids, a_entity_position_ids, a_attention_mask, a_entity_attention_mask = map(lambda x: x.to(args.device), batch)
                src_embedding, tgt_embedding = model(q_input_ids, q_entity_ids, q_entity_position_ids, q_attention_mask, q_entity_attention_mask, a_input_ids, a_entity_ids, a_entity_position_ids, a_attention_mask, a_entity_attention_mask)
            
            src_embeddings.append(src_embedding.cpu().numpy())
            tgt_embeddings.append(tgt_embedding.cpu().numpy())

    src_embeddings = np.concatenate(src_embeddings, 0)
    tgt_embeddings = np.concatenate(tgt_embeddings, 0)
    
    all_predict_logits = np.matmul(src_embeddings, tgt_embeddings.T)
    all_ground_truth = [i for i in range(all_predict_logits.shape[0])]

    rankat = [1, 5, 10]
    r_counts = defaultdict(float)
    for rank in rankat:
        r_counts[rank] = 0
    r_rank = 0

    for num in range(len(all_ground_truth)):
        pred = np.argsort(-all_predict_logits[num]).tolist()
        for rank in rankat:
            if all_ground_truth[num] in pred[:rank]:
                r_counts[rank] += 1
    
        for idx, p in enumerate(pred):
            if p == all_ground_truth[num]:
                r_rank += 1/(idx+1)
                break

    mrr = np.round(r_rank/len(all_ground_truth), 4)
    r_at_k = [np.round(v/len(all_ground_truth), 4) for k, v in sorted(r_counts.items(), key=lambda item: item[0])]

    return mrr*100, r_at_k[0]*100, r_at_k[1]*100, r_at_k[2]*100


def test_eval(model, test_loader, args):

    model.eval()
    test_question_loader, test_candidate_loader, test_ground_truth = test_loader

    # question encoding
    question_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_question_loader, desc='  -(test question encoding)', leave=False):
            if 'know' not in args.encoder_type:
                # prepare data
                src_seq, src_mask = map(lambda x: x.to(args.device), batch)

                # forward
                src_embedding = model.sentence_encoding(src_seq, src_mask)

            elif 'know' in args.encoder_type:
                q_input_ids, q_entity_ids, q_entity_position_ids, q_attention_mask, q_entity_attention_mask = map(lambda x: x.to(args.device), batch)
                src_embedding= model.sentence_encoding(q_input_ids, q_entity_ids, q_entity_position_ids, q_attention_mask, q_entity_attention_mask)
            
            question_embeddings.append(src_embedding.cpu().numpy())

    question_embeddings = np.concatenate(question_embeddings, 0)

    # candidate encoding
    candidate_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_candidate_loader, desc='  -(test candidate encoding)', leave=False):
            if 'know' not in args.encoder_type:
                # prepare data
                src_seq, src_mask = map(lambda x: x.to(args.device), batch)

                # forward
                src_embedding = model.sentence_encoding(src_seq, src_mask)

            elif 'know' in args.encoder_type:
                q_input_ids, q_entity_ids, q_entity_position_ids, q_attention_mask, q_entity_attention_mask = map(lambda x: x.to(args.device), batch)
                src_embedding= model.sentence_encoding(q_input_ids, q_entity_ids, q_entity_position_ids, q_attention_mask, q_entity_attention_mask)
            
            candidate_embeddings.append(src_embedding.cpu().numpy())

    candidate_embeddings = np.concatenate(candidate_embeddings, 0)
    
    predict_logits = np.matmul(question_embeddings, candidate_embeddings.T)

    rankat = [1, 5, 10]
    r_counts = defaultdict(float)
    for rank in rankat:
        r_counts[rank] = 0
    r_rank = 0

    for num in range(len(test_ground_truth)):
        pred = np.argsort(-predict_logits[num]).tolist()
        for rank in rankat:
            for gt in test_ground_truth[num]:
                if gt in pred[:rank]:
                    r_counts[rank] += 1
                    break
    
        for idx, p in enumerate(pred):
            if p in test_ground_truth[num]:
                r_rank += 1/(idx+1)
                break

    mrr = np.round(r_rank/len(test_ground_truth), 4)
    r_at_k = [np.round(v/len(test_ground_truth), 4) for k, v in sorted(r_counts.items(), key=lambda item: item[0])]

    return mrr*100, r_at_k[0]*100, r_at_k[1]*100, r_at_k[2]*100


def run(model, train_loader, dev_loader, test_loader, args):
    args.num_training_steps = int(args.num_train_instances / args.batch_size * args.epoch)
    logger.info("batch size:{}".format(args.batch_size))
    logger.info("total train_steps:{}".format(args.num_training_steps))
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(args.warmup_proportion * args.num_training_steps), 
        num_training_steps=args.num_training_steps
    )
    # mixed precision training
    scaler = GradScaler()
    
    best_metrics = 0
    best_epoch = 0

    for epoch_i in range(args.epoch):
        logger.info('[Epoch {}]'.format(epoch_i))

        start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, scaler, epoch_i, args)
        logger.info('[Epoch{epoch: d}] - (Train) loss ={train_loss: 8.5f}, acc ={acc: 3.2f} %, '\
                'elapse ={elapse: 3.2f} min'.format(epoch=epoch_i, train_loss=train_loss, acc=100*train_acc,
                elapse=(time.time()-start)/60))

        dev_mrr, dev_r1, dev_r5, dev_r10 = dev_eval(model, dev_loader, args)
        logger.info('[Epoch{epoch: d}] - (Dev  ) mrr ={mrr: 3.2f} %, r1 ={r1: 3.2f} %,'\
            ' r5 ={r5: 3.2f} %, r10 ={r10: 3.2f} %'.format(epoch=epoch_i, mrr=dev_mrr, r1=dev_r1, r5=dev_r5, r10=dev_r10))

        test_mrr, test_r1, test_r5, test_r10 = test_eval(model, test_loader, args)
        logger.info('[Epoch{epoch: d}] - (Test ) mrr ={mrr: 3.2f} %, r1 ={r1: 3.2f} %,'\
            ' r5 ={r5: 3.2f} %, r10 ={r10: 3.2f} %'.format(epoch=epoch_i, mrr=test_mrr, r1=test_r1, r5=test_r5, r10=test_r10))

        current_metrics = dev_r1

        if args.save_model:
            model_name = args.save_model + '/ranker.ckpt'
            if not os.path.exists(args.save_model):
                os.makedirs(args.save_model)
            if current_metrics >= best_metrics:
                best_epoch = epoch_i
                best_metrics = current_metrics
                model_state_dict = model.state_dict()
                checkpoint = {
                    'model': model_state_dict,
                    'args': args,
                    'epoch': epoch_i}
                torch.save(checkpoint, model_name)
                logger.info('  - [Info] The checkpoint file has been updated.')
    logger.info(f'Got best test performance on epoch{best_epoch}')
    logger.info('\n')


def prepare_dataloaders(args):
    # set tokenizer
    if args.encoder_type == 'bert' or args.encoder_type == 'colbert' or args.encoder_type == 'gru':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        print(f'building bert tokenizer')

    if args.encoder_type == 'roberta' or args.encoder_type == 'kepler' or args.encoder_type == 'colroberta' :
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        print(f'building roberta tokenizer')

    if args.encoder_type == 'know' or args.encoder_type == 'colknow':
        from transformers import LukeTokenizer
        tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
        print(f'building luke tokenizer')

    args.tokenizer = tokenizer

    # Load data features from cache or dataset file
    cached_dir = "./cached_data/"
    if not os.path.exists(cached_dir):
        os.makedirs(cached_dir)
    plm_name = [s for s in args.plm_path.split('/') if s !=''][-1]
    dataset_name = args.data_file.split('/')[-1].replace('.json', '')
    cached_insts_file = os.path.join(cached_dir, f"{dataset_name}_{plm_name}")

    if 'know' not in args.encoder_type:
        # load processed dataset or process the original dataset
        if os.path.exists(cached_insts_file) and not args.overwrite_cache:
            logger.info("Loading instances from cached file %s", cached_insts_file)
            data_dict =  torch.load(cached_insts_file)
            train_question_insts = data_dict["train"]["question_insts"]
            train_answer_insts = data_dict["train"]["answer_insts"]
            dev_question_insts = data_dict["dev"]["question_insts"]
            dev_answer_insts = data_dict["dev"]["answer_insts"]
            test_question_insts = data_dict["test"]["question_insts"]
            test_candidate_insts = data_dict["test"]["candidate_insts"]
            test_ground_truth = data_dict['test']['ground_truth']
        else:
            logger.info("Creating instances from dataset file at %s", args.data_file)
            with open(args.data_file, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
            # tokenize the seqs
            def tokenize(data, data_type):
                tokens = []
                insts = []
                for index, example in tqdm(enumerate(data), desc=f'  -({data_type})', leave=True):
                    line = data[index]
                    token = tokenizer.tokenize(line)
                    inst = tokenizer.convert_tokens_to_ids(token)

                    tokens.append(token)
                    insts.append(inst)
                
                return tokens, insts

            train_question_tokens, train_question_insts = tokenize(data_dict['train']['questions'], 'train questions')
            train_answer_tokens, train_answer_insts = tokenize(data_dict['train']['answers'], 'train answers')
            dev_question_tokens, dev_question_insts = tokenize(data_dict['dev']['questions'], 'dev questions')
            dev_answer_tokens, dev_answer_insts = tokenize(data_dict['dev']['answers'], 'dev answers')

            test_question_tokens, test_question_insts = tokenize(data_dict['test']['questions'], 'test questions')
            test_candidate_tokens, test_candidate_insts = tokenize(data_dict['test']['candidates'], 'test candidates')
            test_ground_truth = data_dict['test']['ground_truth']
            # ground truth shape [#questions,1]

            print(f'train question: {data_dict["train"]["questions"][0]}')
            print(f'train question token: {train_question_tokens[0]}')
            print(f'train question inst: {train_question_insts[0]}')
            print(f'train answer: {data_dict["train"]["answers"][0]}')
            print(f'train answer token: {train_answer_tokens[0]}')
            print(f'train answer inst: {train_answer_insts[0]}')
            print()
            
            saved_data = {
                'train':{
                    'question_insts': train_question_insts,
                    'question_tokens': train_question_tokens,
                    'answer_insts': train_answer_insts,
                    'answer_tokens': train_answer_tokens
                },
                'dev':{
                    'question_insts': dev_question_insts,
                    'question_tokens': dev_question_tokens,
                    'answer_insts': dev_answer_insts,
                    'answer_tokens': dev_answer_tokens
                },
                'test':{
                    'question_insts': test_question_insts,
                    'question_tokens': test_question_tokens,
                    'candidate_insts': test_candidate_insts,
                    'candidate_tokens': test_candidate_tokens,
                    'ground_truth': test_ground_truth
                }
            }
            logger.info("Saving processed instances to %s", cached_insts_file)
            torch.save(saved_data, cached_insts_file)
        
        args.num_train_instances = int(len(train_question_insts) * args.data_rate)
        logger.info(f"number of train instances {args.num_train_instances}")
        logger.info(f"number of dev instances {len(dev_question_insts)}")
        logger.info(f"number of test instances {len(test_question_insts)}")
    
        # train dataset
        train_dataset = ReqaDataset(
                args = args,
                question_insts=train_question_insts[:args.num_train_instances],
                answer_insts=train_answer_insts[:args.num_train_instances]
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=args.shuffle,
            batch_size=args.batch_size,
            collate_fn=args.train_fn)
        
        # dev dataset
        dev_dataset = ReqaDataset(
                args = args,
                question_insts=dev_question_insts,
                answer_insts=dev_answer_insts
            )

        dev_loader = torch.utils.data.DataLoader(
            dev_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=args.train_fn)
        
        # test dataset
        test_question_dataset = SentDataset(
                args = args,
                sentence_insts=test_question_insts
            )

        test_question_loader = torch.utils.data.DataLoader(
            test_question_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=args.eval_fn)
        
        test_candidate_dataset = SentDataset(
                args = args,
                sentence_insts=test_candidate_insts
            )

        test_candidate_loader = torch.utils.data.DataLoader(
            test_candidate_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=args.eval_fn)

        return train_loader, dev_loader, (test_question_loader, test_candidate_loader, test_ground_truth)


    elif 'know' in args.encoder_type:
        if os.path.exists(cached_insts_file) and not args.overwrite_cache:
            logger.info("Loading instances from cached file %s", cached_insts_file)
            data_dict =  torch.load(cached_insts_file)
            train_question_tuples = data_dict["train"]["question_tuples"]
            train_answer_tuples = data_dict["train"]["answer_tuples"]
            dev_question_tuples = data_dict["dev"]["question_tuples"]
            dev_answer_tuples = data_dict["dev"]["answer_tuples"]
            test_question_tuples = data_dict["test"]["question_tuples"]
            test_candidate_tuples = data_dict["test"]["candidate_tuples"]
            test_ground_truth = data_dict['test']['ground_truth']
        
        else:
            logger.info("Creating instances from dataset file at %s", args.data_file)
            with open(args.data_file, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)

            def turn_into_spans(start, end):
                spans = []
                assert len(start)==len(end)
                for i in range(len(start)):
                    spans.append((start[i], end[i]))
                return spans     

            def turn_into_spans2(list_of_spans, start):
                spans = []
                assert len(start)==len(list_of_spans)
                for i in range(len(list_of_spans)):
                    spans.append((list_of_spans[i][0], list_of_spans[i][1]))
                return spans    


            def tokenize(data, data_type, max_length, theta):
                tokens = []
                tuples = []
                for index, example in tqdm(enumerate(data), desc=f'  -({data_type})', leave=True):
                    line = example["text"]
                    token = tokenizer.tokenize(line)
                    entities = example["entities"]
                    entity_titles = example["entity_titles"]
                    entity_spans = turn_into_spans2(example["entity_spans"], example["entity_begins"])
                    # entity_spans = turn_into_spans(example["entity_begins"], example["entity_ends"])
                    scores = example["scores"]
                    
                    #filter low scored entities:
                    filtered_index = []
                    for j in range(len(scores)):
                        if scores[j] >= theta:
                           filtered_index.append(j) 
                    filtered_entities = [entities[k] for k in filtered_index]
                    filtered_entity_spans = [entity_spans[k] for k in filtered_index]
                    filtered_entity_titles = [entity_titles[k] for k in filtered_index]
                    tuple = tokenizer(line,entities=filtered_entity_titles, entity_spans=filtered_entity_spans, add_prefix_space=True, truncation = True, padding='max_length', max_length=max_length)

                    #filter unknown entities:
                    filtered_2_index = []
                    for j, entity_id in enumerate(tuple["entity_ids"]):
                        if entity_id != 0 and entity_id != 2 and entity_id != 1:
                            filtered_2_index.append(j)
                    filtered_entities = [entities[k] for k in filtered_index if k in filtered_2_index]
                    filtered_entity_spans = [entity_spans[k] for k in filtered_index if k in filtered_2_index]
                    filtered_entity_titles = [entity_titles[k] for k in filtered_index if k in filtered_2_index]
                    tuple = tokenizer(line,entities=filtered_entity_titles, entity_spans=filtered_entity_spans, add_prefix_space=True, truncation = True, padding='max_length', max_length=max_length)
                    
                    tokens.append(token)
                    tuples.append(tuple)
                
                return tokens, tuples

            theta = args.theta
            train_question_tokens, train_question_tuples = tokenize(data_dict['train']['questions'], 'train questions', args.max_question_len, theta)
            train_answer_tokens, train_answer_tuples = tokenize(data_dict['train']['answers'], 'train answers', args.max_answer_len, theta)
            dev_question_tokens, dev_question_tuples = tokenize(data_dict['dev']['questions'], 'dev questions', args.max_question_len, theta)
            dev_answer_tokens, dev_answer_tuples = tokenize(data_dict['dev']['answers'], 'dev answers', args.max_answer_len, theta)

            test_question_tokens, test_question_tuples = tokenize(data_dict['test']['questions'], 'test questions', args.max_question_len, theta)
            test_candidate_tokens, test_candidate_tuples = tokenize(data_dict['test']['candidates'], 'test candidates', args.max_answer_len, theta)
            test_ground_truth = data_dict['test']['ground_truth']
            # ground truth shape [#questions,1]

            print(f'train question: {data_dict["train"]["questions"][0]}')
            print(f'train question token: {train_question_tokens[0]}')
            print(f'train question tuples: {train_question_tuples[0]}')
            print(f'train answer: {data_dict["train"]["answers"][0]}')
            print(f'train answer token: {train_answer_tokens[0]}')
            print(f'train answer tuples: {train_answer_tuples[0]}')
            print()
              
            saved_data = {
                'train':{
                    'question_tuples': train_question_tuples,
                    'question_tokens': train_question_tokens,
                    'answer_tuples': train_answer_tuples,
                    'answer_tokens': train_answer_tokens
                },
                'dev':{
                    'question_tuples': dev_question_tuples,
                    'question_tokens': dev_question_tokens,
                    'answer_tuples': dev_answer_tuples,
                    'answer_tokens': dev_answer_tokens
                },
                'test':{
                    'question_tuples': test_question_tuples,
                    'question_tokens': test_question_tokens,
                    'candidate_tuples': test_candidate_tuples,
                    'candidate_tokens': test_candidate_tokens,
                    'ground_truth': test_ground_truth
                }
            }
            logger.info("Saving processed instances to %s", cached_insts_file)
            torch.save(saved_data, cached_insts_file)
        
        def count_unknown_tokens(tuples_of_entity_ids):
            total_unknown_tokens = 0
            total_tokens = 0
            
            for i, tuple in enumerate(tuples_of_entity_ids):
                for j, entity_id in enumerate(tuple["entity_ids"]):
                    if entity_id == 1:
                        total_unknown_tokens = total_unknown_tokens + 1
                    if entity_id != 0 and entity_id != 2:
                        total_tokens = total_tokens + 1

            return total_unknown_tokens, total_tokens

        total_unknown_tokens, total_tokens = count_unknown_tokens(train_question_tuples)
        print(f'with theta = {args.theta}, train_question_tuples has {total_unknown_tokens} total_unknown_tokens, {total_tokens} total_tokens, with available token rate {1-total_unknown_tokens/total_tokens:.3f}. One sentence have on average {(total_tokens-total_unknown_tokens)/len(train_question_tuples):.3f} annotated entities.')
        total_unknown_tokens, total_tokens = count_unknown_tokens(train_answer_tuples) 
        print(f'with theta = {args.theta}, train_answer_tuples has {total_unknown_tokens} total_unknown_tokens, {total_tokens} total_tokens, with available token rate {1-total_unknown_tokens/total_tokens:.3f}. One sentence have on average {(total_tokens-total_unknown_tokens)/len(train_answer_tuples):.3f} annotated entities.')
        total_unknown_tokens, total_tokens = count_unknown_tokens(dev_question_tuples)
        print(f'with theta = {args.theta}, dev_question_tuples has {total_unknown_tokens} total_unknown_tokens, {total_tokens} total_tokens, with available token rate {1-total_unknown_tokens/total_tokens:.3f}. One sentence have on average {(total_tokens-total_unknown_tokens)/len(dev_question_tuples):.3f} annotated entities.')
        total_unknown_tokens, total_tokens = count_unknown_tokens(dev_answer_tuples)
        print(f'with theta = {args.theta}, dev_answer_tuples has {total_unknown_tokens} total_unknown_tokens, {total_tokens} total_tokens, with available token rate {1-total_unknown_tokens/total_tokens:.3f}. One sentence have on average {(total_tokens-total_unknown_tokens)/len(dev_answer_tuples):.3f} annotated entities.')
        total_unknown_tokens, total_tokens = count_unknown_tokens(test_question_tuples)
        print(f'with theta = {args.theta}, test_question_tuples has {total_unknown_tokens} total_unknown_tokens, {total_tokens} total_tokens, with available token rate {1-total_unknown_tokens/total_tokens:.3f}. One sentence have on average {(total_tokens-total_unknown_tokens)/len(test_question_tuples):.3f} annotated entities.')
        total_unknown_tokens, total_tokens = count_unknown_tokens(test_candidate_tuples)
        print(f'with theta = {args.theta}, test_candidate_tuples has {total_unknown_tokens} total_unknown_tokens, {total_tokens} total_tokens, with available token rate {1-total_unknown_tokens/total_tokens:.3f}. One sentence have on average {(total_tokens-total_unknown_tokens)/len(test_candidate_tuples):.3f} annotated entities.')
        print()


        args.num_train_instances = int(len(train_question_tuples) * args.data_rate)
        logger.info(f"number of train instances {args.num_train_instances}")
        logger.info(f"number of dev instances {len(dev_question_tuples)}")
        logger.info(f"number of test instances {len(test_question_tuples)}")
        logger.info(f"preparing torch dataset")
        # train dataset
        train_dataset = ReqaDataset(
                args = args,
                question_insts=train_question_tuples[:args.num_train_instances],
                answer_insts=train_answer_tuples[:args.num_train_instances]
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=args.shuffle,
            batch_size=args.batch_size,
            collate_fn=args.train_fn)

        # dev dataset
        dev_dataset = ReqaDataset(
                args = args,
                question_insts=dev_question_tuples,
                answer_insts=dev_answer_tuples
            )

        dev_loader = torch.utils.data.DataLoader(
            dev_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=args.train_fn)
        
        # test dataset
        test_question_dataset = SentDataset(
                args = args,
                sentence_insts=test_question_tuples
            )

        test_question_loader = torch.utils.data.DataLoader(
            test_question_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=args.eval_fn)
        
        test_candidate_dataset = SentDataset(
                args = args,
                sentence_insts=test_candidate_tuples
            )

        test_candidate_loader = torch.utils.data.DataLoader(
            test_candidate_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=args.eval_fn)

        logger.info(f"finished dataloading process")
        return train_loader, dev_loader, (test_question_loader, test_candidate_loader, test_ground_truth)


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_file', type=str)
    parser.add_argument("--data_rate", type=float)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--mixed_training", action="store_true")

    parser.add_argument('--max_question_len', type=int)
    parser.add_argument('--max_answer_len', type=int)
    
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)

    parser.add_argument('--plm_path', type=str)
    # plm_path is not used in the code. 
    # If you have local checkpoint files, add this argument, and modify code concerning initialization of models.

    parser.add_argument('--encoder_type', type=str)
    parser.add_argument('--pooler_type', type=str)
    parser.add_argument('--load_model', default='')
    parser.add_argument('--save_model', default=None)

    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--warmup_proportion', type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument('--seed', type=int)
    parser.add_argument('--shuffle', dest='shuffle', action='store_true')
    parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
    parser.add_argument("--theta", default=0.1, type=float)

    parser.set_defaults(shuffle=True)
    
    args = parser.parse_args()
    logger.info(args)
    print(f'{args.shuffle}, argument shuffle')
    # random seed
    set_seed(args)

    # perparing data processing func
    from utils_data import reqa_fn, sent_fn, reqa_luke_fn, sent_luke_fn
    args.train_fn = reqa_fn
    args.eval_fn = sent_fn

    if 'know' in args.encoder_type:
        args.train_fn = reqa_luke_fn
        args.eval_fn = sent_luke_fn

    # loading dataset
    train_loader, dev_loader, test_loader = prepare_dataloaders(args)
    
    # set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # preparing model
    from models.dual_encoder import RankModel, RankModelKnow, RankModelAddScore, RankModelGatedMatching, RankModelKnowledgePair
    if 'know' not in args.encoder_type:
        model = RankModel(args)
    if 'know' in args.encoder_type:
        model = RankModelKnow(args) 
        # model = RankModelAddScore(args)
        # model = RankModelGatedMatching(args)
    
    if args.load_model != '':
        model.load_state_dict(torch.load(args.load_model)['model'])
        logger.info('load model successfully!')
    model.to(args.device)

    # running
    run(model, train_loader, dev_loader, test_loader, args)


if __name__ == '__main__':
    main()
