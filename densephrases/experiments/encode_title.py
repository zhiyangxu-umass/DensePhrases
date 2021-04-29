import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoTokenizer,
)

from densephrases.models import DensePhrases
from densephrases.utils.embed_utils import get_question_results
from densephrases.utils.single_utils import backward_compat
from densephrases.utils.squad_utils import get_question_dataloader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def load_query_encoder(device, args):
    assert args.query_encoder_path

    # Configure paths for query encoder serving
    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.pretrained_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.pretrained_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # Pre-trained DensePhrases
    model = DensePhrases(
        config=config,
        tokenizer=tokenizer,
        transformer_cls=MODEL_MAPPING[config.__class__],
    )
    try:
        model.load_state_dict(backward_compat(
            torch.load(os.path.join(args.query_encoder_path, 'pytorch_model.bin'), map_location=torch.device('cpu'))
        ))
    except Exception as e:
        print(e)
        model.load_state_dict(torch.load(os.path.join(args.query_encoder_path, 'pytorch_model.bin')), strict=False)
    model.to(device)

    logger.info(f'DensePhrases loaded from {args.query_encoder_path} having {MODEL_MAPPING[config.__class__]}')
    logger.info('Number of model parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))
    return model, tokenizer


def get_query2vec(query_encoder, tokenizer, args, batch_size=64):
    device = 'cuda' if args.cuda else 'cpu'
    def query2vec(queries):
        question_dataloader, question_examples, query_features = get_question_dataloader(
            queries, tokenizer, args.max_query_length, batch_size=batch_size
        )
        question_results = get_question_results(
            question_examples, query_features, question_dataloader, device, query_encoder, batch_size=batch_size
        )
        if args.debug:
            logger.info(f"{len(query_features)} queries: {' '.join(query_features[0].tokens_)}")
        outs = []
        for qr_idx, question_result in enumerate(question_results):
            out = (
                question_result.start_vec, question_result.end_vec, query_features[qr_idx].tokens_
            )
            outs.append(out)
        return outs
    return query2vec

def embed_all_title(title, args, query_encoder, tokenizer, batch_size=2048):
    query2vec = get_query2vec(
        query_encoder=query_encoder, tokenizer=tokenizer, args=args, batch_size=batch_size
    )

    all_outs = []
    for q_idx in tqdm(range(0, len(title), batch_size)):
        outs = query2vec(title[q_idx:q_idx+batch_size])
        all_outs += outs
    start = np.concatenate([out[0] for out in all_outs], 0)
    end = np.concatenate([out[1] for out in all_outs], 0)
    tokens = [out[2] for out in all_outs]
    query_vec = np.concatenate([start, end], 1)
    logger.info(f'Query reps: {query_vec.shape}')
    return query_vec, tokens

def load_doc_title(title_path, args):
    """
    Data format: jsonl {"wiki_id": xxxxx, "title":"Apple Inc.", "abstract": "Apple Inc. is an American multinational technology company headquartered in Cupertino, California, that designs, develops, and sells consumer electronics, computer software, and online services." }
    """
    logger.info(f'Loading titles from {title_path}')
    title = []
    wiki2idx= {}
    with open(title_path) as fin:
        for idx, line in enumerate(fin):
            line = json.loads(line)
            wiki2idx[line['wiki_id']] = idx 
            if args.combine:
                text = line["title"]+" "+line["abstract"][0]
            else:
                text = line["title"]
            if args.do_lower_case:
                title.append(text.lower())
            else:
                title.append(text)
    
    logger.info(f'Finish loading {len(title)} titles')
    return wiki2idx, title

def create_title_emb(args, query_encoder=None, tokenizer=None):
    wiki2idx, titles = load_doc_title(args.title_path, args)
    if query_encoder is None:
        print(f'Query encoder will be loaded from {args.query_encoder_path}')
        device = 'cuda' if args.cuda else 'cpu'
        query_encoder, tokenizer = load_query_encoder(device, args)
    title_vec, title_text = embed_all_title(titles, args, query_encoder, tokenizer)
    out_dir = os.path.join(os.environ['DPH_DATA_DIR'],'kilt_ks_wikidump/title_emb')
    np.save(os.path.join(out_dir,'title_emb.npy'),title_vec)
    with open(os.path.join(out_dir,'wiki2idx.json'),'w') as fout:
        json.dump(wiki2idx,fout)
    with open(os.path.join(out_dir,'title_text.json'),'w') as fout:
        json.dump(title_text,fout)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # QueryEncoder
    parser = argparse.ArgumentParser()
    # QueryEncoder
    parser.add_argument('--model_type', default='bert', type=str)
    parser.add_argument("--pretrained_name_or_path", default='SpanBERT/spanbert-base-cased', type=str)
    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--do_lower_case", default=False, action='store_true')
    parser.add_argument('--max_query_length', default=64, type=int)
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--query_encoder_path", default='', type=str)
    parser.add_argument("--query_port", default='-1', type=str)

    # PhraseIndex
    parser.add_argument('--dump_dir', default='dump')
    parser.add_argument('--phrase_dump_dir', default=None)
    parser.add_argument('--phrase_dir', default='phrase')
    parser.add_argument('--index_dir', default='256_flat_SQ4')
    parser.add_argument('--index_name', default='index.faiss')
    parser.add_argument('--idx2id_name', default='idx2id.hdf5')
    parser.add_argument('--index_port', default='-1', type=str)

    # These can be dynamically changed.
    parser.add_argument('--max_answer_length', default=10, type=int)
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--nprobe', default=256, type=int)
    parser.add_argument('--truecase', default=False, action='store_true')
    parser.add_argument("--truecase_path", default='truecase/english_with_questions.dist', type=str)

    # KILT
    parser.add_argument('--is_kilt', default=False, action='store_true')
    parser.add_argument('--kilt_gold_path', default='kilt/trex/trex-dev-kilt.jsonl')
    parser.add_argument('--title2wikiid_path', default='wikidump/title2wikiid.json')
    
    # Serving options
    parser.add_argument('--examples_path', default='examples.txt')

    # Training query encoder
    parser.add_argument('--train_path', default=None)
    parser.add_argument('--per_gpu_train_batch_size', default=48, type=int)
    parser.add_argument('--num_train_epochs', default=10, type=float)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--warmup_steps", default=0.1, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--use_inbatch_neg", action="store_true", help="Whether to run with inb-neg.")
    parser.add_argument("--fp16", action="store_true", help="For fp16")
    parser.add_argument('--output_dir', default=None, type=str)

    # Evaluation
    parser.add_argument('--dev_path', default='open-qa/nq-open/dev_preprocessed.json')
    parser.add_argument('--test_path', default='open-qa/nq-open/test_preprocessed.json')
    parser.add_argument('--pred_output_file', default='./prediction_eval_dump.jsonl')
    parser.add_argument('--candidate_path', default=None)
    parser.add_argument('--regex', default=False, action='store_true')
    parser.add_argument('--eval_batch_size', default=512, type=int)

    # Run mode
    parser.add_argument('--run_mode', default='train_query')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--draft', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--seed', default=1992, type=int)
    parser.add_argument("--title_path", default=os.path.join(os.environ['DPH_DATA_DIR'],'kilt_ks_wikidump/title_emb/titles.jsonl'), type=str)
    parser.add_argument('--combine', default=True, action='store_true')
    args = parser.parse_args()

    
    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        create_title_emb(args)
    else:
        raise NotImplementedError
