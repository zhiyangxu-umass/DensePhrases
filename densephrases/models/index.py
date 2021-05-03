import os
import logging
import os
import pickle
from time import time
import json

import blosc
import faiss
import h5py
import numpy as np
import torch
from tqdm import tqdm

from densephrases.utils.eval_utils import normalize_answer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class MIPS(object):
    def __init__(self, phrase_dump_dir, index_path, idx2id_path, max_idx, cuda=False, result_cache_dump_path=None,
                 return_cached_results=False, compressed_metadata_path=None, logging_level=logging.INFO):
        self.phrase_dump_dir = phrase_dump_dir

        # Read index
        self.index = {}
        logger.info(f'Reading {index_path}')
        self.index = faiss.read_index(index_path, faiss.IO_FLAG_ONDISK_SAME_DIR)
        self.max_idx = max_idx
        logger.info(f'index ntotal: {self.index.ntotal} | PQ: {"PQ" in index_path}')

        # Read idx2id
        self.idx_f = {}
        logger.info('Load idx2id on memory')
        self.idx_f = self.load_idx_f(idx2id_path)
        self.offset = None
        self.scale = None
        self.doc_groups = None

        # Options
        logger.setLevel(logging_level)
        self.num_docs_list = []
        self.cuda = cuda
        if self.cuda:
            assert torch.cuda.is_available(), f"Cuda availability {torch.cuda.is_available()}"
            self.device = torch.device('cuda')
            logger.info("Load IVF on GPU")
            index_ivf = faiss.extract_index_ivf(self.index)
            quantizer = index_ivf.quantizer
            quantizer_gpu = faiss.index_cpu_to_all_gpus(quantizer)
            index_ivf.quantizer = quantizer_gpu
        else:
            self.device = torch.device("cpu")

        # Load metadata on RAM if possible
        doc_group_path =  compressed_metadata_path # 1 min
        if os.path.exists(doc_group_path) and ('PQ' in index_path):
            logger.info(f"Loading metadata on RAM from {doc_group_path} (for PQ only)")
            self.doc_groups = pickle.load(open(doc_group_path, 'rb'))
        else:
            logger.info(f"Will read metadata directly from hdf5 files (requires SSDs for faster inference)")

        # Set cache for reading or overwriting
        if result_cache_dump_path is not None:
            self.result_cache_path = result_cache_dump_path
            self.result_cache = {}
            self.read_cache_results = return_cached_results
            if os.path.exists(self.result_cache_path):
                logger.info(f"Reading the cached results from {self.result_cache_path}")
                self.result_cache = json.loads(self.result_cache_path)
            else:
                logger.info(f"No cache exists at path {self.result_cache_path}. Cannot read if set to read.")
                self.read_cache_results = False
            self.overwrite_cache = not self.read_cache_results
            print(self.read_cache_results)
            print(self.overwrite_cache)
        else:
            logger.info(f"Cache path is not set.")
            self.read_cache_results = False
            self.overwrite_cache = False
            print(self.overwrite_cache)

    def __del__(self):
        print(self.overwrite_cache)
        if self.overwrite_cache:
            with open(self.result_cache_path, 'w') as f:
                json.dump(self.result_cache, f)

    def load_idx_f(self, idx2id_path):
        idx_f = {}
        types = ['doc', 'word', 'sec', 'para']
        with h5py.File(idx2id_path, 'r') as f:
            for key in tqdm(f, desc='loading idx2id'):
                idx_f_cur = {}
                for type_ in types:
                    idx_f_cur[type_] = f[key][type_][:]
                idx_f[key] = idx_f_cur
            return idx_f

    def open_dumps(self):
        if os.path.isdir(self.phrase_dump_dir):
            self.phrase_dump_paths = sorted(
                [os.path.join(self.phrase_dump_dir, name) for name in os.listdir(self.phrase_dump_dir) if 'hdf5' in name]
            )
            dump_names = [os.path.splitext(os.path.basename(path))[0] for path in self.phrase_dump_paths]
            if '-' in dump_names[0] and ('dev' not in dump_names[0]): # Range check
                self.dump_ranges = [list(map(int, name.split('-'))) for name in dump_names]
        else:
            self.phrase_dump_paths = [self.phrase_dump_dir]
        self.phrase_dumps = [h5py.File(path, 'r') for path in self.phrase_dump_paths]

    def close_dumps(self):
        for phrase_dump in self.phrase_dumps:
            phrase_dump.close()

    def decompress_meta(self, doc_idx):
        dtypes = self.doc_groups[doc_idx]['dtypes'] # needed for store from binary stream
        word2char_start = np.frombuffer(blosc.decompress(self.doc_groups[doc_idx]['word2char_start']), dtypes['word2char_start'])
        word2char_end = np.frombuffer(blosc.decompress(self.doc_groups[doc_idx]['word2char_end']), dtypes['word2char_end'])
        f2o_start = np.frombuffer(blosc.decompress(self.doc_groups[doc_idx]['f2o_start']), dtypes['f2o_start'])
        context = blosc.decompress(self.doc_groups[doc_idx]['context']).decode('utf-8')
        wikipedia_ids = self.doc_groups[doc_idx]['wikipedia_ids'].decode('utf-8')
        section_titles = np.frombuffer(blosc.decompress(self.doc_groups[doc_idx]['section_titles']), dtypes['section_titles'])
        title = self.doc_groups[doc_idx]['title'].decode('utf-8')

        return {
            'word2char_start': word2char_start,
            'word2char_end': word2char_end,
            'f2o_start': f2o_start,
            'context': context,
            'title': title,
            'wikipedia_ids': wikipedia_ids,
            'section_titles': section_titles,
            'offset': -2,
            'scale': 20,
        }

    def get_idxs(self, I):
        max_idx = self.max_idx
        offsets = (I / max_idx).astype(np.int64) * int(max_idx)
        idxs = I % int(max_idx)
        doc = np.array(
            [[self.idx_f[str(offset)]['doc'][idx] for offset, idx in zip(oo, ii)] for oo, ii in zip(offsets, idxs)])
        word = np.array([[self.idx_f[str(offset)]['word'][idx] for offset, idx in zip(oo, ii)] for oo, ii in
                         zip(offsets, idxs)])
        sec = np.array([[self.idx_f[str(offset)]['sec'][idx] for offset, idx in zip(oo, ii)] for oo, ii in
                         zip(offsets, idxs)])
        para = np.array([[self.idx_f[str(offset)]['para'][idx] for offset, idx in zip(oo, ii)] for oo, ii in
                         zip(offsets, idxs)])
        return doc, sec, para, word

    def get_doc_group(self, doc_idx):
        if len(self.phrase_dumps) == 1:
            return self.phrase_dumps[0][str(doc_idx)]
        for dump_range, dump in zip(self.dump_ranges, self.phrase_dumps):
            if dump_range[0] * 1000 <= int(doc_idx) < dump_range[1] * 1000:
                if str(doc_idx) not in dump:
                    raise ValueError('%d not found in dump list' % int(doc_idx))
                return dump[str(doc_idx)]

        # Check last
        if str(doc_idx) not in self.phrase_dumps[-1]:
            raise ValueError('%d not found in dump list' % int(doc_idx))
        else:
            return self.phrase_dumps[-1][str(doc_idx)]

    def int8_to_float(self, num, offset, factor):
        if self.offset is None:
            self.offset = offset
            self.scale = factor
        return num / factor + offset

    def dequant(self, offset, scale, input_):
        return self.int8_to_float(input_, offset, scale)

    def adjust(self, each):
        last = each['context'].rfind(' [PAR] ', 0, each['start_pos'])
        last = 0 if last == -1 else last + len(' [PAR] ')
        next = each['context'].find(' [PAR] ', each['end_pos'])
        next = len(each['context']) if next == -1 else next
        each['context'] = each['context'][last:next]
        each['start_pos'] -= last
        each['end_pos'] -= last
        return each

    def search_dense(self, query, q_texts, nprobe=256, top_k=10):
        batch_size, d = query.shape
        self.index.nprobe = nprobe

        # Stack start/end and benefit from multi-threading
        start_time = time()
        query = query.astype(np.float32)
        query_start, query_end = np.split(query, 2, axis=1)
        query_concat = np.concatenate((query_start, query_end), axis=0)

        # Search with faiss
        b_scores, I = self.index.search(query_concat, top_k)
        b_start_scores, start_I = b_scores[:batch_size,:], I[:batch_size,:]
        b_end_scores, end_I = b_scores[batch_size:,:], I[batch_size:,:]
        logger.debug(f'1) {time()-start_time:.3f}s: MIPS')

        # Get idxs from resulting I
        start_time = time()
        b_start_doc_idxs, b_start_sec_idxs, b_start_para_idxs, b_start_idxs = self.get_idxs(start_I)
        b_end_doc_idxs, b_end_sec_idxs, b_end_para_idxs, b_end_idxs = self.get_idxs(end_I)

        # Number of unique docs
        num_docs = sum(
            [len(set(s_doc.flatten().tolist() + e_doc.flatten().tolist())) for s_doc, e_doc in zip(b_start_doc_idxs, b_end_doc_idxs)]
        ) / batch_size
        self.num_docs_list.append(num_docs)
        logger.debug(f'2) {time()-start_time:.3f}s: get index')

        return b_start_doc_idxs, b_start_sec_idxs, b_start_para_idxs, b_start_idxs, start_I, b_end_doc_idxs, b_end_sec_idxs, b_end_para_idxs, b_end_idxs, end_I, b_start_scores, b_end_scores

    def search_phrase(self, query, start_doc_idxs, start_sec_idxs, start_para_idxs, start_idxs, orig_start_idxs, end_doc_idxs, end_sec_idxs, end_para_idxs, end_idxs, orig_end_idxs, start_scores,
            end_scores, top_k=10, max_answer_length=10, return_idxs=False):

        # Reshape for phrase
        num_queries = query.shape[0]
        query = np.reshape(np.tile(np.expand_dims(query, 1), [1, top_k, 1]), [-1, query.shape[1]])
        q_idxs = np.reshape(np.tile(np.expand_dims(np.arange(num_queries), 1), [1, top_k*2]), [-1])
        start_doc_idxs = np.reshape(start_doc_idxs, [-1]) #batch x top-k, 1
        start_sec_idxs = np.reshape(start_sec_idxs, [-1])
        start_para_idxs = np.reshape(start_para_idxs, [-1])
        start_idxs = np.reshape(start_idxs, [-1])
        orig_start_idxs = np.reshape(orig_start_idxs, [-1])
        end_doc_idxs = np.reshape(end_doc_idxs, [-1])
        end_sec_idxs = np.reshape(end_sec_idxs, [-1])
        end_para_idxs = np.reshape(end_para_idxs, [-1])
        end_idxs = np.reshape(end_idxs, [-1])
        orig_end_idxs = np.reshape(orig_end_idxs, [-1])
        start_scores = np.reshape(start_scores, [-1])
        end_scores = np.reshape(end_scores, [-1])
        assert len(start_doc_idxs) == len(start_idxs) == len(end_idxs) == len(start_scores) == len(start_sec_idxs) == len(start_para_idxs)

        # Set default vec
        start_time = time()
        query_start, query_end = np.split(query, 2, axis=1) # batch x top-k, size
        bs = query_start.shape[1]
        default_doc = [doc_idx for doc_idx in set(start_doc_idxs.tolist() + end_doc_idxs.tolist()) if doc_idx >= 0][0]
        default_vec = np.zeros(bs).astype(np.float32)

        # Get metadata from HDF5
        if self.doc_groups is None:
            self.open_dumps()
            groups = {
                doc_idx: self.get_doc_group(doc_idx)
                for doc_idx in set(start_doc_idxs.tolist() + end_doc_idxs.tolist()) if doc_idx >= 0
            }
            groups_all = {
                doc_idx: {
                    'word2char_start': groups[doc_idx]['word2char_start'][:],
                    'word2char_end': groups[doc_idx]['word2char_end'][:],
                    'f2o_start': groups[doc_idx]['f2o_start'][:],
                    'context': groups[doc_idx].attrs['context'],
                    'title': groups[doc_idx].attrs['title'],
                    'section_titles': groups[doc_idx].attrs['section_titles'],
                    'wikipedia_ids': groups[doc_idx].attrs['wikipedia_ids'],
                    'offset': -2, # fixed
                    'scale': 20,
                } for doc_idx in set(start_doc_idxs.tolist() + end_doc_idxs.tolist()) if doc_idx >= 0
            }

            groups_start = [{'end': np.array([
                groups[doc_idx]['start'][ii] for ii in range(
                    start_idx, min(start_idx+max_answer_length, len(groups[doc_idx]['start'])))
                ])} for doc_idx, start_idx in zip(start_doc_idxs, start_idxs)
            ]
            groups_end = [{'start': np.array([
                groups[doc_idx]['start'][ii] for ii in range(
                    max(0, end_idx-max_answer_length+1), end_idx+1)
                ])} for doc_idx, end_idx in zip(end_doc_idxs, end_idxs)
            ]
            self.close_dumps()

        # Get metadata from RAM
        else:
            groups_all = {
                doc_idx: self.decompress_meta(str(doc_idx))
                for doc_idx in set(start_doc_idxs.tolist() + end_doc_idxs.tolist()) if doc_idx >= 0
            }
            groups_start = [{'end': np.array([
                self.index.reconstruct(ii) for ii in range(
                    start_idx, min(start_idx+max_answer_length, self.index.ntotal))
                ])} for doc_idx, start_idx in zip(start_doc_idxs, orig_start_idxs)
            ]
            groups_end = [{'start': np.array([
                self.index.reconstruct(ii) for ii in range(
                    max(0, end_idx-max_answer_length+1), end_idx+1)
                ])} for doc_idx, end_idx in zip(end_doc_idxs, orig_end_idxs)
            ]
            self.dequant = lambda offset, scale, x: x # no need for dequantization when using reconstruct()
        logger.debug(f'1) {time()-start_time:.3f}s: disk access')

        def valid_phrase(start_idx, end_idx, doc_idx, max_ans_len):
            if doc_idx < 0:
                return False

            if start_idx < 0 or start_idx >= len(groups_all[doc_idx]['f2o_start']):
                return False

            if end_idx < 0 or end_idx >= len(groups_all[doc_idx]['f2o_start']):
                return False

            if groups_all[doc_idx]['f2o_start'][end_idx] - groups_all[doc_idx]['f2o_start'][start_idx] > max_ans_len:
                return False

            if groups_all[doc_idx]['f2o_start'][end_idx] - groups_all[doc_idx]['f2o_start'][start_idx] < 0:
                return False

            return True

        # Find end for start_idxs
        start_time = time()
        ends = [group_start['end'] for start_idx, group_start in zip(start_idxs, groups_start)]
        new_end_idxs = [[
            start_idx+i
            if valid_phrase(start_idx, start_idx+i, doc_idx, max_answer_length) else -1 for i in range(max_answer_length)
            ] for start_idx, doc_idx in zip(start_idxs, start_doc_idxs)
        ]
        end_mask = -1e9 * (np.array(new_end_idxs) < 0)  # [Q, L]
        end = np.zeros((query.shape[0], max_answer_length, default_vec.shape[0]), dtype=np.float32)
        for end_idx, each_end in enumerate(ends):
            end[end_idx, :each_end.shape[0], :] = self.dequant(
                float(groups_all[default_doc]['offset']), float(groups_all[default_doc]['scale']), each_end
            )

        with torch.no_grad():
            end = torch.FloatTensor(end).to(self.device)
            query_end = torch.FloatTensor(query_end).to(self.device)
            new_end_scores = (query_end.unsqueeze(1) * end).sum(2).cpu().numpy()
        scores1 = np.expand_dims(start_scores, 1) + new_end_scores + end_mask  # [Q, L]
        pred_end_idxs = np.stack([each[idx] for each, idx in zip(new_end_idxs, np.argmax(scores1, 1))], 0)  # [Q]
        pred_end_vecs = np.stack([each[idx] for each, idx in zip(end.cpu().numpy(), np.argmax(scores1, 1))], 0)
        logger.debug(f'2) {time()-start_time:.3f}s: find end')

        # Find start for end_idxs
        start_time = time()
        starts = [group_end['start'] for end_idx, group_end in zip(end_idxs, groups_end)]
        new_start_idxs = [[
            end_idx-i
            if valid_phrase(end_idx-i, end_idx, doc_idx, max_answer_length) else -1 for i in range(max_answer_length-1,-1,-1)
            ] for end_idx, doc_idx in zip(end_idxs, end_doc_idxs)
        ]
        start_mask = -1e9 * (np.array(new_start_idxs) < 0)  # [Q, L]
        start = np.zeros((query.shape[0], max_answer_length, default_vec.shape[0]), dtype=np.float32)
        for start_idx, each_start in enumerate(starts):
            start[start_idx, -each_start.shape[0]:, :] = self.dequant(
                float(groups_all[default_doc]['offset']), float(groups_all[default_doc]['scale']), each_start
            )

        with torch.no_grad():
            start = torch.FloatTensor(start).to(self.device)
            query_start = torch.FloatTensor(query_start).to(self.device)
            new_start_scores = (query_start.unsqueeze(1) * start).sum(2).cpu().numpy()
        scores2 = new_start_scores + np.expand_dims(end_scores, 1) + start_mask
        pred_start_idxs = np.stack([each[idx] for each, idx in zip(new_start_idxs, np.argmax(scores2, 1))], 0)  # [Q]
        pred_start_vecs = np.stack([each[idx] for each, idx in zip(start.cpu().numpy(), np.argmax(scores2, 1))], 0)
        logger.debug(f'3) {time()-start_time:.3f}s: find start')

        # Get start/end idxs of phrases
        start_time = time()
        doc_idxs = np.concatenate((np.expand_dims(start_doc_idxs, 1), np.expand_dims(end_doc_idxs, 1)), axis=1).flatten()
        sec_idxs = np.concatenate((np.expand_dims(start_sec_idxs, 1), np.expand_dims(end_sec_idxs, 1)), axis=1).flatten()
        para_idxs = np.concatenate((np.expand_dims(start_para_idxs, 1), np.expand_dims(end_para_idxs, 1)), axis=1).flatten()
        start_idxs = np.concatenate((np.expand_dims(start_idxs, 1), np.expand_dims(pred_start_idxs, 1)), axis=1).flatten()
        end_idxs = np.concatenate((np.expand_dims(pred_end_idxs, 1), np.expand_dims(end_idxs, 1)), axis=1).flatten()
        max_scores = np.concatenate((np.max(scores1, 1, keepdims=True), np.max(scores2, 1, keepdims=True)), axis=1).flatten()

        # Prepare for reconstructed vectors for query-side fine-tuning
        if return_idxs:
            start_vecs = np.concatenate(
                (np.expand_dims(np.stack([group_start['end'][0] for group_start in groups_start]), 1),
                 np.expand_dims(pred_start_vecs, 1)), axis=1
            ).reshape(-1, pred_start_vecs.shape[-1])
            end_vecs = np.concatenate(
                (np.expand_dims(pred_end_vecs, 1),
                 np.expand_dims(np.stack([group_end['start'][-1] for group_end in groups_end]), 1)), axis=1
            ).reshape(-1, pred_end_vecs.shape[-1])

        out = [{
            'context': groups_all[doc_idx]['context'], 'title': [groups_all[doc_idx]['title']], 'doc_idx': doc_idx,
            'wiki_idx': groups_all[doc_idx]['wikipedia_ids'],
            'sec_idx': sec_idx,
            'sec_title': groups_all[doc_idx]['section_titles'][sec_idx],
            'para_idx': para_idx,
            'start_pos': groups_all[doc_idx]['word2char_start'][groups_all[doc_idx]['f2o_start'][start_idx]].item(),
            'end_pos': (groups_all[doc_idx]['word2char_end'][groups_all[doc_idx]['f2o_start'][end_idx]].item()
                if (len(groups_all[doc_idx]['word2char_end']) > 0) and (end_idx >= 0)
                else groups_all[doc_idx]['word2char_start'][groups_all[doc_idx]['f2o_start'][start_idx]].item() + 1),
            'start_idx': start_idx, 'end_idx': end_idx, 'score': score,
            'start_vec': start_vecs[group_idx] if return_idxs else None,
            'end_vec': end_vecs[group_idx] if return_idxs else None,
            } if doc_idx >= 0 else {
                'score': -1e8, 'context': 'dummy', 'start_pos': 0, 'end_pos': 0}
            for group_idx, (doc_idx, sec_idx, para_idx, start_idx, end_idx, score) in enumerate(zip(
                doc_idxs.tolist(), sec_idxs.tolist(), para_idxs.tolist() ,start_idxs.tolist(), end_idxs.tolist(), max_scores.tolist()))
        ]
        for each in out:
            each['answer'] = each['context'][each['start_pos']:each['end_pos']]
        out = [self.adjust(each) for each in out]

        # Sort output
        new_out = [[] for _ in range(num_queries)]
        for idx, each_out in zip(q_idxs, out):
            new_out[idx].append(each_out)
        for i in range(len(new_out)):
            new_out[i] = sorted(new_out[i], key=lambda each_out: -each_out['score'])
            new_out[i] = list(filter(lambda x: x['score'] > -1e5, new_out[i])) # In case of no output but masks
        logger.debug(f'4) {time()-start_time:.3f}s: get metadata')
        return new_out

    def aggregate_results(self, results, top_k=10, q_text=None):
        out = []
        doc_ans = {}
        for r_idx, result in enumerate(results):
            da = f'{result["title"]}_{normalize_answer(result["answer"])}'
            # da = f'{normalize_answer(result["answer"])}' # For KILT, merge based on answer string

            if da not in doc_ans:
                doc_ans[da] = r_idx
            else:
                result['score'] = -1e8
                # if result['title'][0] not in results[doc_ans[da]]['title']: # For KILT, merge doc titles
                #     results[doc_ans[da]]['title'] += result['title']
        results = sorted(results, key=lambda each_out: -each_out['score'])
        results = list(filter(lambda x: x['score'] > -1e5, results))[:top_k] # get real top_k if you want
        return results

    def search(self, query, q_texts=None,
               nprobe=256, top_k=10,
               return_idxs=False,
               max_answer_length=10):
        # Fetch from cache if set
        if self.read_cache_results and q_texts is not None:
            return [self.result_cache[q_text] for q_text in q_texts]

        # MIPS on start/end
        start_time = time()
        start_doc_idxs, start_sec_idxs, start_para_idxs, start_idxs, start_I, end_doc_idxs, end_sec_idxs, end_para_idxs, end_idxs, end_I, start_scores, end_scores = self.search_dense(
            query,
            q_texts=q_texts,
            nprobe=nprobe,
            top_k=top_k,
        )
        logger.debug(f'Top-{top_k} MIPS: {time()-start_time:.3f}s')

        # Search phrase
        start_time = time()
        outs = self.search_phrase(
            query, start_doc_idxs, start_sec_idxs, start_para_idxs, start_idxs, start_I, end_doc_idxs, end_sec_idxs, end_para_idxs, end_idxs, end_I, start_scores, end_scores,
            top_k=top_k, max_answer_length=max_answer_length, return_idxs=return_idxs,
        )
        logger.debug(f'Top-{top_k} phrase search: {time()-start_time:.3f}s')

        # Aggregate
        outs = [self.aggregate_results(results, top_k, q_text) for results, q_text in zip(outs, q_texts)]
        if start_doc_idxs.shape[1] != top_k:
            logger.info(f"Warning.. {start_doc_idxs.shape[1]} only retrieved")

        #Add the result to cache
        if self.overwrite_cache and q_texts is not None:
            for q_text, out in zip(q_texts, outs):
                self.result_cache[q_text] = out

        return outs
