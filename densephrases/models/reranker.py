import copy
import logging
from abc import ABC, abstractmethod
from time import time

import numpy as np
import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Reranker(ABC):
    def __init__(self, rerank_weight, modify_original_score=True, rerank_score_field_name=None, cuda=False,
                 logging_level=logging.INFO):

        self.rerank_weight = rerank_weight
        self.emb_score_field_name = rerank_score_field_name
        self.modify_original_score = modify_original_score
        # Options
        logger.setLevel(logging_level)
        self.cuda = cuda
        if self.cuda:
            assert torch.cuda.is_available(), f"Cuda availability {torch.cuda.is_available()}"
            self.device = torch.device('cuda')
        else:
            self.device = torch.device("cpu")

    @abstractmethod
    def get_extra_embed(self, result):
        pass

    def rerank(self, query, results):
        if self.rerank_weight == 0.0:
            logger.info("No reranking is weight is 0.0")
            return results
        start_time = time()
        # Construct start and end queries
        num_queries = query.shape[0]
        results_per_query = len(results[0])
        query = np.tile(np.expand_dims(query, 1), [1, results_per_query, 1])  # batch, results , 2 * size
        query_start, query_end = np.split(query, 2, axis=2)  # batch , results, size

        # Construct start/end embedding vectors and intial score vectors of embedding array and results
        start_extra_emb, end_extra_emb, mips_scores = [], [], []
        for out in results:
            extra_emb_list = []
            scores = []
            for each in out:
                extra_emb_list.append(self.get_extra_embed(each))

                scores.append(each['score'])
            start_extra_emb_per, end_extra_emb_per = np.split(np.stack(extra_emb_list), 2,
                                                              axis=1)  # results, size
            start_extra_emb.append(start_extra_emb_per)
            end_extra_emb.append(end_extra_emb_per)
            mips_scores.append(np.stack(scores))
        start_extra_emb, end_extra_emb, mips_scores = np.stack(start_extra_emb), np.stack(end_extra_emb), np.stack(
            mips_scores)
        print(start_extra_emb.shape, end_extra_emb.shape, query_start.shape, query_end.shape, mips_scores.shape)
        assert start_extra_emb.shape == end_extra_emb.shape == query_start.shape == query_end.shape
        assert start_extra_emb.shape[0] == end_extra_emb.shape[0] == mips_scores.shape[0] == num_queries

        logger.debug(f'1) {time() - start_time:.3f}s: formed extra embed vector')

        # Compute start and end scores
        with torch.no_grad():
            start_extra_emb = torch.FloatTensor(start_extra_emb).to(self.device)
            end_extra_emb = torch.FloatTensor(end_extra_emb).to(self.device)
            query_start_t = torch.FloatTensor(query_start).to(self.device)
            query_end_t = torch.FloatTensor(query_end).to(self.device)
            mips_scores_t = torch.FloatTensor(mips_scores).to(self.device)
            print(start_extra_emb.shape, end_extra_emb.shape, query_start_t.shape, query_end_t.shape)
            start_emb_scores = (query_start_t * start_extra_emb).sum(2)
            end_emb_scores = (query_end_t * end_extra_emb).sum(2)
            print(start_emb_scores.shape, end_emb_scores.shape)
            rerank_start_scores = (1 - self.rerank_weight) * mips_scores_t + self.rerank_weight * start_emb_scores
            rerank_end_scores = (1 - self.rerank_weight) * mips_scores_t + self.rerank_weight * end_emb_scores
            rerank_start_scores = rerank_start_scores.cpu().numpy()
            rerank_end_scores = rerank_end_scores.cpu().numpy()
            # Average the two out to get final rerank scores
            rerank_scores = np.mean(np.array([rerank_start_scores, rerank_end_scores]), axis=0)
            assert rerank_scores.shape == rerank_start_scores.shape == rerank_end_scores.shape

        logger.debug(f'2) {time() - start_time:.3f}s: Computed rerank scores')

        for i in range(num_queries):
            for j in range(len(results[i])):
                if self.modify_original_score:
                    results[i][j]['score'] = rerank_scores[i][j]
                else:
                    results[i][j][self.emb_score_field_name] = rerank_scores[i][j]

        # Sort output
        for i in range(num_queries):
            results[i] = sorted(results[i],
                                key=lambda each_out: -each_out['score'] if self.modify_original_score else -each_out[
                                    self.emb_score_field_name])

        logger.debug(f'3) {time() - start_time:.3f}s: Reranking done')
        return results
