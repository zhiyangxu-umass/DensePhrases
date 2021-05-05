import json
import logging
import os

import numpy as np

from densephrases.models.reranker import Reranker

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class TitleBasedReranker(Reranker):
    def __init__(self, title_emb_path, title_rerank_weight, modify_original_score=True,
                 cuda=False,
                 logging_level=logging.INFO):
        super().__init__(title_rerank_weight, modify_original_score, "title_rerank_score", cuda, logging_level)

        logger.info(f"reading title emb from: {title_emb_path}")
        self.wiki2id_ = json.load(open(os.path.join(title_emb_path, 'wiki2idx.json'), 'r'))
        self.title_emb = np.load(os.path.join(title_emb_path, 'title_emb.npy'))

    def get_extra_embed(self, result):
        title_id = self.wiki2id_[result["wiki_idx"]]
        return self.title_emb[title_id]
