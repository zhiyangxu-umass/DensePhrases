import json
import logging
import os
import math
import pickle
import string
from functools import partial
from multiprocessing import Pool, cpu_count
from time import time

import numpy as np
from tqdm import tqdm

from .file_utils import is_tf_available, is_torch_available
from .data_utils import DataProcessor, whitespace_tokenize
from torch.utils.data import DataLoader, SequentialSampler


if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    # tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
    tok_answer_text = " ".join([sw for w in whitespace_tokenize(orig_answer_text) for sw in tokenizer.tokenize(w)])

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def squad_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length, is_training, context_only,
        question_only, append_title, skip_no_answer):
    features = []

    # start_time = time()
    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            return []

    if tokenizer.padding_side == 'left':
        print('please debug left side padding')
        raise NotImplementedError

    # Query features
    if not context_only:
        all_query_tokens = []
        for (i, token) in enumerate(example.query_tokens):
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                all_query_tokens.append(sub_token)
        all_query_tokens = all_query_tokens[:max_query_length]

        query_len = max_query_length
        query_dict = tokenizer.encode_plus(
            all_query_tokens,
            max_length=max_query_length,
            return_overflowing_tokens=False,
            pad_to_max_length=True,
            stride=query_len,
            truncation_strategy="only_first",
            return_token_type_ids=True, # TODO: token type ids is zero for query
        )

        if tokenizer.pad_token_id in query_dict["input_ids"]:
            non_padded_ids_ = query_dict["input_ids"][: query_dict["input_ids"].index(tokenizer.pad_token_id)]
        else:
            non_padded_ids_ = query_dict["input_ids"]
        tokens_ = tokenizer.convert_ids_to_tokens(non_padded_ids_)

    # Context Features
    if not question_only:
        all_title_tokens = []
        for (i, token) in enumerate(example.title_tokens):
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                all_title_tokens.append(sub_token)
        all_title_tokens = all_title_tokens[:10] # max title token = 10

        tok_to_orig_index = [] #word piece index -> orginal token index 
        orig_to_tok_index = [] #original token index -> word piece
        all_doc_tokens = []
        #one example is one paragraph in a doc
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        # Add negatives when there are
        all_neg_tokens = []
        all_neg_title_tokens = []
        if len(example.neg_doc_tokens) > 0:
            for (i, token) in enumerate(example.neg_doc_tokens):
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    all_neg_tokens.append(sub_token)

            for (i, token) in enumerate(example.neg_title_tokens):
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    all_neg_title_tokens.append(sub_token)

        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
            )

        sequence_added_tokens = ( # 2 for BERT
            tokenizer.max_len - tokenizer.max_len_single_sentence # + 1 # TODO: Why +1 for roberta?
            if "roberta" in str(type(tokenizer)) or "camembert" in str(type(tokenizer))
            else tokenizer.max_len - tokenizer.max_len_single_sentence
        )
        sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair # 3 for BERT
        added_token = {
            'seq': sequence_added_tokens, 'seq_pair': sequence_pair_added_tokens,
        }
        # title_offset = lambda x: (len(all_title_tokens) + added_token[x]) * int(append_title) + 1 * (1 - int(append_title))
        title_offset = (
            lambda x: (len(all_title_tokens) + added_token[x]) * int(append_title) + \
                      (added_token[x] - 1) * (1 - int(append_title))
        )

        spans = []
        span_doc_tokens = all_doc_tokens
        while len(spans) * doc_stride < len(all_doc_tokens):
            if not append_title:
                encoded_dict = tokenizer.encode_plus(
                    span_doc_tokens,
                    max_length=max_seq_length,
                    return_overflowing_tokens=True,
                    pad_to_max_length=True,
                    stride=max_seq_length - doc_stride - sequence_added_tokens, # This makes stride = doc_stride
                    truncation_strategy="only_first",
                    return_token_type_ids=True,
                )
            else:
                encoded_dict = tokenizer.encode_plus(
                    all_title_tokens,
                    span_doc_tokens,
                    max_length=max_seq_length,
                    return_overflowing_tokens=True,
                    pad_to_max_length=True,
                    stride=max_seq_length - doc_stride - title_offset('seq_pair'),
                    truncation_strategy="only_second",
                    return_token_type_ids=True,
                )

                if len(all_neg_tokens) > 0:
                    neg_encoded_dict = tokenizer.encode_plus(
                        all_neg_title_tokens,
                        all_neg_tokens,
                        max_length=max_seq_length,
                        return_overflowing_tokens=False, # Just use the first span
                        pad_to_max_length=True,
                        stride=max_seq_length - doc_stride - title_offset('seq_pair'),
                        truncation_strategy="only_second",
                        return_token_type_ids=True,
                    )

            # length of paragraph except special tokens (and title)
            paragraph_len = min(
                len(all_doc_tokens) - len(spans) * doc_stride,
                max_seq_length - title_offset('seq_pair'),
            )

            if tokenizer.pad_token_id in encoded_dict["input_ids"]:
                if tokenizer.padding_side == "right":
                    non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
                else:
                    last_padding_id_position = (
                        len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                    )
                    non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]

            else:
                non_padded_ids = encoded_dict["input_ids"]

            tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)
            if len(all_neg_tokens) > 0:
                neg_tokens = tokenizer.convert_ids_to_tokens(neg_encoded_dict["input_ids"])

            token_to_orig_map = {}
            for i in range(paragraph_len):
                index = i + title_offset('seq')
                token_to_orig_map[index] = tok_to_orig_index[len(spans) * (doc_stride) + i]

            # encoded_dict["paragraph_len"] = paragraph_len
            encoded_dict["tokens"] = tokens
            encoded_dict["token_to_orig_map"] = token_to_orig_map
            encoded_dict["title_offset"] = title_offset('seq')
            encoded_dict["token_is_max_context"] = {}
            encoded_dict["start"] = len(spans) * (doc_stride)
            encoded_dict["length"] = paragraph_len

            if len(all_neg_tokens) > 0:
                encoded_dict["neg_input_ids"] =  neg_encoded_dict["input_ids"]
                encoded_dict["neg_attention_mask"] =  neg_encoded_dict["attention_mask"]
                encoded_dict["neg_tokens"] = neg_tokens
                encoded_dict["neg_token_type_ids"] =  neg_encoded_dict["token_type_ids"]

            # For query
            if not context_only:
                encoded_dict["input_ids_"] = query_dict["input_ids"]
                encoded_dict["attention_mask_"] = query_dict["attention_mask"]
                encoded_dict["tokens_"] = tokens_
                encoded_dict["token_type_ids_"] = query_dict['token_type_ids']
                encoded_dict["query_len"] = query_len

            spans.append(encoded_dict)

            if "overflowing_tokens" not in encoded_dict:
                break
            span_doc_tokens = encoded_dict["overflowing_tokens"]

        for doc_span_index in range(len(spans)):
            for j in range(spans[doc_span_index]["length"]):
                is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
                index = (
                    j + spans[doc_span_index]["title_offset"]
                )
                spans[doc_span_index]["token_is_max_context"][index] = is_max_context

        for span_idx, span in enumerate(spans):
            # Identify the position of the CLS token
            cls_index = span["input_ids"].index(tokenizer.cls_token_id)

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            # We skip p_mask for now
            p_mask = np.array(span["token_type_ids"])

            # p_mask = np.minimum(p_mask, 1)

            if tokenizer.padding_side == "right":
                # Limit positive values to one
                # p_mask = 1 - p_mask
                pass

            # p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1

            # Set the CLS index to '0'
            # p_mask[cls_index] = 0

            span_is_impossible = example.is_impossible
            start_position = -1
            end_position = -1
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = span["start"]
                doc_end = span["start"] + span["length"] - 1
                out_of_span = False

                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True

                if out_of_span:
                    start_position = -1 # cls_index
                    end_position = -1 # cls_index
                    span_is_impossible = True
                else:
                    if tokenizer.padding_side == "left":
                        doc_offset = last_padding_id_position + 1
                    else:
                        # doc_offset = len(truncated_query) + sequence_added_tokens
                        doc_offset = title_offset('seq')

                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

                    if (span['tokens'][start_position:end_position+1] != \
                            all_doc_tokens[tok_start_position:tok_end_position+1]):
                        logger.warning(
                            "Tokenization strange: '%s' vs. '%s'", span['tokens'][start_position:end_position+1],
                            all_doc_tokens[tok_start_position:tok_end_position+1]
                        )
                    '''
                    if ''.join(span['tokens'][start_position:end_position+1]) != cleaned_answer_text.replace(' ', ''):
                        logger.warning(
                            "Answer strange: '%s' vs. '%s' vs '%s'", ''.join(span['tokens'][start_position:end_position+1]),
                            cleaned_answer_text.replace(' ', ''), actual_text.replace(' ', '')
                        )
                    '''


            # For filter training
            # if span_is_impossible and is_training:
            #     continue

            features.append(
                SquadFeatures(
                    span["input_ids"],
                    span["attention_mask"],
                    np.maximum(np.array(span["token_type_ids"]), 1).tolist(),
                    span["input_ids_"] if not context_only else None,
                    span["attention_mask_"] if not context_only else None,
                    span["token_type_ids_"] if not context_only else None,
                    cls_index,
                    p_mask.tolist(),
                    example_index=0,  # Can not set unique_id and example_index. They will be set after multiple processing.
                    unique_id=0,
                    paragraph_len=span["length"],
                    token_is_max_context=span["token_is_max_context"],
                    tokens=span["tokens"],
                    tokens_=span["tokens_"] if not context_only else None,
                    token_to_orig_map=span["token_to_orig_map"],
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible,
                    qas_id=example.qas_id,
                    span_idx=span_idx,
                    doc_idx=example.doc_idx,
                    wiki_idx=example.wiki_idx,
                    para_idx=example.para_idx,
                    par_idx=example.par_idx,
                    sec_idx=example.sec_idx,
                    sec_title=example.sec_title,
                    title_offset=span["title_offset"],
                    neg_tokens=span["neg_tokens"] if len(example.neg_doc_tokens) > 0 else None,
                    neg_input_ids=span["neg_input_ids"] if len(example.neg_doc_tokens) > 0 else None,
                    neg_attention_mask=span["neg_attention_mask"] if len(example.neg_doc_tokens) > 0 else None,
                    neg_token_type_ids=np.maximum(
                        np.array(span["neg_token_type_ids"]), 1).tolist() if len(example.neg_doc_tokens) > 0 else None,
                )
            )
    else:
        cls_index = query_dict["input_ids"].index(tokenizer.cls_token_id)

        # logger.info(f'prepro 0) {time()-start_time}')
        assert not context_only
        features.append(
            SquadFeatures(
                None,
                None,
                None,
                query_dict["input_ids"],
                query_dict["attention_mask"],
                query_dict["token_type_ids"],
                cls_index,
                np.array(query_dict["token_type_ids"]).tolist(), # Just put anything
                example_index=0,  # Can not set unique_id and example_index. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=0,
                token_is_max_context=None,
                tokens=None,
                tokens_=tokens_,
                token_to_orig_map=None,
                start_position=None,
                end_position=None,
                is_impossible=False,
                qas_id=example.qas_id,
                span_idx=None,
                doc_idx=None,
                par_idx=None,
            )
        )

    return features


def squad_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def squad_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    return_dataset=False,
    threads=1,
    tqdm_enabled=True,
    context_only=False,
    question_only=False,
    append_title=False,
    skip_no_answer=False,
    max_q=None, # not used
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi


    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    # Defining helper methods
    features = []
    threads = min(threads, cpu_count())
    # start_time = time()
    if threads > 1:
        with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
            annotate_ = partial(
                squad_convert_example_to_features,
                max_seq_length=max_seq_length,
                doc_stride=doc_stride,
                max_query_length=max_query_length,
                is_training=is_training,
                context_only=context_only,
                question_only=question_only,
                append_title=append_title,
                skip_no_answer=skip_no_answer,
            )
            features = list(
                tqdm(
                    p.imap(annotate_, examples, chunksize=32),
                    total=len(examples),
                    desc="convert squad examples to features",
                    disable=not tqdm_enabled,
                )
            )
    else:
        squad_convert_example_to_features_init(tokenizer)
        features = [squad_convert_example_to_features(
            example,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=is_training,
            context_only=context_only,
            question_only=question_only,
            append_title=append_title,
            skip_no_answer=skip_no_answer,
        ) for example in examples]

    # logger.info(f'prepro 1) {time()-start_time}')
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(
        features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features

    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

        # Context-side features
        if not question_only:
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            all_title_offset = torch.tensor([f.title_offset for f in features], dtype=torch.long)
        else: # Just copy for shape/type consistency
            pass
            # all_input_ids = torch.tensor([f.input_ids_ for f in features], dtype=torch.long)
            # all_attention_masks = torch.tensor([f.attention_mask_ for f in features], dtype=torch.long)
            # all_token_type_ids = torch.tensor([f.token_type_ids_ for f in features], dtype=torch.long)
            # all_title_offset = torch.tensor([1 for f in features], dtype=torch.long)

        # Neg-context features
        if features[0].neg_input_ids is not None:
            all_neg_input_ids = torch.tensor([f.neg_input_ids for f in features], dtype=torch.long)
            all_neg_attention_masks = torch.tensor([f.neg_attention_mask for f in features], dtype=torch.long)
            all_neg_token_type_ids = torch.tensor([f.neg_token_type_ids for f in features], dtype=torch.long)
        else: # Just copy for shape/type consistency
            if not question_only:
                all_neg_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
                all_neg_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
                all_neg_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            else:
                pass
                # all_neg_input_ids = torch.tensor([f.input_ids_ for f in features], dtype=torch.long)
                # all_neg_attention_masks = torch.tensor([f.attention_mask_ for f in features], dtype=torch.long)
                # all_neg_token_type_ids = torch.tensor([f.token_type_ids_ for f in features], dtype=torch.long)

        # Question-side features
        if not context_only:
            all_input_ids_ = torch.tensor([f.input_ids_ for f in features], dtype=torch.long)
            all_attention_masks_ = torch.tensor([f.attention_mask_ for f in features], dtype=torch.long)
            all_token_type_ids_ = torch.tensor([f.token_type_ids_ for f in features], dtype=torch.long)
        else: # Just copy for shape/type consistency
            all_input_ids_ = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_masks_ = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids_ = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

        if not is_training:
            if question_only:
                all_feature_index_ = torch.arange(all_input_ids_.size(0), dtype=torch.long)
                dataset = TensorDataset(
                    all_input_ids_, all_attention_masks_, all_token_type_ids_, all_feature_index_
                )
            else:
                all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
                dataset = TensorDataset(
                    all_input_ids, all_attention_masks, all_token_type_ids,
                    all_feature_index, all_cls_index, all_p_mask,
                    all_input_ids_, all_attention_masks_, all_token_type_ids_,
                    all_title_offset,
                )
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index, all_p_mask, all_is_impossible,
                all_input_ids_, all_attention_masks_, all_token_type_ids_,
                all_title_offset,
                all_neg_input_ids, all_neg_attention_masks, all_neg_token_type_ids,
            )

        return features, dataset
    elif return_dataset == "tf":
        raise NotImplementedError()
        if not is_tf_available():
            raise RuntimeError("TensorFlow must be installed to return a TensorFlow dataset.")

        def gen():
            for i, ex in enumerate(features):
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                        "feature_index": i,
                        "qas_id": ex.qas_id,
                    },
                    {
                        "start_position": ex.start_position,
                        "end_position": ex.end_position,
                        "cls_index": ex.cls_index,
                        "p_mask": ex.p_mask,
                        "is_impossible": ex.is_impossible,
                    },
                )

        # Why have we split the batch into a tuple? PyTorch just has a list of tensors.
        train_types = (
            {
                "input_ids": tf.int32,
                "attention_mask": tf.int32,
                "token_type_ids": tf.int32,
                "feature_index": tf.int64,
                "qas_id": tf.string,
            },
            {
                "start_position": tf.int64,
                "end_position": tf.int64,
                "cls_index": tf.int64,
                "p_mask": tf.int32,
                "is_impossible": tf.int32,
            },
        )

        train_shapes = (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
                "feature_index": tf.TensorShape([]),
                "qas_id": tf.TensorShape([]),
            },
            {
                "start_position": tf.TensorShape([]),
                "end_position": tf.TensorShape([]),
                "cls_index": tf.TensorShape([]),
                "p_mask": tf.TensorShape([None]),
                "is_impossible": tf.TensorShape([]),
            },
        )

        return tf.data.Dataset.from_generator(gen, train_types, train_shapes)
    else:
        raise NotImplementedError()
        return features


class SquadProcessor(DataProcessor):
    """
    Processor for the SQuAD data set.
    Overriden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and version 2.0 of SQuAD, respectively.
    """

    train_file = None
    dev_file = None

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        if not evaluate:
            answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
            answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
            answers = []
        else:
            answers = [
                {"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
                for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text"])
            ]

            answer = None
            answer_start = None

        return SquadExample(
            qas_id=tensor_dict["id"].numpy().decode("utf-8"),
            question_text=tensor_dict["question"].numpy().decode("utf-8"),
            context_text=tensor_dict["context"].numpy().decode("utf-8"),
            answer_text=answer,
            start_position_character=answer_start,
            title=tensor_dict["title"].numpy().decode("utf-8"),
            answers=answers,
        )

    def get_examples_from_dataset(self, dataset, evaluate=False):
        """
        Creates a list of :class:`~transformers.data.processors.squad.SquadExample` using a TFDS dataset.

        Args:
            dataset: The tfds dataset loaded from `tensorflow_datasets.load("squad")`
            evaluate: boolean specifying if in evaluation mode or in training mode

        Returns:
            List of SquadExample

        Examples::

            import tensorflow_datasets as tfds
            dataset = tfds.load("squad")

            training_examples = get_examples_from_dataset(dataset, evaluate=False)
            evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
        """

        if evaluate:
            dataset = dataset["validation"]
        else:
            dataset = dataset["train"]

        examples = []
        for tensor_dict in tqdm(dataset):
            examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))

        return examples

    def get_train_examples(self, data_dir, filename=None, draft=False, context_only=False, skip_no_answer=False,
            max_q=None, args=None):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """
        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]

        return self._create_examples(
                input_data, "train", draft, context_only=context_only, skip_no_answer=skip_no_answer, args=args)

    def get_dev_examples(self, data_dir, filename=None, draft=False, context_only=False, args=None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        print('### read from this dir ',os.path.join(data_dir, self.dev_file if filename is None else filename))
        with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "dev", draft, context_only=context_only, args=args)

    def _create_examples(self, input_data, set_type, draft, context_only, draft_num_examples=1000, min_len=0, max_len=2500,
            skip_no_answer=False, args=None):
        is_training = set_type == "train"
        examples = []
        length_stat = []
        max_cut_cnt = 0
        min_cut_cnt = 0
        non_para_cnt = 0
        impossible_cnt = 0
        total_cnt = 0
        no_neg_cnt = 0
        truecase = TrueCaser(os.path.join(os.environ['DPH_DATA_DIR'], args.truecase_path))

        for doc_idx, entry in tqdm(enumerate(input_data)):
            title = entry["title"]
            short_context = []
            par_idx = 0
            wiki_idx = entry.get('wikipedia_id',None)
            for _, paragraph in enumerate(entry["paragraphs"]):
                context_text = paragraph["context"]
                para_idx = paragraph.get("paragraph_id",None)
                # if title == 'Frank Partridge (soldier)':
                #     print(para_idx)
                sec_idx = paragraph.get("section_id",None)
                sec_title = paragraph.get("section_title",None)
                if not sec_title is None:
                    sec_title = sec_title.strip().replace('Section::::','')[:-1]
                if ' ' in context_text: # non-breaking space '\xa0' to ' '
                    context_text = context_text.replace(' ', ' ')
                total_cnt += 1

                # For minor cases in nq-reader/train.json
                if type(title) == list:
                    if len(title) > 1:
                        logger.info(f'multiple titles: {title}')
                    title = title[0]
                title = title.strip().replace('_', ' ')

                # Used for embedding context
                if context_only:
                    # Skipping non-para or unanswerable questions
                    if 'is_paragraph' in paragraph:
                        if not paragraph['is_paragraph']:
                            non_para_cnt += 1
                            continue
                    #TODO wire squad example with dev data
                    example = SquadExample(
                        context_text=context_text,
                        title=title,
                        doc_idx=doc_idx,
                        wiki_idx=wiki_idx,
                        par_idx=par_idx,
                        para_idx = int(para_idx)-1,
                        sec_idx=sec_idx,
                        sec_title=sec_title,
                        qas_id=None,
                        question_text=None,
                        answer_text=None,
                        start_position_character=None,
                    )
                    examples.append(example)
                    length_stat.append(len(context_text))
                    par_idx += 1

                    if draft and len(examples) == draft_num_examples:
                        return examples
                    continue

                # Pre-processing questions
                for qa in paragraph["qas"]:
                    print('### question',qa)
                    if "id" not in qa:
                        assert len(paragraph["qas"]) == 1
                        qas_id = paragraph["id"]
                    else:
                        qas_id = qa["id"]
                    question_text = qa["question"].strip()
                    if question_text.endswith('?'):
                        question_text = question_text[:-1]
                    if question_text == question_text.lower():
                        question_text = truecase.get_true_case(question_text)
                    start_position_character = None
                    answer_text = None
                    answers = []

                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = True if len(qa['answers']) == 0 else False

                    if not is_impossible:
                        if is_training:
                            assert type(qa["answers"]) == dict or type(qa["answers"]) == list, type(qa["answers"])
                            if type(qa["answers"]) == dict:
                                qa["answers"] = [qa["answers"]]
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                            if ' ' in answer_text:
                                answer_text = answer_text.replace(' ', ' ')
                            assert context_text[start_position_character:start_position_character+len(answer_text)] == \
                                    answer_text
                        else:
                            answers = qa["answers"]
                    else:
                        impossible_cnt += 1
                        # We skip unanswerables for filter training
                        if skip_no_answer:
                            continue

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        doc_idx=doc_idx,
                        par_idx=par_idx,
                        is_impossible=is_impossible,
                        answers=answers,
                    )
                    examples.append(example)
                    par_idx += 1

                    if draft and len(examples) == draft_num_examples:
                        logger.info(f'no-ans / all example: {impossible_cnt}/{len(examples)} (skipping:{skip_no_answer})')
                        logger.info(f'no neg count: {no_neg_cnt} out of {len(examples)}')
                        return examples

            if par_idx == 0 and context_only:
                logger.info(f'{title} (doc_idx: {doc_idx}) does not have any paragraphs saved')

        if context_only:
            logger.info(f'mean/std length of context: {np.mean(length_stat):.2f} / {np.std(length_stat):.2f}')
            logger.info(f'min/max length of context: {min(length_stat)} / {max(length_stat)}')
            logger.info(f'para/non-para: {len(length_stat)} / {non_para_cnt}')
            logger.info(f'min/max cut cnt: {min_cut_cnt} / {max_cut_cnt}')
            assert min_cut_cnt + max_cut_cnt + non_para_cnt + len(length_stat) == total_cnt

        logger.info(f'no-ans / all example: {impossible_cnt} vs. {len(examples)} (skipping: {skip_no_answer})')
        logger.info(f'no neg count: {no_neg_cnt} out of {len(examples)}')
        return examples


class SquadV1Processor(SquadProcessor):
    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"


class SquadV2Processor(SquadProcessor):
    train_file = "train-v2.0.json"
    dev_file = "dev-v2.0.json"


class SquadExample(object):
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id=None,
        question_text=None,
        context_text=None,
        neg_context_text=None,
        answer_text=None,
        start_position_character=None,
        title=None,
        neg_title=None,
        doc_idx=None,
        wiki_idx = None,
        sec_idx=None,
        sec_title=None,
        par_idx=None,
        para_idx=None,
        answers=[],
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.doc_idx = doc_idx
        self.wiki_idx = wiki_idx
        self.par_idx = par_idx
        self.para_idx = para_idx
        self.sec_idx=sec_idx
        self.sec_title=sec_title
        self.question_text = question_text
        self.context_text = context_text
        self.neg_context_text = neg_context_text
        self.answer_text = answer_text
        self.title = title
        self.neg_title = neg_title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = -1, -1 # 0, 0

        # Split on whitespace so that different tokens may be attributed to their original position.
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        if self.context_text is not None:
            for c in self.context_text:
                if _is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            self.doc_tokens = doc_tokens #[tok, tok ,tok]
            self.char_to_word_offset = char_to_word_offset #[0,0,0,1,1,2,3,4,4,4]

        # Same pre-processing for title tokens
        title_tokens = []
        prev_is_whitespace = True
        if self.title is not None:
            for c in self.title:
                if _is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        title_tokens.append(c)
                    else:
                        title_tokens[-1] += c
                    prev_is_whitespace = False

            self.title_tokens = title_tokens

        # Same pre-processing for neg tokens
        self.neg_doc_tokens = []
        prev_is_whitespace = True
        if self.neg_context_text is not None:
            for c in self.neg_context_text:
                if _is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        self.neg_doc_tokens.append(c)
                    else:
                        self.neg_doc_tokens[-1] += c
                    prev_is_whitespace = False

        # Same pre-processing for title tokens
        self.neg_title_tokens = []
        prev_is_whitespace = True
        if self.neg_title is not None:
            for c in self.neg_title:
                if _is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        self.neg_title_tokens.append(c)
                    else:
                        self.neg_title_tokens[-1] += c
                    prev_is_whitespace = False

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]

        # For queries, do the same
        if self.question_text is not None:
            query_tokens = []
            prev_is_whitespace = True

            # Split on whitespace so that different tokens may be attributed to their original position.
            for q in self.question_text:
                if _is_whitespace(q):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        query_tokens.append(q)
                    else:
                        query_tokens[-1] += q
                    prev_is_whitespace = False

            self.query_tokens = query_tokens


class SquadFeatures(object):
    """
    Single squad example features to be fed to a model.
    Those features are model-specific and can be crafted from :class:`~transformers.data.processors.squad.SquadExample`
    using the :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        input_ids_,
        attention_mask_,
        token_type_ids_,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        tokens_,
        token_to_orig_map,
        start_position,
        end_position,
        is_impossible,
        qas_id: str = None,
        span_idx=None, # Doc span (strieded) idx
        par_idx=None, # paragraph idx
        para_idx = None,
        doc_idx=None,
        wiki_idx=None,
        sec_idx=None,
        sec_title=None,
        title_offset=None,
        neg_tokens=None,
        neg_input_ids=None,
        neg_attention_mask=None,
        neg_token_type_ids=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.input_ids_ = input_ids_
        self.attention_mask_ = attention_mask_
        self.token_type_ids_ = token_type_ids_
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.tokens_ = tokens_
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.qas_id = qas_id
        self.para_idx = para_idx
        self.wiki_idx = wiki_idx
        self.span_idx = span_idx
        self.doc_idx = doc_idx
        self.par_idx = par_idx
        self.sec_idx=sec_idx,
        self.sec_title=sec_title,
        self.title_offset = title_offset
        self.neg_tokens = neg_tokens
        self.neg_input_ids = neg_input_ids
        self.neg_attention_mask = neg_attention_mask
        self.neg_token_type_ids = neg_token_type_ids


class SquadResult(object):
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(
        self, unique_id, start_logits, end_logits, sft_logits, eft_logits
    ):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.sft_logits = sft_logits # start filter logits
        self.eft_logits = eft_logits # end filter logits
        self.unique_id = unique_id


class ContextResult(object):
    """
    Constructs a ContextResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_vec: The vector corresponding to the start of the answer
        end_vec: The vector corresponding to the end of the answer
    """

    def __init__(
        self, unique_id, start_vecs, end_vecs, sft_logits, eft_logits
    ):
        self.start_vecs = start_vecs
        self.end_vecs = end_vecs
        self.sft_logits = sft_logits # start filter logits
        self.eft_logits = eft_logits # end filter logits
        self.unique_id = unique_id


class QuestionResult(object):
    """
    Constructs a QuestionResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_vec: The vector corresponding to the start of the answer
        end_vec: The vector corresponding to the end of the answer
    """

    def __init__(self, unique_id, qas_id, input_ids, start_vec, end_vec):
        self.unique_id = unique_id
        self.qas_id = qas_id
        self.input_ids = input_ids
        self.start_vec = start_vec
        self.end_vec = end_vec


class TrueCaser(object):
    def __init__(self, dist_file_path=None):
        """ Initialize module with default data/english.dist file """
        if dist_file_path is None:
            dist_file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data/english_with_questions.dist")

        with open(dist_file_path, "rb") as distributions_file:
            pickle_dict = pickle.load(distributions_file)
            self.uni_dist = pickle_dict["uni_dist"]
            self.backward_bi_dist = pickle_dict["backward_bi_dist"]
            self.forward_bi_dist = pickle_dict["forward_bi_dist"]
            self.trigram_dist = pickle_dict["trigram_dist"]
            self.word_casing_lookup = pickle_dict["word_casing_lookup"]

    def get_score(self, prev_token, possible_token, next_token):
        pseudo_count = 5.0

        # Get Unigram Score
        nominator = self.uni_dist[possible_token] + pseudo_count
        denominator = 0
        for alternativeToken in self.word_casing_lookup[
                possible_token.lower()]:
            denominator += self.uni_dist[alternativeToken] + pseudo_count

        unigram_score = nominator / denominator

        # Get Backward Score
        bigram_backward_score = 1
        if prev_token is not None:
            nominator = (
                self.backward_bi_dist[prev_token + "_" + possible_token] +
                pseudo_count)
            denominator = 0
            for alternativeToken in self.word_casing_lookup[
                    possible_token.lower()]:
                denominator += (self.backward_bi_dist[prev_token + "_" +
                                                      alternativeToken] +
                                pseudo_count)

            bigram_backward_score = nominator / denominator

        # Get Forward Score
        bigram_forward_score = 1
        if next_token is not None:
            next_token = next_token.lower()  # Ensure it is lower case
            nominator = (
                self.forward_bi_dist[possible_token + "_" + next_token] +
                pseudo_count)
            denominator = 0
            for alternativeToken in self.word_casing_lookup[
                    possible_token.lower()]:
                denominator += (
                    self.forward_bi_dist[alternativeToken + "_" + next_token] +
                    pseudo_count)

            bigram_forward_score = nominator / denominator

        # Get Trigram Score
        trigram_score = 1
        if prev_token is not None and next_token is not None:
            next_token = next_token.lower()  # Ensure it is lower case
            nominator = (self.trigram_dist[prev_token + "_" + possible_token +
                                           "_" + next_token] + pseudo_count)
            denominator = 0
            for alternativeToken in self.word_casing_lookup[
                    possible_token.lower()]:
                denominator += (
                    self.trigram_dist[prev_token + "_" + alternativeToken +
                                      "_" + next_token] + pseudo_count)

            trigram_score = nominator / denominator

        result = (math.log(unigram_score) + math.log(bigram_backward_score) +
                  math.log(bigram_forward_score) + math.log(trigram_score))

        return result

    def first_token_case(self, raw):
        return f'{raw[0].upper()}{raw[1:]}'

    def get_true_case(self, sentence, out_of_vocabulary_token_option="title"):
        """ Returns the true case for the passed tokens.
        @param tokens: Tokens in a single sentence
        @param outOfVocabulariyTokenOption:
            title: Returns out of vocabulary (OOV) tokens in 'title' format
            lower: Returns OOV tokens in lower case
            as-is: Returns OOV tokens as is
        """
        tokens = whitespace_tokenize(sentence)

        tokens_true_case = []
        for token_idx, token in enumerate(tokens):

            if token in string.punctuation or token.isdigit():
                tokens_true_case.append(token)
            else:
                token = token.lower()
                if token in self.word_casing_lookup:
                    if len(self.word_casing_lookup[token]) == 1:
                        tokens_true_case.append(
                            list(self.word_casing_lookup[token])[0])
                    else:
                        prev_token = (tokens_true_case[token_idx - 1]
                                      if token_idx > 0 else None)
                        next_token = (tokens[token_idx + 1]
                                      if token_idx < len(tokens) - 1 else None)

                        best_token = None
                        highest_score = float("-inf")

                        for possible_token in self.word_casing_lookup[token]:
                            score = self.get_score(prev_token, possible_token,
                                                   next_token)

                            if score > highest_score:
                                best_token = possible_token
                                highest_score = score

                        tokens_true_case.append(best_token)

                    if token_idx == 0:
                        tokens_true_case[0] = self.first_token_case(tokens_true_case[0])

                else:  # Token out of vocabulary
                    if out_of_vocabulary_token_option == "title":
                        tokens_true_case.append(token.title())
                    elif out_of_vocabulary_token_option == "lower":
                        tokens_true_case.append(token.lower())
                    else:
                        tokens_true_case.append(token)

        return "".join([
            " " +
            i if not i.startswith("'") and i not in string.punctuation else i
            for i in tokens_true_case
        ]).strip()



def read_text_examples(input_file, draft=False, draft_num_examples=12):
    """Read a text file into a list of SquadExample."""
    input_data = []
    with open(input_file, "r") as reader:
        for line in reader:
            input_data.append(line.strip())

    examples = []
    for idx, text in enumerate(input_data):
            # Using only context
            example = SquadExample(
                context_text=text,
                qas_id=str(idx),
                question_text=None,
                answer_text=None,
                start_position_character=None,
                title=None,
                doc_idx=idx,
                par_idx=0,
            )
            examples.append(example)

    if draft:
        return examples[:draft_num_examples]

    logger.info(f'Reading {len(examples)} examples')
    return examples


def get_question_dataloader(questions, tokenizer, max_query_length=64, batch_size=64):
    examples = [SquadExample(qas_id=q_idx, question_text=q) for q_idx, q in enumerate(questions)]
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=None,
        doc_stride=None,
        max_query_length=max_query_length,
        is_training=False,
        return_dataset="pt",
        threads=1,
        question_only=True,
        tqdm_enabled=False,
    )
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)

    return eval_dataloader, examples, features

# create a new dataloader for title
def get_title_dataloader(titles, questions, tokenizer, max_query_length=64, batch_size=64):
    squad_convert_example_to_features_init(tokenizer)
    all_input_ids_list = []
    all_attention_masks_list = []
    all_token_type_ids_list = []
    all_features = []
    all_examples = []
    for t_idx, title in enumerate(titles):
        examples = [SquadExample(qas_id=t_idx, question_text=t) for t in enumerate(title)]
        all_examples.append(examples)
        features = [squad_convert_example_to_features(
            example,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=is_training,
            context_only=context_only,
            question_only=question_only,
            append_title=append_title,
            skip_no_answer=skip_no_answer,
        ) for example in examples]
        new_features = []
        unique_id = 1000000000
        example_index = 0
        for example_features in tqdm(
            features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
        ):
            if not example_features:
                continue
            for example_feature in example_features:
                example_feature.example_index = example_index
                example_feature.unique_id = unique_id
                new_features.append(example_feature)
                unique_id += 1
            example_index += 1
        features = new_features
        del new_features

        # all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        # all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        # all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

        all_input_ids_ = torch.tensor([f.input_ids_ for f in features], dtype=torch.long)
        all_attention_masks_ = torch.tensor([f.attention_mask_ for f in features], dtype=torch.long)
        all_token_type_ids_ = torch.tensor([f.token_type_ids_ for f in features], dtype=torch.long)

        all_input_ids_list.append(all_input_ids_)
        all_attention_masks_list.append(all_attention_masks_)
        all_token_type_ids_list.append(all_token_type_ids_)
    
    title_input_ids_ = torch.stack(all_input_ids_list, dim=0)  # batch_size, 100, seq_length
    title_attention_masks_ = torch.stack(all_attention_masks_list, dim=0)
    title_token_type_ids_ = torch.stack(all_token_type_ids_list, dim=0)

    all_feature_index_ = torch.arange(title_input_ids_.size(0), dtype=torch.long)

    # query dataloader
    examples = [SquadExample(qas_id=t_idx, question_text=t) for t in enumerate(title)]
    features = [squad_convert_example_to_features(
        example,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=is_training,
        context_only=context_only,
        question_only=question_only,
        append_title=append_title,
        skip_no_answer=skip_no_answer,
    ) for example in examples]
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(
        features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features

    # all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    # all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    # all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

    question_input_ids_ = torch.tensor([f.input_ids_ for f in features], dtype=torch.long)
    question_attention_masks_ = torch.tensor([f.attention_mask_ for f in features], dtype=torch.long)
    question_token_type_ids_ = torch.tensor([f.token_type_ids_ for f in features], dtype=torch.long)

    
    dataset = TensorDataset(
        title_input_ids_, title_attention_masks_, title_token_type_ids_, question_input_ids_, question_attention_masks_, question_token_type_ids_, all_feature_index_
    )

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)

    return eval_dataloader




def get_cq_dataloader(contexts, questions, tokenizer, max_query_length=64, batch_size=64):
    examples = [
        SquadExample(qas_id=q_idx, title='dummy',
                     question_text=q, context_text=contexts[q_idx]) for q_idx, q in enumerate(questions)
    ]
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=max_query_length,
        is_training=False,
        return_dataset="pt",
        threads=1,
        question_only=False,
        tqdm_enabled=False,
    )
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)

    return eval_dataloader, examples, features


def get_bertqa_dataloader(contexts, questions, tokenizer, max_query_length=64, batch_size=64):
    from transformers import squad_convert_examples_to_features as preprocess

    examples = [
        SquadExample(qas_id=q_idx, title='dummy',
                     question_text=q, context_text=contexts[q_idx]) for q_idx, q in enumerate(questions)
    ]
    features, dataset = preprocess(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=max_query_length,
        is_training=False,
        return_dataset="pt",
        threads=1,
        tqdm_enabled=False,
    )
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)

    return eval_dataloader, examples, features
