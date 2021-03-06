import argparse
import json
import os

from densephrases.utils.squad_utils import TrueCaser

def filter_uncached_entries(input_data_file, input_cache_file, out_file):
    input_cache = {}
    with open(input_cache_file, 'r') as f:
        input_cache = json.load(f)
    filtered_entries = []
    with open(input_data_file, 'r') as f:
        for entry in json.load(f)["data"]:
            ques = format_query(entry['question'])
            if len(entry['answers']) == 0:
                continue
            if ques not in input_cache:
                filtered_entries.append(entry)
    print(f'Saving {len(filtered_entries)} uncached questions.')
    print('Writing to %s\n' % out_file)
    with open(out_file, 'w') as f:
        json.dump({'data': filtered_entries}, f)


def format_query(query):
    query = query[:-1] if query.endswith('?') else query
    if args.do_lower_case:
        query = query.lower()
    if args.truecase:
        query = truecase.get_true_case(query) if query == query.lower() else query
    return query


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_file', type=str,
                        default=os.path.join(os.environ['DPH_DATA_DIR'], 'open-qa/nq-open/nq_test_preprocessed.json'))
    parser.add_argument('--input_cache', type=str,
                        default=os.path.join(os.environ['DPH_SAVE_DIR'],
                                             'dph-nqsqd-pb2_dev_wiki/dump/start-pq/1048576_flat_PQ96_8/nq_test_preprocessed_pq_cache.json'))
    parser.add_argument('--truecase', default=True, action='store_true')
    parser.add_argument("--truecase_path", default='truecase/english_with_questions.dist', type=str)
    parser.add_argument("--do_lower_case", default=False, action='store_true')
    args = parser.parse_args()
    assert os.path.exists(args.input_data_file)
    assert os.path.exists(args.input_cache)
    truecase = TrueCaser(os.path.join(os.environ['DPH_DATA_DIR'], args.truecase_path))
    out_file = os.path.join(os.path.dirname(args.input_data_file),
                            os.path.basename(args.input_cache).replace('_cache', ''))
    filter_uncached_entries(args.input_data_file, args.input_cache, out_file)
