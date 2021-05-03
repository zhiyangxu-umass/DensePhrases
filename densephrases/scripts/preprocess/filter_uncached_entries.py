import argparse
import json
import os


def filter_uncached_entries(input_data_file, input_cache_file, out_file):
    input_cache = open(input_cache_file, 'r')
    filtered_entries = []
    with open(input_data_file, 'r') as f:
        for entry in json.load(f)["data"]:
            ques = entry['question']
            ques = ques[:-1] if ques.endswith('?') else ques
            if len(entry['answers']) == 0:
                continue
            if ques not in input_cache:
                filtered_entries.append(entry)
    print(f'Saving {len(filtered_entries)} uncached questions.')
    print('Writing to %s\n' % out_file)
    with open(out_file, 'w') as f:
        json.dump({'data': filtered_entries}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_file', type=str,
                        default=os.path.join(os.environ['DPH_DATA_DIR'], 'open-qa/nq-open/nq_test_preprocessed.json'))
    parser.add_argument('--input_cache', type=str,
                        default=os.path.join(os.environ['DPH_SAVE_DIR'],
                                             'dph-nqsqd-pb2_dev_wiki/dump/start-pq/1048576_flat_PQ96_8/nq_test_preprocessed_pq_cache.json'))
    args = parser.parse_args()
    assert os.path.exists(args.input_data_file)
    assert os.path.exists(args.input_cache)
    out_file = os.path.join(os.path.dirname(args.input_data_file),
                            os.path.basename(args.input_cache).replace('_cache', ''))
    filter_uncached_entries(args.input_data_file, args.input_cache, out_file)
