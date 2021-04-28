import json
import argparse
import os
from tqdm import tqdm


def preprocess_kilt_jsonl(input_file, out_file):
    with open(input_file) as f:
        for line_idx, line in tqdm(enumerate(f)):
            data = json.loads(line)
            id = data['id']
            question = data['input']
            output = data['output']
            answers = []
            provenances_list = []
            data_to_save = []
            for out in output:
                if "answer" not in out or not out["answer"]:
                    continue
                provs = []
                if 'provenance' in out:
                    for prov in out["provenance"]:
                        k = {}
                        k["title"] = prov["title"]
                        k["section_title"] = prov["section"]
                        k["paragraph_id"] = prov["start_paragraph_id"]
                        k["wikipedia_id"] = prov["wikipedia_id"]
                        provs.append(k)
                provenances_list.append(provs)
                answers.append(out["answer"])
            data_to_save.append({
                'id': id,
                'question': question,
                'answers': answers,
                'provenances': provenances_list
            })
    print(f'Saving {len(data_to_save)} questions.')
    print('Writing to %s\n' % out_file)
    with open(out_file, 'w') as f:
        json.dump({'data': data_to_save}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str,
                        default=os.path.join(os.environ['DPH_DATA_DIR'],'open-qa/nq-open/nq-dev-kilt.jsonl'))
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(os.environ['DPH_DATA_DIR'],'open-qa/nq-open'))
    parser.add_argument('--out_file_name', type=str, default='nq_test_preprocessed.json')
    args = parser.parse_args()
    assert os.path.exists(args.out_dir)
    out_path = os.path.join(args.out_dir, args.out_file_name)
    preprocess_kilt_jsonl(args.input_file, out_path)
