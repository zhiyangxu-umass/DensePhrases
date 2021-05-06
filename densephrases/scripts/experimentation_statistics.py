import argparse
import copy
import json
import os

import jsonlines


def get_test_data_map(test_data_file):
    with open(test_data_file, 'r') as f:
        data = json.load(f)['data']
        data_map = {}
        for rec in data:
            data_map[rec['id'].strip()] = {
                'question': rec['question'],
                'output': [
                    {
                        'answer': answer.lower().strip(),
                        'provenances': [
                            {
                                'title': prov['title'].strip().lower(),
                                'sec_title': prov['section_title'].strip()
                                                 .replace('Section::::', '')  # Remove Section
                                                 .split(':')[0]  # Ignore subsections
                                [:-1].lower(),  # Remove dot at end
                                'para_id': int(prov['paragraph_id']),
                            }
                            for prov in rec['provenances'][i]]
                    }
                    for i, answer in enumerate(rec['answers'])]
            }

        print(f"\n\nData Map: \n\n{data[0]} \n\n=> data_map[{data[0]['id']}]={data_map[data[0]['id']]}")
        return data_map


def get_pred_output(pred_out_file):
    with jsonlines.open(pred_out_file) as reader:
        output = []
        first = None
        for line in reader:
            if first is None:
                first = line
            output.append(
                {
                    'qid': line['q_id'].strip(),
                    'question': line['question'].lower().strip(),
                    'output': [
                        {
                            'answer': line['pred_answer'][i].strip().lower(),
                            'title': line['title'][i][0].strip().lower(),
                            # Empty sec_title indicate abstract
                            'sec_title': line['sec_title'][i].strip().lower() if line['sec_title'][i] else 'abstract',
                            'para_id': int(line['para_id'][i]),
                            # If all scores recorded then read from dict else record score as is
                            'mips_score': float(line['score'][i]['score']) if type(line['score'][i]) is dict else float(
                                line['score'][i]),
                            # Only available if all scores recorded in dict, also for title_weight=0.0, same as mips
                            'title_rerank_score': (float(line['score'][i]['title_rerank_score'])
                                                   if 'title_rerank_score' in line['score'][i]
                                                   else float(line['score'][i]['score']))
                            if type(line['score'][i]) is dict else None,
                        }
                        for i in range(len(line['pred_answer']))
                    ]
                }
            )
        print(f"\n\nPrediction output: \n\n{first} => \n\n{output[0]}")
        return output


def get_gold_provenance_with_max_hits(gold_output, pred_output):
    max_hits = -1
    best_prov = None
    if len(gold_output['provenances']) == 0:
        print(
            f"\n\nget_gold_provenance_with_max_hits: \n\nGold:{gold_output} \n\nPred:{pred_output} => {best_prov} {max_hits}")
        return best_prov, max_hits
    for prov in gold_output['provenances']:
        hits = 0
        if prov['title'] == pred_output['title']:
            hits += 1
            if prov['sec_title'] == pred_output['sec_title']:
                hits += 1
                if prov['para_id'] == pred_output['para_id']:
                    hits += 1
                    # Only the first match is sufficient
                    return prov, max_hits
        if hits > max_hits:
            max_hits = hits
            best_prov = prov
    print(
        f"\n\nget_gold_provenance_with_max_hits: \n\nGold:{gold_output} \n\nPred:{pred_output} => {best_prov} {max_hits}")
    return best_prov, max_hits


def get_gold_output_with_max_hits(gold_output_list, pred_output):
    max_meta_hits = -1
    ans_hit = False  # If any of the answers match
    final_ans, final_prov = None, None
    for gold_output in gold_output_list:
        if gold_output['answer'] == pred_output['answer']:
            best_prov, hits = get_gold_provenance_with_max_hits(gold_output, pred_output)
            if not ans_hit:
                # If this is first hit, this overrides all the previous outputs
                final_ans, final_prov, max_meta_hits = gold_output['answer'], best_prov, hits
                ans_hit = True
                continue
        else:
            if ans_hit:  # If there is a answer hit, misses can be ignored
                continue
            best_prov, hits = get_gold_provenance_with_max_hits(gold_output, pred_output)
        # For all other cases, check if the max_meta_hits is larger.
        if hits > max_meta_hits:
            final_ans, final_prov, max_meta_hits = gold_output['answer'], best_prov, hits
        # If max hits received and ans is already matched, then stop scanning
        if ans_hit and max_meta_hits == 3:
            break
        # There should be atleast one answer set. No need to set prov it happens when hit=max_hit=-1 and ans=None
        if final_ans is None:
            final_ans = gold_output['answer']
    final_output = {'answer': final_ans}
    # If none of the answers had provenance, then ignore the metadata.
    if final_prov is not None:
        final_output.update(final_prov)
    print(
        f"\n\nget_gold_output_with_max_hits: \n\nGold:{gold_output_list} \n\nPred:{pred_output} => {final_output} {max_meta_hits} {ans_hit}")
    return final_output, max_meta_hits, ans_hit


def get_stat_skeleton():
    substat_skeleton = {
        'no_prov': {
            'examples': [],
            'total': 0
        },
        'title_hit': {
            'sec_hit': {
                'para_hit': {
                    'examples': [],
                    'total': 0
                },
                'para_miss': {
                    'examples': [],
                    'total': 0
                },
                'total': 0,
            },
            'sec_miss': {
                'examples': [],
                'total': 0
            },
            'total': 0
        },
        'title_miss': {
            'examples': [],
            'total': 0
        },
        'total': 0
    }
    return {
        'ans_hit': copy.deepcopy(substat_skeleton),
        'ans_miss': copy.deepcopy(substat_skeleton),
        'total': 0,
        'skipped': 0
    }


# Select only some representative examples
def get_repr_stat(stat, show=1):
    org_stat = copy.deepcopy(stat)

    def get_repr_substat(stat):
        stat['no_prov']['examples'] = stat['no_prov']['examples'][:show]
        stat['title_miss']['examples'] = stat['no_prov']['examples'][:show]
        stat = stat['title_hit']
        stat['sec_miss']['examples'] = stat['sec_miss']['examples'][:show]
        stat = stat['sec_hit']
        stat['para_miss']['examples'] = stat['para_miss']['examples'][:show]
        stat['para_hit']['examples'] = stat['para_hit']['examples'][:show]
        return stat

    org_stat['ans_hit'] = get_repr_substat(org_stat['ans_hit'])
    org_stat['ans_miss'] = get_repr_substat(org_stat['ans_miss'])
    return org_stat


def update_stats(stat, hits, ans_hit, stat_el):
    stat['total'] += 1
    stat = stat['ans_hit'] if ans_hit else stat['ans_miss']
    stat['total'] += 1
    if hits == -1:
        stat['no_prov']['total'] += 1
        stat['no_prov']['examples'].append(stat_el)
        return
    if hits == 0:
        stat['title_miss']['total'] += 1
        stat['title_miss']['examples'].append(stat_el)
        return
    stat = stat['title_hit']
    stat['total'] += 1
    if hits == 1:
        stat['sec_miss']['total'] += 1
        stat['sec_miss']['examples'].append(stat_el)
        return
    stat = stat['sec_hit']
    stat['total'] += 1
    if hits == 2:
        stat['para_miss']['total'] += 1
        stat['para_miss']['examples'].append(stat_el)
    else:
        stat['para_hit']['total'] += 1
        stat['para_hit']['examples'].append(stat_el)


def generate_stats(data_map, pred_out_list, eval_top_k=10):
    stat = get_stat_skeleton()
    for preds_out in pred_out_list:
        gold_data = data_map[preds_out['qid']]
        best_pred = {'qid': preds_out['qid'], 'question': preds_out['question'],
                     'gold_output': None, 'pred_output': None, 'rank': -1}
        max_meta_hits, any_ans_hit = -1, False
        for i, pred_out in enumerate(preds_out['output'][:eval_top_k]):
            output, hits, ans_hit = get_gold_output_with_max_hits(gold_data['output'], pred_out)
            # If hits are more or there is ans hit for the first time as it overrides previous non-hits
            if hits > max_meta_hits or (ans_hit and not any_ans_hit):
                max_meta_hits, any_ans_hit = hits, ans_hit
                best_pred['gold_output'], best_pred['pred_output'], best_pred['rank']= output, pred_out, i

        print('\n\nBefore stat:', stat)
        update_stats(stat, max_meta_hits, any_ans_hit, best_pred)
        print('\n\nAfter update', stat)
    stat['skipped'] = len(data_map) - stat['total']
    return stat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_input_file', type=str)
    parser.add_argument('--pred_file', type=str)
    parser.add_argument('--top_k', default=10, type=int, help="Top k results for evaluation.")
    parser.add_argument('--stat_out_file', type=str)
    args = parser.parse_args()
    assert os.path.exists(args.test_input_file)
    assert os.path.exists(args.pred_file)

    pred_out = get_pred_output(args.pred_file)
    test_data_map = get_test_data_map(args.test_input_file)
    stat = generate_stats(test_data_map, pred_out, args.top_k)
    print('Statistics with representative examples are as follows')
    print(json.dumps(get_repr_stat(stat, show=1), indent=3))

    with open(args.stat_out_file, 'w') as f:
        print('Writing detailed stats to ', args.stat_out_file)
        json.dump(stat, f)
