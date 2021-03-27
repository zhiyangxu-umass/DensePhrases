import json
import numpy as np

gt_dir = '/mnt/nfs/scratch1/hmalara/DensePhrase_Harsh_Repo/DensePhrases/dph-data/nq-dev-kilt.jsonl'
pre_dir = '/mnt/nfs/scratch1/hmalara/DensePhrase_Harsh_Repo/DensePhrases/outputs/dph-nqsqd-pb2_pq96-nq-10/pred/test_preprocessed_modified_2837.pred'

def read_jsonl(dir_):
    data = {}
    with open(dir_) as f:
        for idx, line in enumerate(f):
            ex = json.loads(line)
            data['test_'+str(idx)] = ex
    return data



def check_coverage(gt, pred):
    total = 0
    ra_wt_stat = [0,0,0,0,0,0,0,0,0,0,0]
    ra_rt_stat = [0,0,0,0,0,0,0,0,0,0,0]
    wa_wt_stat = [0,0,0,0,0,0,0,0,0,0,0]
    wa_rt_stat = [0,0,0,0,0,0,0,0,0,0,0]
    #wrong_stat = [0,0,0,0,0,0,0,0,0,0,0]
    miss = 0
    #same_ans_diff_title = 0
    for key, ex in pred.items():
        pred_titles = ex['title']
        pred_answer = ex['prediction']
        data = gt[key]
        question = ex['question']
        gt_answer = data['output'][0]['answer'].strip()
        proven = data['output'][0].get('provenance',None)
        if not proven is None:
            total += 1
            gt_titles = set([pro['title'].strip() for pro in proven])
            flag = False
            for i, ans_title in enumerate(zip(pred_answer, pred_titles[:10])):
                ans, title = ans_title
                ans = ans.strip()
                #print(title,gt_title)
                #if title[0] in gt_titles:
                     #print(i)
                     #corr_stat[i] += 
                if ans == gt_answer:
                     #corr_stat[i] += not title[0].strip() in gt_titles
                     if not title[0].strip() in gt_titles:
                         ra_wt_stat[i] +=1
                     else:
                         ra_rt_stat[i] +=1
                else:
                     if not title[0].strip() in gt_titles:
                         wa_wt_stat[i] += 1
                         print('question: ',question,' ,gt answer ', gt_answer,' ,pred answer: ',ans, ' ,title: ', title[0],'gt title',gt_titles)
                         wa_rt_stat[i] += 1
                     else:
                         #print('question: ',question,' ,gt answer ', gt_answer,' ,pred answer: ',ans, ' ,title: ', title[0])
                         wa_rt_stat[i] += 1
                #flag = True
                #same_ans_diff_title += ans == gt_answer
                        
            #if not flag:
            #    corr_stat[10]+= 1
            #    wrong_stat[10]+= 1
        else:
            miss +=1
    print('ra wt',np.array(ra_wt_stat)/2653) 
    print('ra rt',np.array(ra_rt_stat)/2653) 
    print('wa wt',np.array(wa_wt_stat)/2653)
    print('wa rt',np.array(wa_rt_stat)/2635)
    return total, miss, wa_wt_stat, wa_rt_stat

gt = read_jsonl(gt_dir)
pred = json.load(open(pre_dir,'r'))
total, miss, corr_stat, wrong_stat = check_coverage(gt, pred)

print('total inst: {}, inst with prov: {}, miss inst: {}, correct stat: {}, wrong stat: {}'.format(len(gt), total, miss, corr_stat, wrong_stat))
        
        
    

