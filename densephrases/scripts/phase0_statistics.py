import jsonlines

gt_dict = {}
obj_list = []

def getAnswerList(obj) :
    temp = obj['output']    
    ret_list = []    
    for ans_dict in temp :
        if ( "answer" in list(ans_dict.keys()) and "provenance" in list(ans_dict.keys()) ) :
            t = {}
            t["answer"] = ans_dict["answer"]
            provs = []
            for prov in ans_dict["provenance"]:
                k = {}
                k["title"] = prov["title"]
                k["section"] = prov["section"]
                k["paragraph"] = prov["start_paragraph_id"]
                k["wiki_id"] = prov["wikipedia_id"]
                provs.append(k)
            t["provenance"] = provs
            ret_list.append(t)
    return ret_list
        

with jsonlines.open('nq-dev-kilt.jsonl') as reader:
    for obj in reader:
        obj_list.append(obj)
        temp_str = (obj["input"].lower()).strip()
        gt_dict[temp_str] = getAnswerList(obj)


# Batches into prediction list
pred_list = []
topK = 10

with jsonlines.open('prediction_all.jsonl') as reader:
    for pred_obj in reader:
        for i in range(0,12):
            temp = {}
            temp["question"] = pred_obj["question"][i]
            temp["score_list"] = pred_obj["score"][i]
            temp["answer_list"] = pred_obj["pred_answer"][i]
            temp["title_list"] = pred_obj["title"][i]
            temp["sec_title_list"] = pred_obj["sec_title"][i]            
            temp["para_id_list"] = pred_obj["para_id"][i]
            
            pred_list.append(temp)
            
            
            
def getPredAnswerList(pred) :
    length = min(len(pred["answer_list"]), topK)
    ret_list = []    
    for i in range(0,length) :
            t = {}
            t["answer"] = pred["answer_list"][i]
            t["title"] = pred["title_list"][i][0]
            t["section"] = ("Abstract" if (pred["sec_title_list"][i]=='') \
                                            else pred["sec_title_list"][i])
            t["paragraph"] = pred["para_id_list"][i]
            t["score"] = pred["score_list"][i]
            ret_list.append(t)
            
    return ret_list
            
pred_dict = {} 
                   
for pred in pred_list :
    question = (pred['question'].lower()).strip()    
    pred_dict[question] = getPredAnswerList(pred)
    
compare_list = []
    
for question in list(pred_dict.keys()) :
   if question in list(gt_dict.keys()) :
        compare_list.append( (question,gt_dict[question],pred_dict[question]) )
   else:
       print(question)
        
def getStat(tup) :
    ques, ans_list, pred_list = tup 
    
    max_count = 0
    max_pred = {}
    max_gt = {}
    
    exact_match = False
    
    for ans in ans_list :
        for pred in pred_list :
            count = 0
            if (ans['answer']==pred['answer']) :
                exact_match = True
                count += 1
                best_prov = {}
                for prov in ans['provenance']:
                    temp_count = 1
                    if(prov['title']==pred['title']) :
                        temp_count += 1
                        if(pred['section'] in prov['section']):
                            temp_count += 1
                            if(prov['paragraph']==pred['paragraph']):
                                temp_count += 1
                    if (temp_count > count):
                        count = temp_count
                        best_prov = prov
                            
                if(count > max_count) :
                    max_count = count
                    ans1 = ans.copy()
                    ans1.pop("provenance")
                    ans1.update(best_prov)
                    max_gt = ans1
                    max_pred = pred
                
    if (max_count==0) :
        for ans in ans_list :
            for pred in pred_list :
                count = 0
                best_prov = {}
                for prov in ans['provenance']:
                    temp_count = 0
                    if(prov['title']==pred['title']) :
                        temp_count += 1
                        if(pred['section'] in prov['section']):
                            temp_count += 1
                            if(prov['paragraph']==pred['paragraph']):
                                temp_count += 1
                    if (temp_count > count):
                        count = temp_count
                        best_prov = prov
                            
                if(count > max_count) :
                    max_count = count
                    ans1 = ans.copy()
                    ans1.pop("provenance")
                    ans1.update(best_prov)
                    max_gt = ans1
                    max_pred = pred
                    
    if exact_match == True : 
        max_count = max_count - 1
        
    return exact_match , max_count , max_gt , max_pred

final_res = []

for t in compare_list :
    q,gt_list,pred_list = t
    a,b,c,d = getStat(t)
    final_res.append( (a,b,c,d,q) )


stat_dict = {}
stat_dict[True] = {0:0,1:0,2:0,3:0}
stat_dict[False] = {0:0,1:0,2:0,3:0}
try_list = []
for temp in final_res :
    a,b,c,d,e = temp
    (stat_dict[a])[b] += 1

print(stat_dict)