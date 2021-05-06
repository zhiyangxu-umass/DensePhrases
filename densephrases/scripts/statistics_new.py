ground_truth_file = "nq_test_preprocessed.json"
pred_truth_file = "pred_file_0.5.jsonl"


import jsonlines

gt_dict = {}
obj_list = []

def getAnswerList(obj) :
    temp = len(obj['answers'])    
    ret_list = []    
    for i in range(0,temp) :
        if ( obj['answers'][i] != "" and obj['provenances'][i] != [] ) :
            t = {}
            t["answer"] = obj['answers'][i]
            
## add functionality of multiple provenances

            t["title"] = (obj['provenances'][i][0])["title"]
            t["section"] = (obj['provenances'][i][0])["section_title"] 
            t["paragraph"] = (obj['provenances'][i][0])["paragraph_id"]
            t["wiki_id"] = (obj['provenances'][i][0])["wikipedia_id"]
            ret_list.append(t)
            
    return ret_list
        

with jsonlines.open(ground_truth_file) as reader:
    for obj_temp in reader:
        for obj in obj_temp["data"] :
            obj_list.append(obj)
            temp_str = (obj["question"].lower()).strip()
            gt_dict[temp_str] = getAnswerList(obj)
        
gt_dict["Who designed the garden city of new Earswick".lower()]




pred_list = []

with jsonlines.open(pred_truth_file) as reader:
    for pred_obj in reader:
            temp = {}
            temp["question"] = pred_obj["question"]
            temp["score_list"] = pred_obj["score"]
            temp["answer_list"] = pred_obj["pred_answer"]
            temp["title_list"] = pred_obj["title"]
            temp["sec_title_list"] = pred_obj["sec_title"]           
            temp["para_id_list"] = pred_obj["para_id"]
            
            pred_list.append(temp)
            
            
            
def getPredAnswerList(pred) :
    length = len(pred["answer_list"])  
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
            ans_match = False
            title_match = False
            section_match = False
            para_match = False
            
            if (ans['answer']==pred['answer']) :
                ans_match = True
                exact_match = True
                count += 1
                if(ans['title']==pred['title']) :
                    title_match = True
                    count += 1
                    if(pred['section'] in ans['section']):
                        section_match = True
                        count += 1
                        if(ans['paragraph']==pred['paragraph']):
                            para_match = True
                            count += 1
                            
            if(count >= max_count) :
                max_count = count
                max_gt = ans
                max_pred = pred
                
    if (max_count==0) :
        for ans in ans_list :
            for pred in pred_list :
                count = 0
                title_match = False
                section_match = False
                para_match = False
            
                if(ans['title']==pred['title']) :
                    title_match = True
                    count += 1
                    if(pred['section'] in ans['section']):
                        section_match = True
                        count += 1
                        if(ans['paragraph']==pred['paragraph']):
                            para_match = True
                            count += 1
                            
                if(count >= max_count) :
                    max_count = count
                    max_gt = ans
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
    
print("============================================================")
    
true_cases = stat_dict[True][0] + stat_dict[True][1] + stat_dict[True][2] + stat_dict[True][3]
false_cases = stat_dict[False][0] + stat_dict[False][1] + stat_dict[False][2] + stat_dict[False][3]
    
print("Exact Match accuracy is", true_cases/(true_cases + false_cases))

             
    
    
        
    
            
            
        
        