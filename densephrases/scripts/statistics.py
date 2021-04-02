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
            
## add functionality of multiple provenances

            t["title"] = (ans_dict["provenance"][0])["title"]
            t["section"] = (ans_dict["provenance"][0])["section"] 
            t["paragraph"] = (ans_dict["provenance"][0])["start_paragraph_id"]
            t["wiki_id"] = (ans_dict["provenance"][0])["wikipedia_id"]
            ret_list.append(t)
            
    return ret_list
        

with jsonlines.open('nq-dev-kilt.jsonl') as reader:
    for obj in reader:
        obj_list.append(obj)
        temp_str = (obj["input"].lower()).strip()
        gt_dict[temp_str] = getAnswerList(obj)
        
gt_dict["Who got the first Nobel Prize in physics".lower()]


pred_list = []

with jsonlines.open('prediction.jsonl') as reader:
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
                            
            if(count > max_count) :
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
                            
                if(count > max_count) :
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
    
        
                
    
    
        
    
            
            
        
        