import json
import os

file_name = os.path.join(os.environ['DPH_DATA_DIR'],'open-qa/nq-open/test_preprocessed')
with open(file_name+'.json') as f:
    data = json.load(f)["data"]
    with open(file_name+'_140-180.json', 'w') as outfile:
        json.dump({"data":data[140*12:180*12]}, outfile)
    with open(file_name+'_180-220.json', 'w') as outfile:
        json.dump({"data":data[180*12:220*12]}, outfile)
    with open(file_name+'_220-260.json', 'w') as outfile:
        json.dump({"data":data[220*12:260*12]}, outfile)
    with open(file_name+'_260-301.json', 'w') as outfile:
        json.dump({"data":data[260*12:]}, outfile)