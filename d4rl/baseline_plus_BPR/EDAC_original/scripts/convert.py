
import pandas as pd
import numpy as np

log_dir = '../results/SAC_10_plr0.0003_qlr0.0003_eta5.00/halfcheetah-medium-expert-v2_0/'
fileout = 'progress.txt'
doc = pd.read_csv(log_dir+'/'+'offline_progress.csv')

items = list(doc['evaluation/Average Returns'])

with open(log_dir+'/'+fileout, 'w') as f:
        f.write('Epoch\tAverageEpRet\tAverageTextEpRet\tTotalEnvInteracts\n')
        for i in range(len(items)):
            f.write(str(i)+'\t'+str(items[i])+'\t'+str(items[i])+'\t'+str(i*1000)+'\n')
dic = {}
dic['exp_name'] = 'EDAC'
#print(dic['exp_name'])
import json
with open(log_dir+'/'+'config.json', 'w') as f:
    json.dump(dic, f)