import pandas as pd

model_dict = {'50h':[1],'100h':[1],'200h':[2,3],'400h':[4,8],'800h':[9,18]
    ,'1600h':[19,28],'3200h':[29,36],'4500h':[46,54],'7100h':[66,74]}


model_dict = {'50h':[1],'100h':[1],'200h':[2,3],'400h':[4,8],'800h':[10,18]
    ,'1600h':[20,28],'3200h':[29,36],'4500h':[46,54],'7100h':[66,74]}

df = pd.read_csv('/Users/jliu/PycharmProjects/Lexical-benchmark/datasets/raw/CHILDES_child.csv'
                   ,usecols = ['month','content','num_tokens'])

month_lst = ['400h']
for month in month_lst:
    selected = df[(df['month'] >= model_dict[month][0]) & (df['month'] <= model_dict[month][1])]
    selected.columns = ['train','month','num_tokens']
    selected.to_csv('/Users/jliu/PycharmProjects/freq_bias_benchmark/data/train/train_utt/' + month[:-1] + '_child.csv')




df = pd.read_csv('/Users/jliu/PycharmProjects/Lexical-benchmark/datasets/processed/generation/generation.csv')
selected = df[(df['month'] >= 16) & (df['month'] <= 30)]
selected.to_csv('/Users/jliu/PycharmProjects/Lexical-benchmark/datasets/processed/generation/gen_AE.csv')