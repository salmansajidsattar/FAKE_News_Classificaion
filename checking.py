import pandas as pd
import numpy as np
from API import utils as ut

model=ut.Load_model()
print("Model loaded")
path='News_Data/combined_News_output.csv'
df=pd.read_csv(path)
true_label=[]
false_label=[]
dict={0:'Fake',1:'True'}
for i in range(len(df)):
    text=df['text'][i]
    flag=df['label'][i]
    try:
        res=ut.prediction(model,text)
        if ((res['prediction']=='Fake') and (res['prediction']==dict[flag])):
            false_label.append(text)
        elif (res['prediction']==dict[flag]):
            true_label.append(text)
        else:
            pass
    except Exception as e:
        print(e)
pd.DataFrame(true_label).to_csv('true_label.csv')
pd.DataFrame(false_label).to_csv('false_label.csv')
    
