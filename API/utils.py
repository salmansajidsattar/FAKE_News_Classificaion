# import streamlit as st
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from tensorflow.keras.preprocessing.text import one_hot
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import math


def Load_model():
    model = load_model("../final_model.h5")
    model.summary()  # included to make it visible when model is reloaded
    return model

def review_cleaning(text,model):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    # text = re.sub('[s%]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def prediction(model,sentence):
    if sentence == "":
        return "Please enter a news article"
    else:
        corpus=[]
        stop_words = set(stopwords.words("english"))
        top_words = set(stopwords.words("english"))
        ps = PorterStemmer()
        news = re.sub('[^a-zA-Z]', ' ',sentence)
        news= news.lower()
        news = news.split()
        news = [ps.stem(word) for word in news if not word in stop_words]
        news = ' '.join(news)
        corpus.append(news)
        voc_size=10000
        #One hot encoding 
        onehot_repr=[one_hot(words,voc_size)for words in corpus] 
        sent_length=5000
        #Padding the sentences
        embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
        Final=np.array(embedded_docs)
        # print(Final.shape)
        pass_data=np.array(Final)
        pass_data = pass_data.reshape(-1, 5000)
        # print(pass_data.shape)
        y_pred=model.predict(pass_data)
        print("y_pred",y_pred)
        prediction_class = np.where(y_pred >= 0.5, 1, 0)
        # predicted_classes = np.argmax(y_pred, axis=1)
        # print("predicted_classes", predicted_classes)
        # print("test",prediction_class)
        # print(y_pred[0][0])
        print("test",prediction_class)
        probability=float(round(y_pred[0][0], 3))
        result=None
        if (prediction_class == 0):
                print("Enter in the flag 0")
                result={
                    "flag":0,
                    "prediction": "Fake",
                    "probability": 1-probability
                }
                print("result", result)
                return result
        if prediction_class == 1:
                print("Enter in the flag 1")
                result={
                    "flag":1,
                    "prediction": "True",
                    "probability": probability
                }
                print("result", result)
                return  result
    
    
# if __name__ == '__main__':
    # st.title('Fake News Classification app ')
    # st.write("A simple fake news classification app utilising 6 traditional ML classifiers and a LSTM model")
    # st.info("LSTM model & tokeniser models loaded ")
    # st.subheader("Input the News content below")
    # sentence = st.text_area("Enter your news content here", "Some news",height=200)
    # predict_btt = st.button("predict")
    # model= Load_model()
    # stop_words = set(stopwords.words("english"))

    # if predict_btt:
    #     corpus = []
    #     # K.set_session(session)
    #     top_words = set(stopwords.words("english"))
    #     ps = PorterStemmer()
    #     news = re.sub('[^a-zA-Z]', ' ',sentence)
    #     news= news.lower()
    #     news = news.split()
    #     news = [ps.stem(word) for word in news if not word in stop_words]
    #     news = ' '.join(news)
    #     corpus.append(news)
    #     voc_size=10000
    #     #One hot encoding 
    #     onehot_repr=[one_hot(words,voc_size)for words in corpus] 
    #     sent_length=5000
    #     #Padding the sentences
    #     embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
    #     Final=np.array(embedded_docs)
    #     print(Final.shape)
    #     import numpy as np
    #     pass_data=np.array(Final)
    #     pass_data = pass_data.reshape(-1, 5000)
    #     print(pass_data.shape)
    #     y_pred=model.predict(pass_data)
    #     prediction_class = np.where(y_pred > 0.5, 1, 0)
    #     print(prediction_class)
    #     print(y_pred[0][0])
    #     st.header("Prediction using LSTM model")

    #     if prediction_class == 0:
    #         st.success('This is a fake news')
    #         prediction_T=float(round(y_pred[0][0],3))
    #         prediction_F=1-prediction_T
    #     if prediction_class == 1:
    #         st.warning('This is not a fake news')
    #         prediction_F=round(y_pred[0][0],3)
    #         prediction_T=1-prediction_F
    #     print("================================================")
    #     print(prediction_F, prediction_T)
    #     class_label = ["FAKE","TRUE"]
    #     # prob_list = [ prediction_F,prediction[0][0]*100]
    #     prob_list = [ prediction_F, prediction_T]
    #     prob_dict = {"true/fake":class_label,"Probability":prob_list}
    #     df_prob = pd.DataFrame(prob_dict)
    #     fig = px.bar(df_prob, x='true/fake', y='Probability')
    #     model_option = "LSTM"
    #     if prediction_F < 0.5:
    #         fig.update_layout(title_text="{} model - prediction probability comparison between true and fake".format(model_option))
    #         st.info("The {} model predicts that there is a higher {} probability that the news content is fake compared to a {} probability of being true".format(model_option,round(prediction_F,3),round(prediction_T,3)))
    #     else:
    #         fig.update_layout(title_text="{} model - prediction probability comparison between true and fake".format(model_option))
    #         st.info("The {} model predicts that there is a higher {} probability that the news content is true compared to a {} probability of being fake".format(model_option,round(prediction_F,3), round(prediction_T,3)))
    #     st.plotly_chart(fig, use_container_width=True)