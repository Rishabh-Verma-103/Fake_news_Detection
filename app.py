import streamlit as st

import pandas as pd
import numpy as np
import altair as alt
import re
import string

import joblib

pipe_gbc = joblib.load(open("model/model.pkl", "rb"))
vectorizer = joblib.load(open("model/vectorizer.pkl","rb"))

fake_news_emoji_dict = {0: " ❎Fake News", 1: " ✅Real News"}

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorizer.transform(new_x_test)
    return new_xv_test

def predict_news(docx):
    docx = manual_testing(docx)
    results = pipe_gbc.predict(docx)
    return results[0]


def get_prediction_proba(docx):
    docx = manual_testing(docx)
    results = pipe_gbc.predict_proba(docx)
    return results


def main():
    st.title("Fake News Detection")
    st.subheader("Detecting Validity of the News")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_news(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = fake_news_emoji_dict[prediction]
            st.write("{}:{}".format(prediction, emoji_icon))
            st.write("Confidence:{}".format(np.max(probability)))

        with col2:
            st.success("Prediction Probability")
            st.write(probability)
            proba_df = pd.DataFrame(probability, columns=pipe_gbc.classes_)
            # st.write(proba_df.T)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)



if __name__ == '__main__':
    main()