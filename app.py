from io import StringIO
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

prompt = PromptTemplate(template="You are a bot that performs sentiment analysis on a transcript conversation with {text} and classify the entire conversation in  ONLY one word as positive,negative or neutral based on the sentiments it has. Return Ouput as ONLY ONE WORD",input_variables=['text'])

prompt1 = PromptTemplate(template="You are given {text} as an input summarize the entire conversation in 2-3 lines ONLY. Return output ONLY 2-3 lines")

parser = StrOutputParser()

st.title("Internshala Internship Mini Assignment:Customer call transcript sentiment analyzer using Groq.")

transcript = st.file_uploader("Upload a text file containing transcript",type='txt')

if st.button("Submit"):
    if transcript != None:
        stringio = StringIO(transcript.getvalue().decode("utf-8"))
        fi=str(stringio.getvalue())
        chain = prompt | model | parser
        chain_2 = prompt1 | model | parser
        sentiment = chain.invoke({'text':fi})
        summary = chain_2.invoke({'text':fi})
        st.write("Original Transcript\n")
        st.write(fi)
        st.write("Summary:")
        st.write(summary)
        st.write("Sentiment:\n")
        st.write(sentiment)
        data = {'Transcript':fi,'Summary':summary,'Sentiment':sentiment}
        df=pd.DataFrame(data,index=[0])
        df.to_csv('call_analysis.csv',index=False)
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
          label="📥 Download Analysis as CSV",
          data=csv,
          file_name="call_analysis.csv",
          mime="text/csv",
          )
    else:
        st.error("Please Upload a transcript file as input")


