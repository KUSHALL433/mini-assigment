from io import StringIO
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

st.set_page_config(
    page_title="Call Transcript Analysis",
    page_icon="☎️",
)

model = ChatGroq(model="llama-3.1-8b-instant")

# Previous prompts didn't worked well 

# prompt = PromptTemplate(template="You are a bot that performs sentiment analysis on a transcript conversation with {text} and classify the entire conversation in  ONLY one word as positive,negative or neutral based on the sentiments it has. Return Ouput as ONLY ONE WORD",input_variables=['text'])

# prompt1 = PromptTemplate(template="You are given {text} as an input summarize the entire conversation in 2-3 lines ONLY. Return output ONLY 2-3 lines")

# NEW PROMPTS
prompt = PromptTemplate(template="""You are a bot that performs sentiment analysis on a transcript conversation.
                        If the given text:
                        -------------
                        {text} 
                        -------------
                        has transcript converation then classify the entire conversation in ONLY one word as positive,negative or neutral based on the sentiments. Return as output only ONE WORD. 
                        If the context is not related to call transcript then output provide as ouput I DON'T KNOW.""",
                        input_variables=['text']
                        )

prompt1 = PromptTemplate(template="""You are a bot that performs sentiment analysis on a transcript conversation.
                        And you are given below text as input:
                        ------------
                        {text} 
                        ------------
                        If the given text is related to a call transcript then summarize the entire conversation in 2-3 lines ONLY.
                         
                        Return as output ONLY 2-3 lines be concise and return no other explanation.
                         
                        If the text is not related to a call transcript then output only I DON'T KNOW.""",input_variables=['text'])

parser = StrOutputParser()

st.title("Call transcript summary generator and sentiment analyzer.",anchor=False)

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





