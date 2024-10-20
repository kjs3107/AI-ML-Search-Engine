##integrate code with openAI api
import os
from constance import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain


from langchain.memory import ConversationBufferMemory

#for integrating multiple chain to combine ussing sequential chain in langchain
parent_chain=SequentialChain(chains=[chain,chain2],input_variables=['concepts'],output_vaiables=['algorithm','date'],verbose=True)
import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

#streamlit frameworks
st.title("AI-ML Consepts Serach Engine")
input_text = st.text_input("search the topic u want")

##prompt templates
first_input_prompt = PromptTemplate(
    input_variables=["concepts"],
    template = "tell me about the ai-ml algorithm {concepts}"
)

#memory

concepts_memory = ConversationBufferMemory(input_key='concepts',memory_key='chat_history')
algorithm_memory = ConversationBufferMemory(input_key='algorithm',memory_key='chat_history')
#verbose meeaning= the verbose=True parameter in the LLMChain is used to enable detailed logging of the 
#chain's operations when it runs. When set to True, the verbose mode will print or log additional 
#information, such as the input and output of each step, which can be very helpful for debugging and
# understanding what the chain is doing internally.

##openai llms models
llm = OpenAI(temperature=0.8)
chain=LLMChain(llm=llm, prompt=first_input_prompt,verbose=True,output_key='algorithm' , memory=algorithm_memory )

##prompt templates
second_input_prompt = PromptTemplate(
    input_variables=["algorithm"],
    template = "when was the {algorithm} arrived"
)

chain2=LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='date',)

if input_text:
    st.write(chain.run{'concepts':input_text})

    with st.expander('algorithm_memory'):
        st.info(algorithm_memory.buffer)