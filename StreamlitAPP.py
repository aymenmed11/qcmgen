import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv 
from src.qcmgenerator.utils import read_file,get_table_data
from src.qcmgenerator.logger import logging
import streamlit as st
from langchain_community.callbacks import get_openai_callback
from src.qcmgenerator.QCMgenerator import generate_evaluate_chain

# LOADING THE JSON FILE
with open('C:/Users/DELL/qcmgen/Response.json', 'r') as file:
    Response_json=json.load(file)

# creating a title for the app
st.title("QCMs Creator Application with LangChain 🦜️🔗")

# create the form using st.form
with st.form("user_inputs"):
    # file upload
    uploaded_file=st.file_uploader("Upload a PDF or TXT file")

    # Input fields
    qcm_count=st.number_input("No. of QCMs", min_value=3, max_value=50)

    # subject
    subject=st.text_input("Insert Subject", max_chars=22)

    # quiz tone
    tone=st.text_input("Complexity Level Of Questions", max_chars=20, placeholder="Simple")

    # Add button
    button=st.form_submit_button("Create QCMs")

    # check if the button is clicked and all fields have input
    if button and uploaded_file is not None and qcm_count and subject and tone:
        with st.spinner("loading..."):
            try:
                text=read_file(uploaded_file)
                # Count Tokens and the cost of API call
                with get_openai_callback() as cb:
                    response=generate_evaluate_chain(
                        {
                            "text":text,
                            "number":qcm_count,
                            "subject":subject,
                            "tone":tone,
                            "response_json":json.dumps(Response_json)
                                                                        }
                    )

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error")

            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")
                if isinstance(response,dict):
                    # extract the quiz data from the response
                    quiz=response.get("quiz", None)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index=df.index+1
                            st.table(df)
                            # Display the review in a text box as well
                            st.text_area(label="Review", value=response["review"])
                        else:
                            st.error("Error in the table data")
                else:
                    st.write(response)


