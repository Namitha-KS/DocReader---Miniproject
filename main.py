import streamlit as st
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Model and tokenizer loading
checkpoint = "LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map='cpu',
    torch_dtype=torch.float32
)

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    return qa

def process_answer(instruction):
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer, generated_text

def main():
    st.title("DocReader")
    with st.expander("About the App"):
        st.markdown(
            """
            This tool is a document reader that uses a language model to answer questions based on the content of the document. Built as a part of S6 Miniproject by Manjima, Malavika, Namitha, Rebekah.
            """
        )

    if st.button("Summary"):
        with st.spinner('Generating Summary...'):
            qa = qa_llm()
            summary_texts = []
            db = Chroma(persist_directory="db")
            documents = db.get_all_documents()

            for doc in documents:
                generated_text = qa(doc['text'])
                summary_texts.append(generated_text['result'])

            summary_text = "\n".join(summary_texts)
            st.text_area("SUMMARY OF SS MODULE- 1.pdf", summary_text, height=500)

    question = st.text_area("Enter your Question")
    if st.button("Ask"):
        st.info("Your Question: " + question)

        st.info("Your Answer")
        answer, metadata = process_answer(question)
        st.write(answer)
        st.write(metadata)

if __name__ == "__main__":
    main()
