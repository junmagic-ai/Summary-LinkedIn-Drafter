import streamlit as st
import os
from io import BytesIO
from streamlit_option_menu import option_menu

from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate

from PyPDF2 import PdfReader
import textwrap, ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi

@st.cache_data
def chunk_text(text,char):
    chunks = textwrap.wrap(text, width=char, break_long_words=False)
    return chunks

@st.cache_data
def convert_to_txt (file,file_ext):
    full_text=""
    if file_ext == ".epub":
        try:
            book = epub.read_epub(file)
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    html_content = item.get_content().decode('utf-8')
                    soup = BeautifulSoup(html_content, 'html.parser')  
                    text = soup.get_text()
                    full_text += text
        except Exception as oops:
            st.error("Something wrong with the file - could be corrupt, encrypted, or wrong format. Error: "+str(oops))
    elif file_ext == ".pdf":
        try:
            pdf = PdfReader(file)
            for page in pdf.pages:
                full_text+=page.extract_text()+" "
        except Exception as oops:
            st.error("Something wrong with the file - could be corrupt, encrypted, or wrong format. Error: "+str(oops))
    else:
        full_text=file.read()
    return full_text

def get_transcript(url):
    transcript=""
    go_ahead = False
    if "watch?v=" in url:
        video = YouTube(url)
        go_ahead = True
    elif "youtu.be" in url:
        v_id = url.split("/")[-1]
        video = YouTube("https://youtube.com/watch?v="+v_id)
        go_ahead = True
    else:
        st.write("Invalid YouTube URL")
        transcript = "Error"
    if(go_ahead):
        try:
            srt = YouTubeTranscriptApi.get_transcript(video.video_id)
            for item in srt:
                transcript = transcript+(item['text'])+" "
        except Exception as e:
            st.write("Error loading transcript - it doesn't exist or is not in English. Try another video.")
    
    return transcript

def process_files_input():
    files = st.file_uploader("Upload a file and press Start:", ["pdf","epub","txt"], accept_multiple_files=False)
    st.write("")
    submitted = st.button("Start")

    if files and submitted:
        alltext = ""
        for file in files:
            file_ext = os.path.splitext(file.name)[1]
            file_data = file.getvalue()
            byte_file = BytesIO(file_data)
            text = convert_to_txt(byte_file, file_ext)
            alltext += str(text) + " " if file_ext == ".txt" else text + "\n\n"
        return alltext
    return None

def process_tube_input():
    url = st.text_input("Paste a Youtube link below and press Start:")
    st.write("")
    pressed = st.button("Start")
    return get_transcript(url) if url and pressed else None

def process_text_input():
    text = st.text_area("Paste any text below and press Start:")
    st.write("")
    pasted = st.button("Start")
    return text if pasted and text else None

def summarise(txt):
    # Instantiate the LLM model
    llm=ChatOpenAI(openai_api_base=st.secrets["OPENAI_BASE_URL"],openai_api_key=st.secrets["OPENAI_API_KEY"], temperature=0,model_name="gpt-3.5-turbo-1106", max_tokens=15000)
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=50000, chunk_overlap=1000)
    texts = text_splitter.split_text(txt)

    map_prompt = """

    You are a world best summariser. Concisely summarise key bullet points from the text below. Include relevant examples, omit excess details:
    {text}
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    
    chain = load_summarize_chain(
        llm=llm, 
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=False
    )

    docs = [Document(page_content=t) for t in texts]

    return chain.run(docs)

def create_summary(text):
    summary = ""
    my_bar = st.progress(0)

    chunks = chunk_text(text, 10000)
    
    for i, chunk in enumerate(chunks):
        my_bar.progress((i + 1) / len(chunks))
        try:
            chunk_summary = summarise(chunk)
        except Exception as e:
            st.write("Something went wrong: " + str(e))
            return

        summary += "\n".join(chunk_summary.splitlines()) + "\n\n"

    st.markdown("#### Summary: ")
    st.text_area(label="",value=summary, height=300)
    st.write("")
    st.download_button("Download Summary", data=summary, file_name="Summary.txt")
    st.write("---")
    st.markdown("#### LinkedIn: ")
    st.write(linkedin(summary))

def linkedin(txt):
    prompt_template = "Turn this text into a comprehensive, informative and engaging LinkedIn post. Include relevant emojis and hashtags: {text}"
    prompt = PromptTemplate(
        input_variables=["text"], template=prompt_template
    )
    llm = LLMChain(llm=ChatOpenAI(openai_api_base=st.secrets["OPENAI_BASE_URL"],openai_api_key=st.secrets["OPENAI_API_KEY"], temperature=0,model_name="gpt-3.5-turbo-1106", max_tokens=15000), prompt=prompt)
    return llm.run(txt)

if __name__=="__main__":

    st.set_page_config(page_title="LinkedIn Writer", page_icon=":gear:", layout="wide")
    
    st.header("Summariser & LinkedIn Post Generator")
    summarise_what = option_menu(
        menu_title=None,
        options=["Files","Tube","Text"],
        icons=["file-pdf","youtube","file-text"],
        default_index=0,
        orientation="horizontal"
    )
    st.write("")

    if summarise_what in ["Files", "Tube", "Text"]:
        if summarise_what == "Files":
            input_data = process_files_input()
        elif summarise_what == "Tube":
            input_data = process_tube_input()
        elif summarise_what == "Text":
            input_data = process_text_input()
        if input_data:
            create_summary(input_data)

