import streamlit as st
import openai
import pdfplumber
import csv
import os
import json
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.llms import OpenAI

# إعداد مفتاح API الخاص بـ OpenAI
openai.api_key = 'YOUR_OPENAI_API_KEY'  # استبدل 'YOUR_OPENAI_API_KEY' بمفتاح API الخاص بك

# دوال تحميل الملفات الثابتة (PDF, CSV, TXT, JSON, HTML)
def load_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def load_csv(file_path):
    text = ""
    with open(file_path, newline='', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            text += " ".join(row) + "\n"
    return text

def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def load_json(file_path):
    text = ""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        text = json.dumps(data, ensure_ascii=False, indent=2)
    return text

def load_html(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
    return text

# تحميل محتوى صفحة ويب من رابط HTML
def load_web_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    return text

# تحميل الروابط من ملف links.txt أو links.json وتحليلها كصفحات HTML
def load_links():
    links_file_path = os.path.join("data", "links.json")
    documents = []

    if os.path.exists(links_file_path):
        try:
            with open(links_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                for link in data["links"]:
                    content = load_web_content(link)
                    documents.append(Document(page_content=content, metadata={"source": link}))
        except json.JSONDecodeError:
            st.error("هناك خطأ في تنسيق ملف links.json. تأكد من أن الملف يحتوي على تنسيق JSON صالح.")
    
    return documents

# إعداد LangChain
def setup_chain(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return ConversationalRetrievalChain(OpenAI(), db.as_retriever())

# قراءة جميع الملفات من مجلد data وتحميل الروابط
def load_all_files():
    documents = []
    data_folder = "data"
    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)
        content = None
        if filename.endswith(".pdf"):
            content = load_pdf(file_path)
        elif filename.endswith(".csv"):
            content = load_csv(file_path)
        elif filename.endswith(".txt") and filename != "links.txt":
            content = load_txt(file_path)
        elif filename.endswith(".json") and filename != "links.json":
            content = load_json(file_path)
        elif filename.endswith(".html") or filename.endswith(".htm"):
            content = load_html(file_path)
        
        if content:
            documents.append(Document(page_content=content, metadata={"source": filename}))

    # إضافة محتوى الروابط كصفحات HTML
    documents.extend(load_links())
    return documents

# واجهة Streamlit للمستخدم
st.set_page_config(page_title="Chatbot - Ask Only", page_icon="🤖")
st.title("Chatbot 🤖 - اسألني فقط")
st.write("يمكنك طرح الأسئلة بناءً على المعلومات المتاحة في الملفات الثابتة والروابط.")

# تحميل الملفات والروابط
documents = load_all_files()
chain = setup_chain(documents)

# حقل إدخال السؤال
user_input = st.text_input("اكتب سؤالك هنا:")

# عرض الإجابة
if user_input:
    with st.spinner("جاري البحث عن الإجابة..."):
        response = chain({"question": user_input})
        st.write("الإجابة:", response["answer"])
