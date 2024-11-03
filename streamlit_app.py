import os
import json
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.llms import OpenAI

# تعيين مفتاح OpenAI API كمتحول بيئي
os.environ["OPENAI_API_KEY"] = "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"  # استبدل 'YOUR_OPENAI_API_KEY' بالمفتاح الفعلي

# تحميل محتوى صفحة ويب من رابط HTML
def load_web_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    return text

# تحميل الروابط من ملف links.json وتحليلها كصفحات HTML
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

# إعداد LangChain مع معالجة الأخطاء
def setup_chain(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    
    # معالجة أخطاء تهيئة OpenAIEmbeddings
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        st.error(f"حدث خطأ أثناء تهيئة OpenAIEmbeddings: {e}")
        return None
    
    db = FAISS.from_documents(docs, embeddings)
    return ConversationalRetrievalChain(OpenAI(), db.as_retriever())

# واجهة Streamlit للمستخدم
st.set_page_config(page_title="Chatbot - Ask Only", page_icon="🤖")
st.title("Chatbot 🤖 - اسألني فقط من الروابط")
st.write("يمكنك طرح الأسئلة بناءً على المعلومات المتاحة في الروابط الموجودة في ملف links.json.")

# تحميل الروابط
documents = load_links()
chain = setup_chain(documents)

# حقل إدخال السؤال
user_input = st.text_input("اكتب سؤالك هنا:")

# عرض الإجابة
if user_input:
    with st.spinner("جاري البحث عن الإجابة..."):
        response = chain({"question": user_input})
        st.write("الإجابة:", response["answer"])
