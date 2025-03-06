import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from a PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

# Function to rank resumes based on similarity
def rank_resumes(job_description, resumes):
    vectorizer = TfidfVectorizer()
    documents = [job_description] + resumes
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    ranked_resumes = sorted(
        enumerate(similarity_scores), key=lambda x: x[1], reverse=True
    )
    return ranked_resumes, similarity_scores

# Streamlit UI
st.title("Resume Ranking System")

uploaded_files = st.file_uploader("Upload resumes (PDF)", accept_multiple_files=True)
job_desc = st.text_area("Enter Job Description")

if st.button("Rank Resumes"):
    if uploaded_files and job_desc:
        resumes_text = [extract_text_from_pdf(file) for file in uploaded_files]
        rankings, similarity_scores = rank_resumes(job_desc, resumes_text)
        
        st.subheader("Ranked Resumes")
        resume_names = []
        scores = []
        
        for i, (index, score) in enumerate(rankings):
            resume_name = f"Resume {index+1}"
            resume_names.append(resume_name)
            scores.append(score)
            st.write(f"✅ Rank {i+1}: {resume_name} (Score: {score:.2f})")
        
        # Display bar chart for similarity scores
        st.subheader("Resume Similarity Scores")
        scores_df = pd.DataFrame({"Resume": resume_names, "Score": scores})
        st.bar_chart(scores_df.set_index("Resume"))
    else:
        st.warning("Please upload resumes and enter a job description.")
