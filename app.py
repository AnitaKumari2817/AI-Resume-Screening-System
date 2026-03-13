import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import joblib
import pdfplumber
import time

# Page Config
st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

#CSS
st.markdown("""
<style>

/* Remove Streamlit Header */
header {visibility: hidden;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

.block-container {
padding-top: 0rem;
}

/* Background */
.stApp{
background: linear-gradient(135deg,#f5f7fa,#e4ecfb);
font-family: 'Segoe UI', sans-serif;
color:black;
}

/* Main Card */
.main-card{
background: rgba(255,255,255,0.9);
padding:40px;
border-radius:20px;
text-align:center;
box-shadow:0 10px 30px rgba(0,0,0,0.15);
animation: fadeIn 1s ease;
}

/* Title Animation */
.title{
font-size:38px;
font-weight:bold;
color:black;
animation: float 4s ease-in-out infinite;
}

/* Subtitle */
.subtitle{
font-size:16px;
color:black;
margin-bottom:25px;
}

/* Upload Box */
[data-testid="stFileUploader"] section{
border:2px dashed #7b8cff;
border-radius:15px;
padding:30px;
background:#f8faff;
transition:0.3s;
}

[data-testid="stFileUploader"] section:hover{
transform:scale(1.03);
box-shadow:0 10px 20px rgba(0,0,0,0.15);
}

/* Upload Text */
[data-testid="stFileUploaderDropzone"]{
color:black !important;
font-weight:500;
}

[data-testid="stFileUploaderDropzoneInstructions"]{
color:black !important;
}

[data-testid="stFileUploaderFileName"]{
color:black !important;
}

/* Animations */

@keyframes fadeIn{
0%{opacity:0; transform:translateY(20px);}
100%{opacity:1; transform:translateY(0);}
}

@keyframes float{
0%{transform:translateY(0px);}
50%{transform:translateY(-8px);}
100%{transform:translateY(0px);}
}

</style>
""", unsafe_allow_html=True)


model = joblib.load("model.pkl")

#UI
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown('<div class="title">🤖 AI Resume Screening System</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle">Upload your resume and let AI evaluate your profile instantly</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# PDF Extraction
def extract_text_from_pdf(file):

    text = ""

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    return text.lower()


# Prediction
if uploaded_file is not None:

    text = extract_text_from_pdf(uploaded_file)

    with st.spinner("🤖 AI is scanning your resume..."):

        time.sleep(2)

        years_experience = min(text.count("experience") * 2, 15)

        skills = [
            "python",
            "machine learning",
            "data science",
            "ai",
            "deep learning",
            "flask",
            "sql",
            "nlp"
        ]

        detected_skills = [skill for skill in skills if skill in text]

        skills_match_score = min(sum(text.count(skill) for skill in skills) * 15, 100)

        if "phd" in text:
            education_level = 3
        elif "master" in text:
            education_level = 2
        else:
            education_level = 1

        project_count = min(text.count("project"), 10)

        resume_length = min(len(text), 900)

        github_activity = min(text.count("github") * 80, 800)

        features = [[
            years_experience,
            skills_match_score,
            education_level,
            project_count,
            resume_length,
            github_activity
        ]]

        prediction = model.predict(features)


    st.markdown("---")

    if prediction[0] == "Yes" or prediction[0] == 1:
        st.success("✅ Candidate Shortlisted")
    else:
        st.error("❌ Candidate Not Shortlisted")


    score = (
        (years_experience/15)*25 +
        (skills_match_score/100)*35 +
        (project_count/10)*15 +
        (github_activity/800)*15 +
        (education_level/3)*10
    )

    score = round(score,2)

    st.write("### 📊 Resume Match Score")

    st.progress(int(score))

    st.write(f"**{score}% Compatibility**")

    #Detected Skills
    st.write("### 🧠 Detected Skills")

    if detected_skills:
        st.success(", ".join(detected_skills))
    else:
        st.warning("No major skills detected")

    # Resume Stats
    st.write("### 📈 Resume Insights")

    st.write(f"**Years Experience Score:** {years_experience}/15")
    st.write(f"**Projects Detected:** {project_count}")
    st.write(f"**Education Level:** {education_level}")

st.markdown("</div>", unsafe_allow_html=True)