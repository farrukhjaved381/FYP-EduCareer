import streamlit as st
import tempfile, os, re, ast
import pandas as pd
import gensim
from difflib import get_close_matches
from streamlit.components.v1 import html
import nltk
from nltk.corpus import stopwords

# ✅ THIS MUST COME FIRST
st.set_page_config(
    page_title="EduCareer Dashboard",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="auto"
)

# ---------------------------------------
# ✨ Custom CSS
st.markdown("""
    <style>
    .title {
        font-size: 36px !important;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.3em;
    }
    .subtitle {
        font-size: 18px;
        color: #34495e;
    }
    .card {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
        box-shadow: 0 0 8px rgba(0,0,0,0.05);
    }
    .section-title {
        font-size: 22px;
        color: #1f618d;
        margin-top: 1em;
    }
    </style>
""", unsafe_allow_html=True)

# Custom modules
from upload_cv import extract_text_from_pdf, extract_text_from_docx
from job_recommendations import recommend_jobs
from course_recommendations import recommend_courses
from career_path_recommendation import recommend_career_path
from skill_gap import analyze_skill_gap_for_career
stop_words = set(stopwords.words('english'))
# =============================
# 📦 LOAD MODELS & DATA
# =============================
@st.cache_resource
def load_skill2vec_model(path='~/fyp/ai-ml/models/skill2vec.model'):
    try:
        return gensim.models.Word2Vec.load(os.path.expanduser(path))
    except Exception as e:
        st.error(f"Skill2Vec load error: {e}")
        return None

@st.cache_data
def load_known_skills(filepath='~/fyp/ai-ml/datasets/skill2vec/cleaned_skill2vec_50K_with_skills.csv'):
    filepath = os.path.expanduser(filepath)
    try:
        df = pd.read_csv(filepath, usecols=['cleaned_extracted_skills'], dtype={'cleaned_extracted_skills': str})
        if 'cleaned_extracted_skills' in df.columns:
            skills = set()
            for item in df['cleaned_extracted_skills'].dropna():
                try:
                    skill_list = ast.literal_eval(item)
                    if isinstance(skill_list, list):
                        for skill in skill_list:
                            if isinstance(skill, str) and skill:  # Check if skill is not empty
                                skills.add(skill.lower().strip())
                    elif isinstance(item, str) and item: # Check if item is not empty
                      skills.add(item.lower().strip())
                except (SyntaxError, ValueError):
                    if isinstance(item, str) and item:
                      skills.add(item.lower().strip())
            if 'DEBUG' in locals() and DEBUG:  # Check if DEBUG is defined and True
                print(f"✅ Loaded {len(skills)} cleaned known skills.")
            return skills
    except Exception as e:
        st.error(f"❌ Failed to load cleaned known skills: {e}")
    return set()

@st.cache_data
def load_csv(filepath):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"CSV not found: {filepath}")
        return None

skill2vec_model = load_skill2vec_model()
known_skills = load_known_skills()
career_skills_df = load_csv("../datasets/career_paths/tech_roles_skills.csv")
jobs_df = load_csv("../datasets/jobs/jobs.csv")
courses_df = load_csv("../datasets/courses/courses.csv")

# =============================
# 🔍 TEXT & SKILL CLEANING
# =============================
def clean_text(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).lower().strip()

def clean_skills_with_skill2vec(text, known_skills, model, similarity_threshold=0.9, embedding_threshold=0.6):
    if not model or not known_skills:
        return []

    text = clean_text(text)
    words = [
        word.lower()
        for word in text.split()
        if len(word) > 2 and not word.isdigit() and word.lower() not in stop_words
    ]
    extracted_skills = set()

    # 1. Try to match whole words directly (case-insensitive)
    for word in words:
        if word in known_skills:
            extracted_skills.add(word)

    # 2. Try fuzzy matching for remaining words
    remaining_words = [word for word in words if word not in extracted_skills]
    for word in remaining_words:
        matches = get_close_matches(word, known_skills, n=1, cutoff=similarity_threshold)
        if matches:
            extracted_skills.add(matches[0])

    # 3. Try to identify potential multi-word skills using n-grams (bi-grams and tri-grams)
    n_grams = []
    for n in range(2, 4):
        for i in range(len(words) - n + 1):
            n_gram = " ".join(words[i:i + n])
            if n_gram in known_skills:
                extracted_skills.add(n_gram)

    # 4. Semantic similarity for remaining single words (use with more caution)
    remaining_single_words = [word for word in words if " " not in word and word not in extracted_skills]
    for word in remaining_single_words:
        cleaned_word = re.sub(r'[^a-z0-9]', '', word)
        if cleaned_word in model.wv:
            try:
                similar_words = model.wv.most_similar(cleaned_word, topn=2) # Even more conservative
                for sim_word, score in similar_words:
                    sim_word_clean = re.sub(r'[^a-z0-9]', '', sim_word)
                    if sim_word_clean in known_skills and score >= embedding_threshold:
                        extracted_skills.add(sim_word_clean)
            except KeyError:
                continue

    # 5. Post-processing: Remove any remaining very short or problematic words
    final_extracted_skills = {skill for skill in extracted_skills if len(skill) > 2 and skill not in stop_words}

    return sorted(list(final_extracted_skills))

# ---------------------------------------
# 🚀 MAIN UI
def main():
    st.markdown("<div class='title'>📘 EduCareer: AI-Powered Career Recommendation System</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Upload your CV/Resume and get personalized job suggestions, course paths, and career insights.</div>", unsafe_allow_html=True)
    st.markdown("---")

    # 📤 File Upload
    uploaded_file = st.file_uploader("📄 Upload your CV or Resume", type=["pdf", "docx"])
    extracted_text = ""

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        if uploaded_file.type == "application/pdf":
            extracted_text = extract_text_from_pdf(tmp_path)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            extracted_text = extract_text_from_docx(tmp_path)
        os.unlink(tmp_path)

    # 🔍 Text Extraction + Skill Processing
    if extracted_text:
        extracted_skills = clean_skills_with_skill2vec(extracted_text, known_skills, skill2vec_model)

        # 🔧 Expanders
        with st.expander("🧠 Extracted KeyWords from CV"):
            st.success(", ".join(extracted_skills) if extracted_skills else "No skills extracted.")

        with st.expander("📜 Raw CV Text"):
            st.code(extracted_text[:2000] + "..." if len(extracted_text) > 2000 else extracted_text, language="text")

        # ============================================
        # 💼 JOB RECOMMENDATIONS
        st.markdown("<div class='section-title'>💼 Top Job Recommendations</div>", unsafe_allow_html=True)
        keyword_jobs = recommend_jobs(extracted_skills, jobs_df)

        if keyword_jobs:
            for idx, job in enumerate(keyword_jobs[:3]):
                with st.container():
                    
                    st.markdown(f"**{idx+1}. {job.get('Job Title')}**")
                    st.write(f"🏢 {job.get('Company', 'N/A')} | 🌍 {job.get('Location', 'N/A')}")
                    st.success(f"✅ Matching Skills: {', '.join(job.get('Matching Skills', []))}")
                    gaps = analyze_skill_gap_for_career(extracted_skills, job.get('Job Title', ''))
                    if gaps:
                        st.warning(f"⚠️ Skill Gaps: {', '.join(gaps)}")
                    if job.get('URL'):
                        st.markdown(f"[🔗 Apply Now]({job['URL']})", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No keyword-based jobs found.")

        # ============================================
        # 📚 COURSE RECOMMENDATIONS
        st.markdown("<div class='section-title'>📚 Recommended Courses</div>", unsafe_allow_html=True)
        courses = recommend_courses(extracted_skills, courses_df)

        if courses:
            for course in courses:
                required_skills = []
                try:
                    required_skills = ast.literal_eval(course.get('Required Skills', '[]'))
                except: pass
                matching = [s for s in extracted_skills if s.lower() in map(str.lower, required_skills)]
                
                st.markdown(f"**🎓 {course.get('Course Name')}** (_{course.get('Platform')}_)")
                st.success(f"Matching Skills: {', '.join(matching)}")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No relevant courses found.")

        # ============================================
        # 🧭 CAREER PATH RECOMMENDATIONS
        st.markdown("<div class='section-title'>🧭 Career Path Suggestions</div>", unsafe_allow_html=True)
        career_paths = recommend_career_path(extracted_skills, career_skills_df)

        if career_paths:
            for idx, path in enumerate(career_paths):
                
                st.markdown(f"**{idx+1}. {path['Career Path']}**")
                st.success(f"✅ Matching Skills: {', '.join(path['Matching Skills'])}")
                st.info(f"➡️ Next Roles: {', '.join(path['Next Roles'])}")
                st.write(f"📊 Relevance Score: `{path['Relevance Score']:.2f}`")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("No relevant career paths found.")

    else:
        st.info("👆 Please upload a CV/Resume to generate personalized results.")

    # ============================================
    # 🔚 Footer
    st.markdown("---")
    st.caption("🚀 Powered by Skill2Vec • Developed by Farrukh & Usman • BSCS FYP 2025 🎓")

# 🔁 Run App
if __name__ == "__main__":
    main()