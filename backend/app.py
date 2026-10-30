import streamlit as st
import pandas as pd

from sklearn.tree import DecisionTreeRegressor

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CareerForge AI",
    page_icon="🚀",
    layout="centered"
)

st.title("🚀 CareerForge AI")
st.subheader("Your Internship Readiness Assistant")

st.write(
    "Evaluate your current profile and discover areas "
    "where you can improve your internship readiness."
)

st.info(
    "This is an educational prototype trained on synthetic data. "
    "Its score is not a real-world internship selection probability."
)


# ==========================================
# LOAD THE DATASET
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("students_regression.csv")


df = load_data()


# ==========================================
# TRAIN THE REGRESSION MODEL
# ==========================================

@st.cache_resource
def train_model(data):
    features = data[["CGPA", "DSA_Problems", "Projects"]]

    target = data["Internship_Readiness_Score"]

    model = DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    )

    model.fit(features, target)

    return model


model = train_model(df)


# ==========================================
# USER PROFILE INPUTS
# ==========================================

st.header("📋 Enter Your Profile")

cgpa = st.number_input(
    "Your CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.5,
    step=0.1
)

dsa_problems = st.number_input(
    "Number of DSA Problems Solved",
    min_value=0,
    max_value=1000,
    value=100,
    step=10
)

projects = st.number_input(
    "Number of Projects Completed",
    min_value=0,
    max_value=50,
    value=2,
    step=1
)


# ==========================================
# ANALYZE PROFILE
# ==========================================

if st.button("Analyze My Profile", use_container_width=True):

    student = pd.DataFrame([{
        "CGPA": cgpa,
        "DSA_Problems": dsa_problems,
        "Projects": projects
    }])

    predicted_score = float(model.predict(student)[0])

    # Keep the displayed score within 0–100
    predicted_score = max(0.0, min(100.0, predicted_score))

    st.divider()

    st.header("📊 Your Results")

    st.metric(
        label="Internship Readiness Score",
        value=f"{predicted_score:.1f}/100"
    )

    st.progress(int(round(predicted_score)))

    if predicted_score < 40:
        st.warning(
            "Your profile has several areas to strengthen. "
            "Start with consistent learning and practical work."
        )

    elif predicted_score < 60:
        st.info(
            "You're developing your profile. "
            "Consistent practice and project work can help."
        )

    elif predicted_score < 80:
        st.success(
            "Your profile is progressing well. "
            "Keep strengthening your technical skills."
        )

    else:
        st.success(
            "Your profile scores strongly under this model's "
            "readiness criteria. Keep preparing for interviews."
        )

    # ======================================
    # PERSONALIZED ACTION PLAN
    # ======================================

    st.header("🎯 Your Action Plan")

    suggestions = []

    if cgpa < 7.5:
        suggestions.append(
            "📚 **Academics:** Strengthen your understanding of core subjects "
            "and work toward improving your CGPA."
        )
    else:
        suggestions.append(
            "📚 **Academics:** Maintain your CGPA and revise core CS subjects."
        )

    if dsa_problems < 100:
        suggestions.append(
            "💻 **DSA:** Build consistency with arrays, strings, two pointers, "
            "sliding window, trees, and other core patterns."
        )
    elif dsa_problems < 200:
        suggestions.append(
            "💻 **DSA:** Continue solving problems, revise older questions, "
            "and practise medium-level problems."
        )
    else:
        suggestions.append(
            "💻 **DSA:** Focus on problem-solving quality, timed practice, "
            "and explaining your approach and complexity."
        )

    if projects < 2:
        suggestions.append(
            "🛠️ **Projects:** Build at least one practical project "
            "and learn to explain its architecture and implementation."
        )
    else:
        suggestions.append(
            "🛠️ **Projects:** Improve your existing projects, document "
            "technical decisions, and prepare to demonstrate them."
        )

    suggestions.append(
        "🎤 **Interview preparation:** Practise OOP, DBMS, operating systems, "
        "computer networks, and explaining your projects."
    )

    for suggestion in suggestions:
        st.markdown(f"- {suggestion}")

    # ======================================
    # PROFILE SUMMARY
    # ======================================

    st.header("🧾 Profile Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("CGPA", f"{cgpa:.1f}/10")
    col2.metric("DSA Problems", f"{dsa_problems}")
    col3.metric("Projects", f"{projects}")

    st.caption(
        "The readiness score is an estimate produced by a model trained "
        "on synthetic examples. It does not represent your actual chance "
        "of getting shortlisted by an employer."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "CareerForge AI | Internship Readiness Prototype"
)