import os
import string

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Emotion Detector", page_icon="🎭", layout="centered")

DATA_PATH = "train.txt"

EMOJI = {
    "joy": "😄", "sadness": "😢", "anger": "😠",
    "fear": "😨", "love": "❤️", "surprise": "😲",
}

EXAMPLES = {
    "Joy": "i feel so happy and grateful for everything today",
    "Sadness": "i feel lonely and miserable since she left",
    "Anger": "i am furious that they ignored my complaint again",
    "Fear": "i am terrified of walking home alone tonight",
    "Love": "i feel so affectionate and caring towards my family",
    "Surprise": "i was amazed and shocked by the unexpected news",
}


# ---------- Text preprocessing (same steps as the notebook) ----------
@st.cache_resource
def get_stopwords():
    try:
        import nltk
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            return set(stopwords.words("english"))
    except Exception:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        return set(ENGLISH_STOP_WORDS)


STOP_WORDS = get_stopwords()


def clean_text(txt: str) -> str:
    txt = txt.lower()
    txt = txt.translate(str.maketrans("", "", string.punctuation))
    txt = "".join(c for c in txt if not c.isdigit())
    txt = "".join(c for c in txt if c.isascii())
    return " ".join(w for w in txt.split() if w not in STOP_WORDS)


# ---------- Training (cached so it only runs once) ----------
@st.cache_resource(show_spinner="Training models… (first run only)")
def train_models(data: pd.DataFrame):
    df = data.copy()
    df["text"] = df["text"].astype(str).apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["emotion"], test_size=0.20, random_state=42
    )

    bow = CountVectorizer()
    Xtr_bow, Xte_bow = bow.fit_transform(X_train), bow.transform(X_test)
    tfidf = TfidfVectorizer()
    Xtr_tf, Xte_tf = tfidf.fit_transform(X_train), tfidf.transform(X_test)

    nb_bow = MultinomialNB().fit(Xtr_bow, y_train)
    nb_tfidf = MultinomialNB().fit(Xtr_tf, y_train)
    logreg = LogisticRegression(max_iter=1000).fit(Xtr_tf, y_train)

    return {
        "Naive Bayes (Bag of Words)": (bow, nb_bow, accuracy_score(y_test, nb_bow.predict(Xte_bow))),
        "Naive Bayes (TF-IDF)": (tfidf, nb_tfidf, accuracy_score(y_test, nb_tfidf.predict(Xte_tf))),
        "Logistic Regression (TF-IDF)": (tfidf, logreg, accuracy_score(y_test, logreg.predict(Xte_tf))),
    }, len(df), df["emotion"].value_counts()


def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, sep=";", header=None, names=["text", "emotion"])
    st.warning("`train.txt` not found next to app.py. Upload it to continue.")
    up = st.file_uploader("Upload train.txt (format: text;emotion)", type=["txt", "csv"])
    if up is None:
        st.stop()
    return pd.read_csv(up, sep=";", header=None, names=["text", "emotion"])


# ---------- UI ----------
st.title("🎭 Emotion Detector")
st.caption("Type a sentence and the model predicts the emotion behind it.")

data = load_data()
models, n_rows, class_counts = train_models(data)

with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox(
        "Model", list(models.keys()), index=2
    )
    st.metric("Test accuracy", f"{models[model_name][2]*100:.2f}%")
    st.caption(f"Trained on {n_rows:,} samples (80/20 split).")

tab_predict, tab_models = st.tabs(["🔮 Predict", "📊 Model comparison"])

with tab_predict:
    st.write("**Try an example:**")
    cols = st.columns(3)
    for i, (label, text) in enumerate(EXAMPLES.items()):
        if cols[i % 3].button(label, width="stretch"):
            st.session_state["input_text"] = text

    user_text = st.text_area(
        "Your text", key="input_text", height=120,
        placeholder="e.g. i feel really excited about tomorrow",
    )

    if st.button("Detect emotion", type="primary"):
        cleaned = clean_text(user_text or "")
        if not cleaned:
            st.error("Please enter some text (stop-words only won't work).")
        else:
            vec, clf, _ = models[model_name]
            probs = clf.predict_proba(vec.transform([cleaned]))[0]
            result = pd.Series(probs, index=clf.classes_).sort_values(ascending=False)
            top = result.index[0]

            st.success(f"{EMOJI.get(top, '')}  **{top.capitalize()}**  ({result.iloc[0]*100:.1f}% confidence)")
            st.bar_chart(result.rename("probability"))
            with st.expander("Cleaned text used by the model"):
                st.code(cleaned)

with tab_models:
    comp = pd.DataFrame(
        {"Model": list(models.keys()),
         "Accuracy (%)": [round(v[2] * 100, 2) for v in models.values()]}
    ).set_index("Model")
    st.bar_chart(comp)
    st.dataframe(comp, width="stretch")
    st.subheader("Emotion distribution in training data")
    st.bar_chart(class_counts)
