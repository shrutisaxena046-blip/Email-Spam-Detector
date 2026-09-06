import streamlit as st
import pickle
import string

st.set_page_config(
    page_title="Spam Detector",
    layout="wide"
)

# RUNTIME ANALYTICS - SESSION DATA

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# LOAD TRAINED MODEL AND TF-IDF VECTORIZER

with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# HEADER

st.title("📧 Email Spam Detector")

st.write(
    "Use Machine Learning to determine whether a message "
    "is **Spam** or **Not Spam**."
)

st.info(
    "💡 Enter a message below and click **Check Message** "
    "to analyze it."
)

# MESSAGE INPUT
message = st.text_area(
    "📝 Enter your message",
    height=180,
    placeholder="Example: Congratulations! You have won a prize..."
)

# BUTTONS

col1, col2 = st.columns([1, 1])

with col1:
    check_button = st.button(
        " Check Message",
        use_container_width=True
    )

with col2:
    clear_button = st.button(
        "Clear",
        use_container_width=True
    )
# CLEAR MESSAGE
if clear_button:
    st.rerun()

# CHECK MESSAGE
if check_button:

    
    # CHECK FOR EMPTY MESSAGE
    if message.strip() == "":
        st.warning("Please enter a message first.")

    else:

    
        # TEXT PREPROCESSING
    

        clean_message = message.lower()

        clean_message = clean_message.translate(
            str.maketrans("", "", string.punctuation)
        )


        # TF-IDF TRANSFORMATION
        message_tfidf = vectorizer.transform(
            [clean_message]
        )

        # PREDICTION
        prediction = model.predict(
            message_tfidf
        )[0]
        # PREDICTION PROBABILITIES
    
        probabilities = model.predict_proba(
            message_tfidf
        )[0]

        confidence = max(probabilities) * 100

        # GET CLASS PROBABILITIES
        class_probabilities = dict(
            zip(
                model.classes_,
                probabilities
            )
        )
        spam_probability = (
            class_probabilities.get("spam", 0) * 100
        )

        ham_probability = (
            class_probabilities.get("ham", 0) * 100
        )
        # SAVE RUNTIME DATA
        
        st.session_state.prediction_history.append({
            "prediction": prediction,
            "confidence": confidence
        })

        # RESULT SECTION
        st.divider()
        st.subheader("Prediction Analysis")

        # PROBABILITY CARDS
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                " Not Spam Probability",
                f"{ham_probability:.2f}%"
            )

        with col2:
            st.metric(
                "Spam Probability",
                f"{spam_probability:.2f}%"
            )
        # PROBABILITY GRAPH
        st.write("###  Probability Distribution")
        st.progress(
            int(ham_probability),
            text=f" Not Spam: {ham_probability:.2f}%"
        )
        st.progress(
            int(spam_probability),
            text=f" Spam: {spam_probability:.2f}%"
        )
        # FINAL RESULT
        st.write("###  Final Result")
        if prediction == "spam":
            st.error(" SPAM MESSAGE")
        else:
            st.success(" NOT SPAM")
        # CONFIDENCE
        st.metric(
            " Model Confidence",
            f"{confidence:.2f}%"
        )
        # INFORMATION
        st.caption(
            "The confidence value represents the probability "
            "assigned by the trained machine learning model."
        )