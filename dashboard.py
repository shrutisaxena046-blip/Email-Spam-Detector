import streamlit as st
import pandas as pd


# DASHBOARD TITLE

st.title(" Spam Detection Dashboard")

st.write(
    "Overview of dataset distribution and machine learning model performance."
)

# KEY PERFORMANCE INDICATORS

total_messages = 5572
spam_messages = 747
ham_messages = 4825

spam_percentage = (
    spam_messages / total_messages
) * 100

ham_percentage = (
    ham_messages / total_messages
) * 100

accuracy = 96.86

# FINAL MODEL
model_name = "Logistic Regression"

# KPI CARDS
col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "📨 Total Messages",
        f"{total_messages:,}"
    )

with col2:

    st.metric(
        " Spam Messages",
        f"{spam_messages:,}"
    )

with col3:

    st.metric(
        " Ham Messages",
        f"{ham_messages:,}"
    )

with col4:

    st.metric(
        " Model Accuracy",
        f"{accuracy}%"
    )


# FINAL MODEL INFORMATION

st.subheader("Final Model")

st.write(
    f"Selected Model: **{model_name}**"
)

st.write(
    "Logistic Regression was selected because it achieved "
    "better accuracy than Multinomial Naive Bayes."
)

# MESSAGE DISTRIBUTION

st.subheader(" Dataset Overview")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Spam Rate",
        f"{spam_percentage:.2f}%"
    )

with col2:

    st.metric(
        "Ham Rate",
        f"{ham_percentage:.2f}%"
    )

# SPAM VS HAM DISTRIBUTION
st.subheader(" Spam vs Ham Distribution")

chart_data = {
    "Category": ["Spam", "Ham"],
    "Messages": [spam_messages, ham_messages]
}

st.bar_chart(
    chart_data,
    x="Category",
    y="Messages"
)


# MODEL COMPARISON

st.subheader(" Model Comparison")

model_comparison = pd.DataFrame({
    "Model": [
        "Multinomial Naive Bayes",
        "Logistic Regression"
    ],
    "Accuracy": [
        95.70,
        96.86
    ]
})

st.bar_chart(
    model_comparison.set_index("Model")
)

# RUNTIME ANALYTICS - CURRENT USER SESSION
st.subheader(" Runtime Analytics")

history = st.session_state.prediction_history

# CHECK IF USER HAS TESTED ANY MESSAGES

if len(history) == 0:

    st.info(
        "No messages have been checked yet. "
        "Go to Spam Detector and test a message."
    )


else:
    # CONVERT HISTORY INTO DATAFRAME

    runtime_data = pd.DataFrame(history)

    # CALCULATE RUNTIME STATISTICS

    total_checked = len(runtime_data)

    spam_count = (
        runtime_data["prediction"] == "spam"
    ).sum()

    ham_count = (
        runtime_data["prediction"] == "ham"
    ).sum()

    average_confidence = (
        runtime_data["confidence"].mean()
    )

    # RUNTIME KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            " Messages Checked",
            total_checked
        )

    with col2:

        st.metric(
            " Spam Detected",
            spam_count
        )

    with col3:

        st.metric(
            "Not Spam",
            ham_count
        )

    with col4:

        st.metric(
            "Avg Confidence",
            f"{average_confidence:.2f}%"
        )

    # RUNTIME DISTRIBUTION

    st.write("###  Your Prediction Distribution")

    runtime_chart_data = pd.DataFrame({
        "Prediction": ["Spam", "Not Spam"],
        "Count": [spam_count, ham_count]
    })

    st.bar_chart(
        runtime_chart_data.set_index("Prediction")
    )

    # PREDICTION HISTORY
    

    st.write("###  Prediction History")

    display_data = runtime_data.copy()

    display_data["confidence"] = display_data[
        "confidence"
    ].round(2)

    st.dataframe(
        display_data,
        use_container_width=True
    )


# PROJECT INFORMATION

st.divider()

st.subheader("ℹ About This Project")

st.write(
    """
    This Python training project uses Natural Language Processing
    and Machine Learning to classify messages as Spam or Not Spam.

    TF-IDF is used for text feature extraction.

    Two machine learning models were evaluated:
    Multinomial Naive Bayes and Logistic Regression.

    Logistic Regression was selected as the final model
    with an accuracy of 96.86%.
    """
)