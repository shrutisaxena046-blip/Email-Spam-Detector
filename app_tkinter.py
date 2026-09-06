# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import tkinter as tk
from tkinter import messagebox
import pickle
import string


# ============================================================
# 2. LOAD TRAINED MODEL AND TF-IDF VECTORIZER
# ============================================================

with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# ============================================================
# 3. FUNCTION TO DETECT SPAM
# ============================================================

def check_spam():

    message = message_box.get("1.0", tk.END).strip()

    if message == "":
        messagebox.showwarning("Warning", "Please enter a message.")
        return

    # Convert message to lowercase
    clean_message = message.lower()

    # Remove punctuation
    clean_message = clean_message.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Convert message into TF-IDF features
    message_tfidf = vectorizer.transform([clean_message])

    # Make prediction
    prediction = model.predict(message_tfidf)[0]

    # Display result
    if prediction == "spam":
        result_label.config(
            text="⚠️ SPAM MESSAGE",
        )
    else:
        result_label.config(
            text="✅ NOT SPAM",
        )


# ============================================================
# 4. CREATE APPLICATION WINDOW
# ============================================================

window = tk.Tk()

window.title("Email Spam Detector")
window.geometry("600x450")


# ============================================================
# 5. APPLICATION TITLE
# ============================================================

title_label = tk.Label(
    window,
    text="📧 Email Spam Detector",
    font=("Arial", 22, "bold")
)

title_label.pack(pady=20)


# ============================================================
# 6. MESSAGE INSTRUCTION
# ============================================================

instruction_label = tk.Label(
    window,
    text="Enter your email or message below:",
    font=("Arial", 13)
)

instruction_label.pack(pady=10)


# ============================================================
# 7. TEXT BOX FOR USER INPUT
# ============================================================

message_box = tk.Text(
    window,
    height=10,
    width=60,
    font=("Arial", 12)
)

message_box.pack(pady=10)


# ============================================================
# 8. CHECK BUTTON
# ============================================================

check_button = tk.Button(
    window,
    text="Check Message",
    command=check_spam,
    font=("Arial", 13, "bold"),
    padx=20,
    pady=8
)

check_button.pack(pady=15)


# ============================================================
# 9. RESULT LABEL
# ============================================================

result_label = tk.Label(
    window,
    text="",
    font=("Arial", 18, "bold")
)

result_label.pack(pady=15)


# ============================================================
# 10. START APPLICATION
# ============================================================

window.mainloop()