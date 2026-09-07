# 1. IMPORT REQUIRED LIBRARIES
import pickle
# 2. LOAD THE SAVED MODEL AND TF-IDF VECTORIZER
with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)
# 3. GET MESSAGE FROM USER
message = input("\nEnter your message: ")
# 4. CLEAN THE MESSAGE
message = message.lower()

message = message.translate(
    str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
)
# 5. CONVERT MESSAGE INTO TF-IDF FEATURES
message_tfidf = vectorizer.transform([message])

# 6. PREDICT SPAM OR HAM

prediction = model.predict(message_tfidf)

# 7. DISPLAY RESULT
print("\nPrediction:", prediction[0])

if prediction[0] == "spam":
    print("This message is SPAM!")
else:
    print("This message is NOT SPAM.")