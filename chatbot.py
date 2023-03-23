import random
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample conversation data
conversation = [
    "Hi, how are you?",
    "I'm doing well, thank you for asking. How can I assist you today?",
    "Can you recommend a good restaurant in the area?",
    "Sure, what type of cuisine are you in the mood for?",
    "I'm in the mood for Italian food.",
    "There's a great Italian restaurant called Mario's on Main Street. Would you like me to look up the address and phone number for you?",
    "Yes, please. That would be great.",
    "The address for Mario's is 123 Main Street, and their phone number is (555) 555-1234.",
    "Thanks for your help!",
    "You're welcome. Have a great day!"
]

# Tokenization
nltk.download('punkt')
sent_tokens = nltk.sent_tokenize("\n".join(conversation))
word_tokens = nltk.word_tokenize("\n".join(conversation))

# Preprocessing
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Vectorization
vectorizer = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')

# Model Training
tfidf = vectorizer.fit_transform(sent_tokens)

# Chat function
def chatbot_response(user_input):
    bot_response = ''
    sent_tokens.append(user_input)
    tfidf = vectorizer.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf == 0):
        bot_response = bot_response + "I am sorry! I don't understand you."
        return bot_response
    else:
        bot_response = bot_response + sent_tokens[idx]
        return bot_response

# Testing the chatbot
print("Hello, I am your personal chatbot. How can I help you today?")
while True:
    user_input = input()
    if user_input.lower() == 'bye':
        print("Goodbye!")
        break
    else:
        print(chatbot_response(user_input))
