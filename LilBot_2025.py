"""
Learning Reflection:

This lab helped me understand how chatbots work using natural language processing. I learned how to break 
down text into smaller parts, find out what words are important, and compare them to figure out the best 
response. It was fun building my own chatbot that can talk about algorithms, something I'm interested in 
and find a little challenging. I also got to change how the bot says hi and responds to people, which made 
it feel more personal and fun to use.

"""

#Meet Lil Bot: Your new friend! and Algorithms Assistant 🤖

import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.stem import WordNetLemmatizer

#Reading in the corpus
with open('chatbot_2025.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()

#TOkenisation
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
#print(sent_tokens)#use for testing
word_tokens = nltk.word_tokenize(raw)# converts to list of words
#print(word_tokens)#use for testing

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#Greeting Keyword Matching
GREETING_INPUTS = ["hello", "hi", "greetings", "sup", "what's up", "hey", "yo", "good morning", "good afternoon", "howdy"] 
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I'm glad you're here!", "Howdy!", "Hey hey!", "Welcome!", "Bonjour", "Sup!"] 

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

#Generating response
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english', token_pattern=None) 
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response = robo_response+"I'm not sure I understand. Try asking about an algorithm!"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

flag=True
print("Lil Bot: I'm Lil Bot — your algorithm buddy! Ask me about sorting, recursion, graphs, and more. Type 'bye' to exit.")
while(flag==True):
    user_response = input("You: ")
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("Lil Bot: You're welcome. Keep coding! 💻")
        else:
            if(greeting(user_response)!=None):
                print("Lil Bot: "+greeting(user_response))
            else:
                print("Lil Bot: ",end="") 
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("Lil Bot: Peace out! ✌️ Keep learning algorithms!") 
