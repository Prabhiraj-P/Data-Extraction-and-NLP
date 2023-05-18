#get_ipython().system('pip install pandas')
#get_ipython().system('pip install requests')
#get_ipython().system('pip install bs4')
#get_ipython().system('pip install csv')
#get_ipython().system('pip install re')




#importing libraries
import pandas as pd
import requests
import numpy as np
from bs4 import BeautifulSoup
import csv
import re



import nltk
nltk.download('cmudict')
nltk.download('punkt')
nltk.download('vader_lexicon')
positive_fname='D:/blackcoffer/positive-words.txt' #file path of positive word list txt
negative_fname='D:/blackcoffer/negative-words.txt'  #file path of negetive word list txt
stopword1_fname='D:/blackcoffer/StopWords_Generic.txt' #file path of stopword word list txt
input_fname='D:/blackcoffer/Input.xlsx - Sheet1.csv'   #input file path .csv file
stopword2_fname='D:/blackcoffer/StopWords_GenericLong.txt' #file path of stopword word list txt



#list  of pronouns used taken from internet
pronouns = ['i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'we', 'us', 'our', 'ours', 'they', 'them', 'their', 'theirs']



from os import remove
positive=[]
 # Declaring positive words list
negative=[] 
# Declaring negative words list
stop_word=[]
 # Declaring stopwords words list

with open(positive_fname,mode='r') as file:
  positive_words=file.readlines() #opening txt file

for word in positive_words:            #adding files to list
  positive.append(word[:len(word)-1])

with open(negative_fname,mode='r',encoding='ISO-8859-1') as file:
  negative_words=file.readlines()

for word in negative_words:
  negative.append(word[:len(word)-1])     #adding files to list
#opening stop words 
with open(stopword1_fname,mode='r',encoding='ISO-8859-1') as file1,open(stopword2_fname,mode='r',encoding='ISO-8859-1')as file2:
  stopword_words=file1.readlines()
  stopword_words2=file2.readlines()
for word in stopword_words+stopword_words2:
  stop_word.append(word[:len(word)-1])     #Creating stopword list




input_df=pd.read_csv(input_fname) #reading csv Dataframe input_df

#list to remove unwanted lines from article. repeating sendences
waste=["Output exceeds the size limit. Open the full output data in a text editor1",'AutoGPT Setup','\\xa0',"Introduction","Contact us: hello@blackcoffer.com","© All Right Reserved, Blackcoffer(OPC) Pvt. Ltd","Ranking customer behaviours for business strategy","Algorithmic trading for multiple commodities markets, like Forex, Metals, Energy, etc.","Trading Bot for FOREX","Python model for the analysis of sector-specific stock ETFs for investment purposes","Playstore & Appstore to Google Analytics (GA) or Firebase to Google Data Studio Mobile App KPI Dashboard","Google Local Service Ads LSA API To Google BigQuery to Google Data Studio","AI Conversational Bot using RASA","Recommendation System Architecture","Rise of telemedicine and its Impact on Livelihood by 2040","Rise of e-health and its impact on humans by the year 2030","Rise of e-health and its impact on humans by the year 2030","Rise of telemedicine and its Impact on Livelihood by 2040","AI/ML and Predictive Modeling","Solution for Contact Centre Problems","How to Setup Custom Domain for Google App Engine Application?","Code Review Checklist"]


# #Code to web scrap from given links



from pandas.io.formats.format import format_array
i=0
main_text=[]  # declating main list to add up all article
for url in input_df['URL']: # to loop through all links in input
 i+=1                          # To know which linkis currently scrapping
 text=""                    # declaring text to nill
 response=requests.get(url)
 soup=BeautifulSoup(response.content,'html.parser')
 main_div=soup.find_all('div',{ 'class':"tdb-block-inner td-fix-index"})
 text2=soup.find_all('p')  #to find all p tag
 print("{}/{} of file completed".format(i,len(input_df))) 
 for div in text2:
   if div.text not in waste:
    div_text=re.sub('\“',' ',div.text)
    text=text+" "+div_text     #add all lines of  article to one into one string
 main_text.append(text)       # Append text to main_rext


import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import cmudict, stopwords




sia.lexicon.update(positive=positive) #positive and negetive words given txt
sia.lexicon.update(negative=negative)





#assigning everything with zero
result={
    'URL_ID':[],
    'URL':[],
'POSITIVE_SCORE':[], #
'NEGATIVE_SCORE':[], #
'POLARITY_SCORE':[], #
'SUBJECTIVITY_SCORE':[], #
'AVG_SENTENCE_LENGTH':[],
'PERCENTAGE_OF_COMPLEX_WORDS':[], #
'FOG_INDEX':[],
'AVG_NUMBER_OF_WORDS_PER_SENTENCE':[], #
'COMPLEX_WORD_COUNT':[], #
'WORD_COUNT':[], #
'SYLLABLE_PER_WORD':[], #
'PERSONAL_PRONOUNS':[], #
'AVG_WORD_LENGTH':[]
}



nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('cmudict')
nltk.download('stopwords')




sia = SentimentIntensityAnalyzer()
d = cmudict.dict()


def count_syllables(words, pronunciation_dict):
    syllable_count = 0
    for word in words:
     if word in pronunciation_dict:
        phonetic_representations = pronunciation_dict[word]
        for pronunciation in phonetic_representations:
            for phoneme in pronunciation:
                if phoneme[-1].isdigit():
                    syllable_count += 1
    return syllable_count



# Function to calculate the variables for a given text
def calculate_variables(text):
    # Preprocess the text
    text = text.lower()
    sentences = sent_tokenize(text)
    words =[word for word in word_tokenize(text) if word.lower() not in stop_word] #removing stopwords
          

    # Calculate variables
    sentiment_scores = sia.polarity_scores(" ".join(words))
    positive_score = sentiment_scores['pos']
    negative_score = sentiment_scores['neg']
    polarity_score = sentiment_scores['compound']
    subjectivity_score = sentiment_scores['compound']

    average_sentence_length = len(word_tokenize(text)) / len(sentences)
    syllable_count=count_syllables(words,d)
    #to count complex count
    complex_word_count=0
    for word in words:
        if count_syllables(word,d)>=3:
            complex_word_count+=1
    percentage_complex_words = (complex_word_count / len(words)) * 100

    complex_words = []
    for word in words:
      if count_syllables(word,d) >= 3: #and word not in stopwords:
        complex_words.append(word)
    fog_index = 0.4 * (average_sentence_length + len(complex_words) / len(words) * 100)

    word_count = len(words)

    avg_syllables_per_word = sum(count_syllables(word,d) for word in words) / word_count

    tagged_words = nltk.pos_tag(words)
    personal_pronouns = sum(1 for word in word_tokenize(text) if word in pronouns )

    average_word_length = sum(len(word) for word in words) / word_count
    result['POSITIVE_SCORE'].append(positive_score)
    result['NEGATIVE_SCORE'].append(negative_score)
    result['POLARITY_SCORE'].append(polarity_score)
    result['SUBJECTIVITY_SCORE'].append(subjectivity_score)
    result['AVG_SENTENCE_LENGTH'].append(average_sentence_length)
    result['PERCENTAGE_OF_COMPLEX_WORDS'].append(percentage_complex_words)
    result['FOG_INDEX'].append(fog_index)
    result['AVG_NUMBER_OF_WORDS_PER_SENTENCE'].append((len(words)/len(sentences)))
    result['COMPLEX_WORD_COUNT'].append(complex_word_count)  
    result['WORD_COUNT'].append(word_count)
    result['SYLLABLE_PER_WORD'].append(avg_syllables_per_word)
    result['PERSONAL_PRONOUNS'].append(personal_pronouns)
    result['AVG_WORD_LENGTH'].append(average_word_length) 
    return [positive_score, negative_score, polarity_score, subjectivity_score, average_sentence_length,
            percentage_complex_words, fog_index, len(sentences), complex_word_count, word_count,
           avg_syllables_per_word, personal_pronouns, average_word_length]



j=0
for i in main_text:
    j=j+1
    calculate_variables(i)    
    print(j)





result['URL']=input_df['URL']
result['URL_ID']=input_df['URL_ID']

#converting result into dataframe
result_df=pd.DataFrame(result)

#converting output to csv
result_df.to_csv('D:\\blackcoffer\\submission.csv',index=False)
