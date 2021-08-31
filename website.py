import pandas as pd
import os
#import webbrowser
import tweepy
from tweepy import OAuthHandler
import re
import numpy as np
import string
#from textblob import TextBlob
import preprocessor as p
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fuzzywuzzy import fuzz
from flask import Flask, redirect, url_for, request
app = Flask(__name__)

@app.route('/success/<name>')
def success(name):
    #Twitter credentials for the app
    consumer_key = 'xxxx'
    consumer_secret = 'xxxx'
    access_key= 'xxxx'
    access_secret = 'xxxx'
    #pass twitter credentials to tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    #file location changed to for clearer path
    hashtag_tweets = "demo1.csv"
    #columns of the csv file
    COLS = ['id', 'created_at', 'source', 'original_text','clean_text', #'sentiment','polarity','subjectivity', 
            'lang','favorite_count', 'retweet_count', 'original_author', 'possibly_sensitive', 
            'hashtags','user_mentions', 'place', 'place_coord_boundaries']
    #set two date variables for date range
    start_date = '2019-11-01'
    #end_date = '2019-11-30'
    # Happy Emoticons
    emoticons_happy = set([
        ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
        ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
        '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
        'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
        '<3'
        ])
    # Sad Emoticons
    emoticons_sad = set([
        ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
        ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
        ':c', ':{', '>:\\', ';('
        ])
    #Emoji patterns
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    #combine sad and happy emoticons
    emoticons = emoticons_happy.union(emoticons_sad)
    operators = set(('against', 'or','own','itself','off','over','the','has','had','can','because','if','as','until','for', 'under','again','between','into','through','after','before','up','down','in','out','once','any','all','both','each','few','more','most','some','such','no','nor','not','only','same','so','than','too','very','will','just',"don't","didn't","doesn't","hadn't","hasn't","haven't","isn't","needn't","wasn't","weren't","won't","wouldn't"))
    stop = set(stopwords.words('english')) - operators
    #method clean_tweets()
    def clean_tweets(tweet):
        stop_words = stop
        word_tokens = word_tokenize(tweet)
     
        #after tweepy preprocessing the colon left remain after removing mentions
        #or RT sign in the beginning of the tweet
        tweet = re.sub(r':', '', tweet)
        tweet = re.sub(r'‚Ä¶', '', tweet)
        #replace consecutive non-ASCII characters with a space
        tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
     
     
        #remove emojis from tweet
        tweet = emoji_pattern.sub(r'', tweet)
     
        #filter using NLTK library append it to a string
        filtered_tweet = [w for w in word_tokens if not w in stop_words]
        filtered_tweet = []
     
        #looping through conditions
        for w in word_tokens:
            #check tokens against stop words , emoticons and punctuations
            if w not in stop_words and w not in emoticons and w not in string.punctuation:
                filtered_tweet.append(w)
        return ' '.join(filtered_tweet)
    
    #method write_tweets()
    def write_tweets(keyword, file):
        # If the file exists, then read the existing data from the CSV file.
        if os.path.exists(file):
            df = pd.read_csv(file, header=0)
        else:
            df = pd.DataFrame(columns=COLS)
        #page attribute in tweepy.cursor and iteration
        for page in tweepy.Cursor(api.search, q=keyword,
                                  count=3200, include_rts=False, since=start_date).pages(50):
            for status in page:
                new_entry = []
                status = status._json
     
                ## check whether the tweet is in english or skip to the next tweet
                if status['lang'] != 'en':
                    continue
     
                #when run the code, below code replaces the retweet amount and
                #no of favorires that are changed since last download.
                if status['created_at'] in df['created_at'].values:
                    i = df.loc[df['created_at'] == status['created_at']].index[0]
                    if status['favorite_count'] != df.at[i, 'favorite_count'] or \
                       status['retweet_count'] != df.at[i, 'retweet_count']:
                        df.at[i, 'favorite_count'] = status['favorite_count']
                        df.at[i, 'retweet_count'] = status['retweet_count']
                    continue
     
     
               #tweepy preprocessing called for basic preprocessing
                clean_text = p.clean(status['text'])
     
                #call clean_tweet method for extra preprocessing
                filtered_tweet=clean_tweets(clean_text)
     
                #new entry append
                new_entry += [status['id'], status['created_at'],
                              status['source'], status['text'],filtered_tweet, status['lang'],
                              status['favorite_count'], status['retweet_count']]
     
                #to append original author of the tweet
                new_entry.append(status['user']['screen_name'])
     
                try:
                    is_sensitive = status['possibly_sensitive']
                except KeyError:
                    is_sensitive = None
                new_entry.append(is_sensitive)
     
                # hashtagas and mentiones are saved using comma separted
                hashtags = ", ".join([hashtag_item['text'] for hashtag_item in status['entities']['hashtags']])
                new_entry.append(hashtags)
                mentions = ", ".join([mention['screen_name'] for mention in status['entities']['user_mentions']])
                new_entry.append(mentions)
     
                #get location of the tweet if possible
                try:
                    location = status['user']['location']
                except TypeError:
                    location = ''
                new_entry.append(location)
     
                try:
                    coordinates = [coord for loc in status['place']['bounding_box']['coordinates'] for coord in loc]
                except TypeError:
                    coordinates = None
                new_entry.append(coordinates)
     
                single_tweet_df = pd.DataFrame([new_entry], columns=COLS)
                df = df.append(single_tweet_df, ignore_index=True)
                csvFile = open(file, 'a' ,encoding='utf-8')
        df.to_csv(csvFile, mode='a', columns=COLS, index=False, encoding="utf-8")
        
    #declare keywords as a query for three categories
    hashtag_keywords = name
    # call main method passing keywords and file path
    if os.path.isfile('C:\\Users\\ADMIN\\Desktop\\demo1.csv'):
        os.remove("C:\\Users\\ADMIN\\Desktop\\demo1.csv")
        
    write_tweets(hashtag_keywords, hashtag_tweets)
    
    df1 = pd.read_csv("hashtag.csv")
    df1 = df1.dropna(axis=0, how='all', thresh=None, subset=None, inplace=False)
    df2 = df1['place'].str.split(",", n = 1, expand = True)
    df2['places'] =df2[0]
    
    df1['location']= df2['places']
    df1.head()
    district = pd.read_excel("C:\\Users\\ADMIN\\Desktop\\list of districts.xlsx", sheet_name="Sheet2", colname="DISTRICS")
    
    
    bad = df2['places']
    
    good = district['DISTRICTS']
    
    # you can even set custom threshold and only return matches if above certain
    # matching threshold
    
    
    def correctspell(word, spellcorrect, thresh=70):
        mtchs = map(lambda x: fuzz.ratio(x, word) if fuzz.ratio(
            x, word) > thresh else None, spellcorrect)
        max = np.max(mtchs)
        if max is not None:
            return spellcorrect[mtchs.index(max)]
        else:
            return None
    
    
    # get correct spelling
    map(lambda x: correctspell(x, good, thresh=70), bad)
    
    df1['Match'] = ""
    india = pd.DataFrame()
    undefined = pd.DataFrame()
    
    for i in range(len(df1)):
        df1.iloc[i,15] = (district['DISTRICTS'] == df1.iloc[i,14] ).any() 
    
    df1['Match'] = df1['Match'].astype(str)
    india = df1.loc[df1['Match'] == 'True']
    india.reset_index(inplace = True) 
    
    undefined = df1.loc[df1['Match'] == 'False']
    undefined.reset_index(inplace = True)
    
    india.insert(17, "Negative","")
    india.insert(18, "Positive","")
    india.insert(19, "Neutral","")
    india.insert(20, "Compound","")
    
    undefined.insert(17, "Negative","")
    undefined.insert(18, "Positive","")
    undefined.insert(19, "Neutral","")
    undefined.insert(20, "Compound","")
    
    sid_obj = SentimentIntensityAnalyzer()
    for i in range(len(india)):
        sentiment_dict = sid_obj.polarity_scores(india.iloc[i,5])
        india.iloc[i,17] = sentiment_dict['neg']
        india.iloc[i,18] = sentiment_dict['pos']
        india.iloc[i,19] = sentiment_dict['neu']
        india.iloc[i,20] = sentiment_dict['compound']
        
    sid_obj = SentimentIntensityAnalyzer()
    for i in range(len(undefined)):
        sentiment_dict = sid_obj.polarity_scores(undefined.iloc[i,5])
        undefined.iloc[i,17] = sentiment_dict['neg']
        undefined.iloc[i,18] = sentiment_dict['pos']
        undefined.iloc[i,19] = sentiment_dict['neu']
        undefined.iloc[i,20] = sentiment_dict['compound']
    
    if os.path.isfile('C:\\Users\\ADMIN\\Desktop\\india.csv'):
        os.remove("C:\\Users\\ADMIN\\Desktop\\india.csv")
    if os.path.isfile('C:\\Users\\ADMIN\\Desktop\\undefined.csv'):
        os.remove("C:\\Users\\ADMIN\\Desktop\\undefined.csv")
        
    india.to_csv("india.csv")
    undefined.to_csv("undefined.csv")
    na = df1.iloc[1:10,:]
    
    return na

@app.route('/twitter',methods = ['POST', 'GET'])
def twitter():
   if request.method == 'POST':
      user = request.form['hashtag']
      return redirect(url_for('success',name = user))
   else:
      user = request.args.get('hashtag')
      return redirect(url_for('success',name = user))

if __name__ == '__main__':
   app.run()
