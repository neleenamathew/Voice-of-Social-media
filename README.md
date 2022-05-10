# Vox-Populi - Voice of Social Media 

Vision: Our project is about reflecting the voice of millions of twitter users in India, to create value and sense from millions and zillions of tweets and hashtags

How we stand different?: There are many projects done on similar track. We stand different with the way in which we have captured the sentiments of people and projected it on the Heatmap of India using Power Bi.

What's next?: Our project can be of great help to media hubs, political activists, celebrity PRs etc. We have not fully completed our project. We want to include other languages for sentiment analysis apart from English. We also want to create a website where the tweet extraction, preprocessing, sentiment analysis and heatmap creation happens in background of the website and our final output should be where you search a particular hashtag and it returns the heatmap of India with sentiment analysis. Here you'll see a gist of our vision.
Anyone who is interested can feel free to collaborate and add or improvise on our project.


Required packages:

tweet-preprocessor
(download and install using "pip install PATH/tweet-preprocessor.zip)

fuzzywuzzy

seaborn

numpy

os # ability to use operating system

pandas

tweepy

geopy.geocoders

re # regex package for handling regular packages

string # allows to customize own strings

preprocessor # to clean, parse, tokenize tweets

nltk.corpus # for stopwords,wordnet,sentiwordnet 

nltk.tokenize # for word_tokenize

matplotlib.pyplot

vaderSentiment #for sentiment analysis

flask # to create website

Files to Run:

voice_of_social_media.py

dashboard

Required pre-work:

Twitter developer account

Create an app using twitter developer account to extract tweets

Note down the following keys and use it in python to extract tweets:

consumer_key = 'xxxx'

consumer_secret = 'xxxx'

access_key= 'xxxx'

access_secret = 'xxxx'


