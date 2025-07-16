"""Libaries"""
# Standard libraries
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import requests
from dotenv import load_dotenv

# Stock prices libraries
import scipy.stats
import yfinance as yf

# NLP libaries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Plotting libraries
import matplotlib.pyplot as plt

# Statistics libraries
import scipy

"""Settings"""
START_DATE = (datetime.now() - relativedelta(months=1)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')

STOCK = "NVDA"
PAGES = 5

"""Functions"""
# Fetch stock name and prices
def fetch_stock_name_data(stock, start_date, end_date, verbose: bool =True):
    ticker = yf.Ticker(stock)
    stock_name = ticker.info["displayName"]
    stock_data = ticker.history(start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)

    stock_data['Date'] = pd.to_datetime(stock_data["Date"]).dt.date

    if verbose:
        print(stock_data.head())

    return stock_name, stock_data

# Fetch news on stock 
def fetch_stock_news(stock_name, start_date, end_date, pages: int, verbose: bool =True):

    news_data = pd.DataFrame(columns= ['Date', 'Headline'])

    api_key = os.getenv("NEWS_API_KEY")

    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': stock_name,
        'from': start_date, 
        'end': end_date,
        'sortBy': 'relevancy',
        'apiKey': api_key,
        'language': 'en',
        'pageSize': 10
    }

    for page in range(1, pages):
        # Make the request
        params['page'] = page
        response = requests.get(url, params=params)
        data = response.json()

        # Check for errors
        if data['status'] != 'ok':
            raise Exception(f"NewsAPI error: {data['message']}")

        # Extract articles
        articles = data['articles']

        # Convert to DataFrame
        new_news_data = pd.DataFrame(articles)
        new_news_data = new_news_data[['publishedAt', 'title']]
        new_news_data.columns = ['Date', 'Headline']
        
        new_news_data['Date'] = pd.to_datetime(new_news_data["Date"]).dt.date
        news_data = pd.concat([news_data, new_news_data])

    if verbose:
        print(news_data.head())
        print(len(news_data))

    return news_data

# Download nltk punkt_tab and stopwords
def download_nltk():
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# Preprocess sentence
def preprocess_text(text, stop_words):
    words = word_tokenize(text)
    words = [word for word in words if word.isalpha()]
    words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(words)

# Preprocess headlines
def preprocess_headlines(news_data):
    stop_words = set(stopwords.words('english'))
    news_data['Cleaned_headline'] = news_data['Headline'].apply(preprocess_text, args=(stop_words,))

# Sentiment analysis per headline
def get_sentiment_score(text, analyzer):
    score = analyzer.polarity_scores(text)
    return score['compound']

# Sentiment analysis
def sentiment_analysis(news_data, verbose: bool =True):
    # Preprocessing steps
    download_nltk()
    preprocess_headlines(news_data=news_data)

    # Perform sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    news_data['Sentiment_score'] = news_data['Cleaned_headline'].apply(get_sentiment_score, args=(analyzer,))
    
    if verbose:
        print(news_data.head())

    return news_data

# Aggregate sentiments into pandas dataframe
def aggregate_day_sentiment(news_data, verbose=True):
    aggregated_news_data = news_data.groupby('Date')['Sentiment_score'].sum().reset_index()

    if verbose:
        print(aggregated_news_data.head())

    return aggregated_news_data

# Merge stock prices and sentiment scores
def merge_data(stock_data, sentiment_data, verbose: bool =True):
    merged_data = pd.merge(stock_data, sentiment_data, left_on="Date", right_on="Date", how="inner")

    if verbose:
        print(merged_data.head())

    return merged_data

def plot_graphs(combined_data, stock_name):
    # Visualize the data with a secondary y-axis and bar plot for aggregated sentiment scores
    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'{stock_name} Stock Price')
    ax1.plot(combined_data['Date'], combined_data['Close'], label=f'{stock_name} Stock Price')
    ax1.tick_params(axis='y')

    ax1_deviation = combined_data["Close"].max() - combined_data["Close"].min()
    ax1_mean = combined_data["Close"].mean() 
    ax1.set_ylim(ax1_mean - ax1_deviation, ax1_mean + ax1_deviation)  # Set the left y-axis range

    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Aggregated Sentiment Score')

    # Use different colors for positive and negative bar values
    colors = ['green' if val >= 0 else 'red' for val in combined_data['Sentiment_score']]
    ax2.bar(combined_data['Date'], combined_data['Sentiment_score'], label='Aggregated Sentiment Score', color=colors, alpha=0.6)
    ax2.tick_params(axis='y')

    ax2_deviation = combined_data["Sentiment_score"].max() - combined_data["Sentiment_score"].min()
    ax2_mean = combined_data["Sentiment_score"].mean() 
    ax2.set_ylim(ax2_mean - ax2_deviation, ax2_mean + ax2_deviation)  # Set the right y-axis range

    fig.tight_layout()
    plt.title(f'{stock_name} Stock Price vs Aggregated Sentiment Score')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
    plt.show()

# Merge sentiment and 
"""Main Code"""
if __name__ == "__main__":
    # Load .env for news api-key
    load_dotenv()

    # Fetch Stock name and Stock data
    stock_name, stock_data = fetch_stock_name_data(stock = STOCK, start_date=START_DATE, end_date=END_DATE)

    # Fetch news
    news_data = fetch_stock_news(stock_name= stock_name, start_date=START_DATE, end_date=END_DATE, pages=PAGES)

    # Sentiment analysis
    news_data = sentiment_analysis(news_data=news_data)
    
    # Aggregate sentiment dataframe
    sentiment_data = aggregate_day_sentiment(news_data=news_data)

    # Combined data
    combined_data = merge_data(stock_data=stock_data, sentiment_data=sentiment_data)

    print(scipy.stats.spearmanr(combined_data["Close"].values, combined_data["Sentiment_score"].values)[0])

    plt.scatter(combined_data['Sentiment_score'], combined_data['Close'])
    plt.show()

    # Plot data
    plot_graphs(combined_data=combined_data, stock_name=stock_name)





