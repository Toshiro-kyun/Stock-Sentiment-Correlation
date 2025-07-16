"""Settings"""
START_DATE = (datetime.now() - relativedelta(months=1)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')

STOCK = "NVDA"
PAGES = 5
MAX_LAG = 5

"""Constants"""
SEPERATOR = "-" * 200

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
import yfinance as yf

# NLP libaries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Plotting libraries
import matplotlib.pyplot as plt

# Statistics libraries
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

"""Functions"""
# Print pandas dataframe overview
def pd_view(df: pd.DataFrame):
    pd.set_option("display.max_rows", 10)
    print(df)
    pd.reset_option("display.max_rows")

    print(SEPERATOR)

# Fetch stock name and prices
def fetch_stock_name_data(stock, start_date, end_date, verbose: bool =True):
    ticker = yf.Ticker(stock)
    stock_name = ticker.info["displayName"]
    stock_data = ticker.history(start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)

    stock_data['Date'] = pd.to_datetime(stock_data["Date"]).dt.date

    if verbose:
        pd_view(df=stock_data)

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
        pd_view(df=news_data)

    return news_data

# Download nltk punkt_tab and stopwords
def download_nltk():
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    print(SEPERATOR)

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
    news_data['Sentiment'] = news_data['Cleaned_headline'].apply(get_sentiment_score, args=(analyzer,))
    
    if verbose:
        pd_view(df= news_data)

    return news_data

# Aggregate sentiments into pandas dataframe
def aggregate_day_sentiment(news_data, verbose: bool =True):
    aggregated_news_data = news_data.groupby('Date')['Sentiment'].sum().reset_index()

    if verbose:
        pd_view(df= aggregated_news_data)

    return aggregated_news_data

# Merge stock prices and sentiment scores
def merge_data(stock_data, sentiment_data, verbose: bool =True):
    merged_data = pd.merge(stock_data, sentiment_data, left_on="Date", right_on="Date", how="outer")

    if verbose:
        pd_view(df= merged_data)

    return merged_data

# Employ time-lagged statistical methods
def correlation_coef_statistics(stock_data, sentiment_data, stats_methods, max_lag: int, add_granger: bool = False, verbose: bool =True):
    stats_data = pd.DataFrame()
    stats_data["lag"] = range(-max_lag, max_lag + 1)

    for name, method in stats_methods:
        r_values, p_values = {}, {}
        for lag in range(-max_lag, max_lag + 1):
            shifted_stock_data = stock_data.shift(-lag)
            valid = shifted_stock_data.notna() & sentiment_data.notna()
            r_values[lag], p_values[lag] = method(sentiment_data[valid], shifted_stock_data[valid])
        
        stats_data[name + "--val"] = stats_data["lag"].map(r_values)
        stats_data[name + "--pval"] = stats_data["lag"].map(p_values)

    if add_granger:
        r_values, p_values = {}, {}
        granger_df = pd.DataFrame({"Stock_data": stock_data, "Sentiment_data": sentiment_data}).dropna()
        granger_results = grangercausalitytests(granger_df, maxlag=max_lag, verbose=False)
    
        for lag in range(1, max_lag + 1):
            ftest_result = granger_results[lag][0]['ssr_ftest']
            r_values[lag] = ftest_result[0]
            p_values[lag] = ftest_result[1]
        
        stats_data["Granger--val"] = stats_data["lag"].map(r_values)
        stats_data["Granger--pval"] = stats_data["lag"].map(p_values)

    if verbose:
        pd_view(df= stats_data)

    return stats_data

# Find optimal lag from p-values
def find_lag(stats_data):
    pvals_df = stats_data[[column for column in stats_data.columns if column.endswith("--pval")]]
    min_row_idx = pvals_df.min(axis=1).idxmin()
    value = stats_data.loc[min_row_idx, 'lag']

    return value

# Plot statistical data
def plot_stats_data(stats_data, stock_name):
    fig, (subplt1, subplt2) = plt.subplots(1, 2)
    for column in stats_data.columns:
        if column == "lag":
            continue
        
        if column.endswith("--val"):
            subplt2.plot(stats_data["lag"], stats_data[column], label=column)
        elif column.endswith("--pval"):
            subplt1.plot(stats_data["lag"], stats_data[column], label=column)

    subplt1.legend(loc="upper left")
    subplt1.axhline(y=0.05, color="r", linestyle="--")
    subplt1.set_title("P-values")
    subplt2.legend(loc="upper left")
    subplt2.set_title("Other values")
    subplt2.axhline(y=0, color="r", linestyle="--")

    fig.suptitle("Statisitcal Significance of Sentiment on Stock Price per Lag Interval")
    plt.show()

# Plot sentiment and stock price movement
def plot_graphs(combined_data, stock_name, lag: int =0):

    # Implement shift
    combined_data = combined_data[["Date", "Close", "Sentiment"]].copy()
    combined_data["Close"] = combined_data["Close"].shift(-lag)

    combined_data = combined_data.dropna()

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
    colors = ['green' if val >= 0 else 'red' for val in combined_data['Sentiment']]
    ax2.bar(combined_data['Date'], combined_data['Sentiment'], label='Aggregated Sentiment Score', color=colors, alpha=0.6)
    ax2.tick_params(axis='y')

    ax2_deviation = combined_data["Sentiment"].max() - combined_data["Sentiment"].min()
    ax2_mean = combined_data["Sentiment"].mean() 
    ax2.set_ylim(ax2_mean - ax2_deviation, ax2_mean + ax2_deviation)  # Set the right y-axis range

    fig.tight_layout()
    plt.title(f'{stock_name} Stock Price vs Aggregated Sentiment Score - Lag = {lag}')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
    plt.show()

"""Main Code"""
if __name__ == "__main__":
    print(SEPERATOR)
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

    # Apply statistical techniques to find significance with time-lag
    stats_methods = [
        ("Spearmanr", stats.spearmanr),
        ("Pearsonr", stats.pearsonr),
        ("Kendaltau", stats.kendalltau)
    ]

    stats_data = correlation_coef_statistics(stock_data=combined_data["Close"] , 
                                              sentiment_data= combined_data["Sentiment"], 
                                              stats_methods= stats_methods,
                                              max_lag= MAX_LAG,
                                              add_granger= False)
    
    #  Plot statistical data
    plot_stats_data(stats_data=stats_data, stock_name= stock_name)

    # Plot Stock Price and Sentiment over time

    #   Without time-lag
    plot_graphs(combined_data=combined_data, stock_name=stock_name)

    # Find the optimal lag
    optimal_lag = find_lag(stats_data)

    #   With optimized-time-lag
    plot_graphs(combined_data=combined_data, stock_name=stock_name, lag=optimal_lag)






