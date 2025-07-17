# Correlation Analysis of Stock Sentiment on Stock Prices
<ins>A tool of analysis to determine whether Stock Sentiment is correlated to Stock Price.</ins> 

This project uses the NewsAPI to rate Stock sentiment using vaderSentiment and then calculates various correlation scores in relation to the Stock Price. Shoutout to: https://github.com/mar-antaya/nvda-sentiment

## Usage
- Insert your NewsAPI key into a .env file. Create your API key here: https://newsapi.org
  ````
  NEWS_API_KEY = "YOUR NEWS-API KEY HERE"
  ````
- Insert your settings into settings.py. Refer to the settings section below.

- Run main.py
  ````
  main.py 
  ````

## Settings

  ````
  START_DATE = "2025-01-01"         # Starting date in YYYY-MM-DD format
  END_DATE = "2025-02-01"           # End date in YYYY-MM-DD format
  
  STOCK = "TSLA"                    # Stock abbreviation
  PAGES = 1                         # Number of pages to fetch from News API
  MAX_LAG = 5                       # Maximum lag to consider
  ````

## Notes
- Only 100 requests per day can be made to NewsAPI with a free account
- Granger Causality is automatically added - can only be computed for lag=5 

## Interpretation
- If p < 0.05 - Statistically Significant Correlation between Stock Sentiment & Stock Price 
    - Lag > 0: Stock Sentiment leads Stock Price in {lag} days
    - Lag < 0: Stock Price leads Stock Sentiment in {lag} days
- If p > 0.05 - No Statistically Significant Correlation between Stock Sentiment & Stock Price 

Completed during 2025/07[^1].
