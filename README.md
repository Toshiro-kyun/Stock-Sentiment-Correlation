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
  python3 main.py
  
  ````

## Settings

  ````
  START_DATE = (datetime.now() - relativedelta(months=1)).strftime('%Y-%m-%d') # Starting date in YYYY-MM-DD format
  END_DATE = datetime.now().strftime('%Y-%m-%d')                               # End date in YYYY-MM-DD format
  
  STOCK = "TSLA"                                                               # Stock abbreviation
  PAGES = 5                                                                    # Number of pages to fetch from News API
  MAX_LAG = 5                                                                  # Maximum lag to consider
  ````

## Notes
- Only 100 requests per day can be made to NewsAPI with a free account
- For interpretation:

## Interpretation
If lag > 0 and p < 0.05, there is a statistically significant correlation between Stock Sentiment and Stock Price -- Likely that Stock Sentiment leads Stock Price in n days\
If lag < 0 and p < 0.05, there is a statistically significant correlation between Stock Sentiment and Stock Price -- Likely that Stock Price leads Stock Sentiment in n days\
If p > 0.05, there is no statistically significant correlation


