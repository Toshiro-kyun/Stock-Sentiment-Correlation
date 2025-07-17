# Correlation Analysis of Stock Sentiment on Stock Prices
**A tool of analysis to determine whether Stock Sentiment is correlated to Stock Price.** This project uses the NewsAPI to rate Stock sentiment using vaderSentiment and then calculates various correlation scores in relation to the Stock Price. Shoutout to: https://github.com/mar-antaya/nvda-sentiment

## Usage
- Insert your NewsAPI key into a .env file. Create your API key here: https://newsapi.org
- Insert your settings into settings.py
- Run main.py


## Notes:
- Only 100 requests per day can be made to NewsAPI with a free account 

## Interpretation
If lag > 0 and p < 0.05, there is a statistically significant correlation between Stock Sentiment and Stock Price -- Likely that Stock Sentiment leads Stock Price in n days\
If lag < 0 and p < 0.05, there is a statistically significant correlation between Stock Sentiment and Stock Price -- Likely that Stock Price leads Stock Sentiment in n days\
If p > 0.05, there is no statistically significant correlation


