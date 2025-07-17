from datetime import datetime
from dateutil.relativedelta import relativedelta

"""Settings"""
START_DATE = (datetime.now() - relativedelta(months=1)).strftime('%Y-%m-%d') # Date a month ago
END_DATE = datetime.now().strftime('%Y-%m-%d')                               # Date of now
STOCK = "INSERT-STOCK-HERE"
PAGES = 1
MAX_LAG = 5
VERBOSE = True