from datetime import datetime
from dateutil.relativedelta import relativedelta

"""Settings"""
START_DATE = (datetime.now() - relativedelta(months=1)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')
STOCK = "TSLA"
PAGES = 5
MAX_LAG = 5
VERBOSE = True