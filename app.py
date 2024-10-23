#Main Flask App

from flask import Flask,session, render_template, request, redirect, url_for,jsonify
import requests
import pyrebase
from collections import OrderedDict
import os
import firebase_admin
from firebase_admin import credentials, db,auth
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
app = Flask(__name__)



load_dotenv()



# Set the secret key from the environment variable
app.secret_key = os.getenv("SECRET_KEY")
if app.secret_key is None:
    raise ValueError("No SECRET_KEY set for Flask application")
# Retrieve the service account configuration
Config= os.getenv("GOOGLE_SERVICE_ACCOUNT")

# Parse the JSON string to a dictionary
config = json.loads(service_account_info)

newsdataio_url = "https://newsdata.io/api/1/news"  # Example URL
api_key=os.getenv("api_key")
# Initialize Firebase Admin SDK
cred = credentials.Certificate(Config)  # Update with your service account key
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://investment-insights-ae2e6-default-rtdb.firebaseio.com/'
})

# Define your Firebase database reference
ref = db.reference('/users')

from transformers import BertTokenizer
import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import BertTokenizer,BertModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import tqdm
import numpy as np
##------------------------------------------------------------------ALL FUNCTIONS---------------------------------------------------------------------------------------------------------------
# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 3

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits


 # Replace with your tokenizer name if different


def find_sentiment(text):
    if torch.cuda.is_available():       
        device = torch.device("cuda")
    
    else:
        device = torch.device("cpu")
    # Define the path to your saved model
    model_path = "bert_for_sentimet.pt"

    # Load the model (assuming BertClassifier is your model class)
    model = torch.load(model_path)
    # Import the tokenizer (assuming you saved it or can download it)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Replace with your tokenizer name if different

    # Function to preprocess text for BERT
    def preprocess_for_bert(text):
        """Performs tokenization and mask creation for a single text input."""
        encoded_text = tokenizer(text, add_special_tokens=True, return_tensors='pt')
        return encoded_text['input_ids'], encoded_text['attention_mask']

        # Define your test text
        

        # Preprocess the test text
    test_inputs, test_masks = preprocess_for_bert(text)

        # Create a dummy dataset and dataloader for the test case (batch size of 1)
    test_dataset = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1)

    # Function to predict sentiment (assuming yours returns probabilities)
    def predict_sentiment(model, test_dataloader):
        """Perform a forward pass on the model to predict probabilities on the test set."""
        all_logits = []
        with torch.no_grad():
            for batch in test_dataloader:
                b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]
                logits = model(b_input_ids, b_attn_mask)
                all_logits.append(logits)
        all_logits = torch.cat(all_logits, dim=0)
        probs = F.softmax(all_logits, dim=1).cpu().numpy()
        return probs

    # Predict sentiment
    sentiment_probs = predict_sentiment(model, test_dataloader)[0]
    sentiment_probs = list(sentiment_probs)
    sentiment_mapping = { 0:'positive',  1:'negative',  2:'neutral'}
    maximum = max(sentiment_probs)
    sentiment = sentiment_mapping[sentiment_probs.index(maximum)]
    
    # Print the predicted sentiment probabilities
    return sentiment,text


def create_user(email,password):
  try:
      user= auth.create_user_with_email_and_password(email,password)
      return "User Creation Successful. You can now login"
  except Exception as e :
      error_message = str(e)
      return f"OOPS Something went wrong \n{error_message}"

def sentiment_analysis(headlines, article_links, descriptions):
    sentiment_results = []
    for headline, link, description in zip(headlines, article_links, descriptions):
        # Sentiment Analysis Placeholders
        sentiment = find_sentiment(headline)[0]  # Placeholder sentiment analysis result

        # Store headline, link, description, and sentiment in a dictionary
        result = OrderedDict()
        result['headline'] = headline
        result['link'] = link
        result['description'] = description
        result['sentiment'] = sentiment
        sentiment_results.append(result)


    return sentiment_results



def fetch_user_watchlist(user_email):
    try:
        print("Fetching user watchlist for:", user_email)
        user_watchlists = ref.order_by_child('email').equal_to(user_email).get()
        print("User watchlists found:", user_watchlists)
        watchlist = []
        for key, value in user_watchlists.items():
            print("Key:", key)
            print("Value:", value)
            watchlist.extend(value.get("watchlist", []))  # Use extend instead of append to flatten the list
        print("Final watchlist:", watchlist)
        return watchlist
    except Exception as e:
        print("Error fetching watchlist:", e)
        return []


##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------STOCK FUNCTIONS-------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------
def getTicker(company_name):
    yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}

    res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
    data = res.json()

    company_code = data['quotes'][0]['symbol']
    return company_code

def get_stock_data(company_name, start_date="2020-01-01", end_date="2024-05-22"):
  """
  This function retrieves historical stock price data for a company (checks availability, avoids downloads).

  Args:
      company_name (str): The name of the company.
      start_date (str, optional): The starting date in YYYY-MM-DD format. Defaults to None (maximum available).
      end_date (str, optional): The ending date in YYYY-MM-DD format. Defaults to None (today).

  Returns:
      pandas.DataFrame: A DataFrame containing historical stock prices (empty if unavailable).
  """

  # Get ticker symbol (assuming you have a separate function for this)
  ticker_symbol = getTicker(company_name)

  # Create a yfinance Ticker object
  ticker = yf.Ticker(ticker_symbol)

  # Download historical data for the specified period (avoid downloading all)
  if start_date and end_date:
      data = ticker.history(period = '3mo')
  else:
      sensex = getTicker('Sensex')
      ticker_default = yf.Ticker(sensex)
      data = ticker_default.history(period='3mo')
    
  # Check if DataFrame is empty (handle unavailable company data)
  if data.empty:
      print('Sensex!')
      sensex = getTicker('Sensex')
      ticker_default = yf.Ticker(sensex)
      data = ticker_default.history(period ='3mo')
  else:
      pass
  

  data.reset_index(inplace=True)
  data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
  data.drop(['Dividends','Stock Splits'], inplace=True, axis=1)
  data.to_dict(orient='records')
  
  return data

def analyze_stock_data(stock_data):
  """
  Analyzes stock data and returns a dictionary with key metrics.

  Args:
      stock_data: A list of dictionaries, where each dictionary represents
          a day's stock data with keys 'Date', 'Open', 'High', 'Low',
          'Close', and 'Volume'.

  Returns:
      A dictionary containing the following key metrics:
          - greatest_increase: The date and percentage increase of the
              greatest increase in the last month.
          - greatest_decrease: The date and percentage decrease of the
              greatest decrease in the last month.
          - current_close: The closing price of the last day.
          - volatility: The average daily fluctuation over the last week.
          - moving_average_1_month: The 30-day moving average of the closing price.
  """

  # Convert data to pandas DataFrame for easier manipulation
  import pandas as pd
  df = pd.DataFrame(stock_data)

  # Ensure dates are parsed as datetime
  df['Date'] = pd.to_datetime(df['Date'])

  # Filter data for the last month
  last_month_data = df[df['Date'].dt.month == df['Date'].iloc[-1].month]

  # Greatest increase
  greatest_increase = last_month_data.sort_values(by='Close', ascending=False).iloc[0]
  greatest_increase_pct = (greatest_increase['Close'] - greatest_increase['Open']) / greatest_increase['Open'] * 100

  # Greatest decrease
  greatest_decrease = last_month_data.sort_values(by='Close', ascending=True).iloc[0]
  greatest_decrease_pct = (greatest_decrease['Open'] - greatest_decrease['Close']) / greatest_decrease['Open'] * 100

  # Current close value
  current_close = df.iloc[-1]['Close']

  # Volatility (average daily fluctuation over the last week)
  last_week_data = df[df['Date'] >= df['Date'].iloc[-1] - pd.Timedelta(days=7)]
  volatility = last_week_data['Close'].pct_change().std() * 100

  # Moving average of the stock over 1 month
  moving_average_1_month = df['Close'].rolling(window=30).mean().iloc[-1]

  # Create and return the dictionary
  results = {
      "greatest_increase": {
          "date": greatest_increase['Date'].strftime('%Y-%m-%d'),
          "increase_pct": greatest_increase_pct
      },
      "greatest_decrease": {
          "date": greatest_decrease['Date'].strftime('%Y-%m-%d'),
          "decrease_pct": greatest_decrease_pct
      },
      "current_close": current_close,
      "volatility": volatility,
      "moving_average_1_month": moving_average_1_month
  }

  return results
    


def fetch_yahoo_data(company, interval, ema_period=20, rsi_period=14):
    ticker = getTicker(company)
    end_date = datetime.now()
    if interval in ['1m', '5m']:
        start_date = end_date - timedelta(days=7)  # Past 7 days for minute intervals
    elif interval in ['15m', '60m']:
        start_date = end_date - timedelta(days=60)  # Past 60 days for 15-minute and hourly intervals
    elif interval == '1d':
        start_date = end_date - timedelta(days=365)  # Past 1 year for daily intervals
    elif interval == '1wk':
        start_date = end_date - timedelta(weeks=1)  # Past week for weekly intervals
    elif interval == '1mo':
        start_date = end_date - timedelta(days=30)  # Past month for monthly intervals (approximately 30 days)
    elif interval == '3mo':
        start_date = end_date - timedelta(days=90)  # Past 3 months for quarterly intervals (approximately 90 days)
    elif interval == '6mo':
        start_date = end_date - timedelta(days=180)  # Past 6 months for semi-annual intervals (approximately 180 days)
    elif interval == '1yr':
        start_date = end_date - timedelta(days=365)  # Past 1 year for annual intervals
    elif interval == '5yr':
        start_date = end_date - timedelta(days=365*5)
    else:
        raise ValueError("Unsupported interval")

    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    data['EMA'] = ta.ema(data['Close'], length=ema_period)
    data['RSI'] = ta.rsi(data['Close'], length=rsi_period)

    candlestick_data = [
        {
            'time': int(row.Index.timestamp()),
            'open': row.Open,
            'high': row.High,
            'low': row.Low,
            'close': row.Close
        }
        for row in data.itertuples()
    ]

    ema_data = [
        {
            'time': int(row.Index.timestamp()),
            'value': row.EMA
        }
        for row in data.itertuples() if not pd.isna(row.EMA)
    ]

    rsi_data = [
        {
            'time': int(row.Index.timestamp()),
            'value': row.RSI if not pd.isna(row.RSI) else 0  # Convert NaN to zero
        }
        for row in data.itertuples()
    ]

    return candlestick_data, ema_data, rsi_data



##-----------------------------------------------------------------FLASK APP ROUTES-------------------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            session['user'] = email
            return redirect(url_for('watchlist'))
        except Exception as e:
            error_message = str(e)
            return f"Failed to login,{error_message}"
    return render_template('login.html')
 

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        watchlist_total = request.form.getlist('watchlist')
        # Split the string into a list
        list_of_strings = watchlist_total[0].split(",")
        watchlist = []
# Print each element of the list on a new line
        for element in list_of_strings:
          watchlist.append(element)

        print(watchlist)

        if password != confirm_password:
            error_message = "Passwords do not match."
            return render_template('signup.html', error=error_message)

        try:
            # Create user with email and password
            user = auth.create_user(email=email, password=password)
            
            # Add additional user data to Firebase Realtime Database
            user_data = {"email": email, "watchlist": watchlist}
            ref.child(user.uid).set(user_data)

            return redirect(url_for('login'))
            
        except Exception as e:
            error_message = f"Failed to signup: {str(e)}"
            return render_template('signup.html', error=error_message)

    return render_template('signup.html')





@app.route('/search_news', methods=['POST'])
def search_news():
    # Extract keyword and user's API key from request body
    global keyword
    keyword = request.json.get('keyword')  
    user_api_key = api_key

    
    company_name = keyword
    # Make request to newsdata.io API with user's API key
    params = {
        'apikey': user_api_key,
        'qInTitle': keyword,
        'language': "en"
    }
    response = requests.get(newsdataio_url, params=params)
    news_data = response.json()
    headlines = [article['title'] for article in news_data['results']]
    descriptions=[article['description']for article in news_data['results']]
    article_links=[article['link']for article in news_data['results']]
    articles = [
        {'headline': headline, 'description': description, 'link': link, 'sentiment': 'Neutral'}
        for headline, description, link in zip(headlines, descriptions, article_links)
    ]
    stock_data = get_stock_data(company_name)
    stock_analysis = analyze_stock_data(stock_data)
    return jsonify({
        "status": "success", 
        keyword: articles, 
        "stock_analysis": stock_analysis
    })
    # return jsonify({"status": "success", keyword: sentiment_analysis(headlines,article_links,descriptions),"stock_analysis": stock_analysis})




@app.route('/company_analysis', methods=['GET'])
def company_analysis():
    # Get the company name from the request
    company = request.args.get('company')
    user_api_key = api_key
    newsdataio_url = "https://newsdata.io/api/1/news"  # Example URL

    # Make request to newsdata.io API with user's API key and company name
    params = {
        'apikey': user_api_key,
        'qInTitle': company,
        'language': "en"
    }
    response = requests.get(newsdataio_url, params=params)
    news_data = response.json()
    stock_data = get_stock_data(company)
    stock_analysis = analyze_stock_data(stock_data)
    # Extract relevant information from the news data
    headlines = [article['title'] for article in news_data['results']]
    descriptions = [article['description'] for article in news_data['results']]
    article_links = [article['link'] for article in news_data['results']]

    # Perform sentiment analysis on the news articles
    sentiment_results = sentiment_analysis(headlines, article_links, descriptions)

    # Return the sentiment analysis results as JSON
    # return jsonify({"status": "success", "company": company_name, "sentiment_analysis": sentiment_results})
    return render_template('company_analysis.html', company=company, articles=sentiment_results,stock_analysis = stock_analysis)

@app.route('/watchlist')
def watchlist():
    if 'user' in session:
        #print(session['user'])
        global user_watchlist
        user_watchlist = fetch_user_watchlist(session['user'])
        print("Value:", user_watchlist)
        
        return render_template('watchlist.html', watchlist=user_watchlist)
    else:
        return redirect(url_for('login'))
   
@app.route('/',methods=["GET","POST"])
def home():
    return render_template('login.html')


@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/api/data/<company>/<interval>/<int:ema_period>/<int:rsi_period>')
def get_data(company, interval, ema_period, rsi_period):
    candlestick_data, ema_data, rsi_data = fetch_yahoo_data(company, interval, ema_period, rsi_period)
    return jsonify({'candlestick': candlestick_data, 'ema': ema_data, 'rsi': rsi_data})

@app.route('/api/symbols')
def get_symbols():
    
    symbols = user_watchlist
    return jsonify(symbols)

if __name__ == '__main__':
    app.run(debug=True,port=8080)