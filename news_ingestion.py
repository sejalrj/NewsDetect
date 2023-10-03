import requests
import json
import pandas as pd
from datetime import datetime, timedelta

def main():
    # set API key and base URL
    api_key = "6ac15789ec154df3b073b43edb81ff7e"
    base_url = f"https://newsapi.org/v2/everything?apiKey={api_key}"

    # set search parameters
    query = "news" # query keyword
    language = "en" # language of articles
    page_size = 100 # number of articles per request
    total_results = 200 # maximum number of articles to fetch


    # # initialize dataframe to store articles
    df_articles = pd.DataFrame(columns=["title", "content", "source_name"])


    # set initial start date as today minus 1 day
    start_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    # loop until desired number of articles is reached
    while len(df_articles) < total_results:


        # set end date as today
        end_date = datetime.today().strftime("%Y-%m-%d")
        # set request URL
        url = f"{base_url}&q={query}&language={language}&pageSize={page_size}&from={start_date}&to={end_date}"
        # make request to News API
        response = requests.get(url)
        # parse response and extract articles
        articles = response.json()["articles"]
        # loop through articles and add to dataframe
        for article in articles:
            title = article["title"]
            content = article["content"]
            description = article["description"]
            source_name = article["source"]["name"] # get source name
            # add article to dataframe
            df_articles = df_articles.append({"title": title, "content": description + content, "source_name": source_name}, ignore_index=True)
            # stop if maximum number of articles is reached
            if len(df_articles) >= total_results:
               break
        # update start date for next request
        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=(len(df_articles) // page_size))).strftime("%Y-%m-%d")
        # print progress
        print(f"Fetched {len(df_articles)} articles so far...")

    df_articles.to_csv('/Users/abhishek/Documents/news.csv', index=False)