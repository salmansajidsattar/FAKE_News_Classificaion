from newsapi import NewsApiClient

api = NewsApiClient(api_key='04695eb463fc4240bf49029db7db9d72')
headlines=api.get_top_headlines(sources='bbc-news')
print(len(headlines['articles']))