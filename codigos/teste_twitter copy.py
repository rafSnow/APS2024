import snscrape.modules.twitter as sntwitter
import pandas as pd
maxTweets = 100
i = 0
tweets_list = []
for tweet in sntwitter.TwitterSearchScraper(' since:2022-10-02 until:2022-10-07').get_items():
  if i > maxTweets:
    break
  tweets_list.append([tweet.date, tweet.url, tweet.username, tweet.content])
  i = i + 1
tweets_df = pd.DataFrame(tweets_list, columns=['date', 'url','username', 'content' ])
tweets_df.to_csv()