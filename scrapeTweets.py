import pandas as pd
from ntscraper import Nitter
from time import sleep
import random

def scrape_with_retries(handle, max_tweets=10):
    # List of currently active Nitter instances (as of late 2025)
    # We iterate through these if one fails.
    instances = [
        "https://nitter.privacydev.net",
        "https://nitter.poast.org",
        "https://xcancel.com",
        "https://nitter.lucabased.xyz",
        "https://nitter.salastil.com"
    ]
    
    random.shuffle(instances)  # Randomize order to distribute load

    for instance in instances:
        try:
            print(f"   Trying instance: {instance} ...")
            #init nitter scraper
            scraper = Nitter(log_level=1, skip_instance_check=True)
            #scrape
            tweets = scraper.get_tweets(handle, mode='user', number=max_tweets, instance=instance)
            
            # Check if we actually got data
            if tweets and 'tweets' in tweets and len(tweets['tweets']) > 0:
                print(f"   ✅ Success with {instance}")
                return tweets['tweets']
            else:
                print(f"{instance} returned no tweets. Trying next...")
                
        except Exception as e:
            continue
            
    print(f"All instances failed for {handle}.")
    return []

def run_scraper(profiles, count=20):
    all_data = []

    for handle in profiles:
        print(f"\n--- Scraping @{handle} ---")
        tweets = scrape_with_retries(handle, max_tweets=count)
        
        for t in tweets:
            all_data.append({
                "profile": handle,
                "date": t.get('date'),
                "text": t.get('text'),
                "link": t.get('link'),
                "likes": t.get('stats', {}).get('likes', 0),
                "retweets": t.get('stats', {}).get('retweets', 0)
            })
            
        # Sleep to be polite
        sleep(2)

    return all_data

if __name__ == "__main__":
    #handles
    targets = ["realDonaldTrump", "rihanna", "elonmusk", "JDVance", "HillaryClinton", "AP", "BarackObama", "narrativefoil", "hansmollman", "Keefler_Elf"]

    data = run_scraper(targets, count=100)
    
    if data:
        df = pd.DataFrame(data)
        print(f"\n✅ Total Tweets Scraped: {len(df)}")
        df.to_csv("nitter_scraped_tweets.csv", index=False)
        print("Saved to 'nitter_scraped_tweets.csv'")
    else:
        print("\n No data collected. Twitter (X) might be blocking all known Nitter instances right now.")