from fastapi import FastAPI, Query
import feedparser
from datetime import datetime

app = FastAPI(title="LiveMint News API")

MINT_RSS_FEEDS = {
    "news": "https://www.livemint.com/rss/news",
    "markets": "https://www.livemint.com/rss/markets",
    "companies": "https://www.livemint.com/rss/companies",
    "technology": "https://www.livemint.com/rss/technology",
}

def parse_feed(url: str, limit: int):
    feed = feedparser.parse(url)

    items = []
    for entry in feed.entries[:limit]:
        items.append({
            "title": entry.title,
            "link": entry.link,
            "summary": getattr(entry, "summary", ""),
            "published": (
                datetime(*entry.published_parsed[:6]).isoformat()
                if hasattr(entry, "published_parsed")
                else None
            ),
        })
    return items


@app.get("/mint/news")
def get_mint_news(
    category: str = Query("news", enum=list(MINT_RSS_FEEDS.keys())),
    limit: int = Query(10, ge=1, le=50),
):
    url = MINT_RSS_FEEDS[category]
    data = parse_feed(url, limit)

    return {
        "source": "livemint",
        "category": category,
        "count": len(data),
        "articles": data,
    }
