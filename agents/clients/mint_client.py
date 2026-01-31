"""
LiveMint RSS News Client
Fetches news from LiveMint RSS feeds
"""

import feedparser
from typing import Dict, List, Optional
from datetime import datetime


class MintClient:
    """
    LiveMint RSS feed client for Indian financial news
    """
    
    FEEDS = {
        "news": "https://www.livemint.com/rss/news",
        "markets": "https://www.livemint.com/rss/markets",
        "companies": "https://www.livemint.com/rss/companies",
        "technology": "https://www.livemint.com/rss/technology",
    }
    
    # Sentiment keywords
    POSITIVE_WORDS = {
        'surge', 'soar', 'jump', 'rally', 'gain', 'rise', 'climb', 'advance',
        'strong', 'positive', 'bullish', 'optimistic', 'growth', 'profit',
        'success', 'breakthrough', 'innovation', 'record', 'beat', 'exceed',
        'outperform', 'upgrade', 'boost', 'momentum', 'recovery', 'expansion',
        'buy', 'recommend', 'target', 'upside'
    }
    
    NEGATIVE_WORDS = {
        'plunge', 'tumble', 'drop', 'fall', 'decline', 'crash', 'sink',
        'weak', 'negative', 'bearish', 'pessimistic', 'loss', 'deficit',
        'failure', 'concern', 'worry', 'risk', 'miss', 'disappoint',
        'underperform', 'downgrade', 'cut', 'slump', 'recession', 'crisis',
        'sell', 'warning', 'fraud', 'scam'
    }
    
    def __init__(self):
        pass
    
    def _parse_feed(self, url: str, limit: int = 10) -> List[Dict]:
        """Parse an RSS feed and return articles"""
        try:
            feed = feedparser.parse(url)
            items = []
            
            for entry in feed.entries[:limit]:
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6]).isoformat()
                
                items.append({
                    "title": entry.title,
                    "link": entry.link,
                    "summary": getattr(entry, "summary", ""),
                    "published": published,
                    "source": "LiveMint"
                })
            
            return items
        except Exception as e:
            print(f"[MINT ERROR] Failed to parse feed: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Simple sentiment analysis"""
        text_lower = text.lower()
        words = set(text_lower.split())
        
        positive_count = len(words & self.POSITIVE_WORDS)
        negative_count = len(words & self.NEGATIVE_WORDS)
        
        if positive_count > negative_count:
            sentiment = "positive"
            score = min(1.0, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = "negative"
            score = max(-1.0, -0.5 - (negative_count - positive_count) * 0.1)
        else:
            sentiment = "neutral"
            score = 0.0
        
        return {"sentiment": sentiment, "score": round(score, 2)}
    
    def get_market_news(self, category: str = "markets", limit: int = 10) -> List[Dict]:
        """
        Get general market news from LiveMint
        
        Args:
            category: Feed category (news, markets, companies, technology)
            limit: Number of articles to fetch
        
        Returns:
            List of news articles
        """
        url = self.FEEDS.get(category, self.FEEDS["markets"])
        articles = self._parse_feed(url, limit)
        
        # Add sentiment to each article
        for article in articles:
            text = f"{article['title']} {article['summary']}"
            sentiment_data = self._analyze_sentiment(text)
            article["sentiment"] = sentiment_data["sentiment"]
            article["sentiment_score"] = sentiment_data["score"]
        
        print(f"[MINT] Fetched {len(articles)} articles from {category}")
        return articles
    
    def get_stock_news(
        self,
        symbol: str,
        company_name: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get news related to a specific stock from LiveMint
        Searches across all feeds for relevant articles
        
        Args:
            symbol: Stock ticker (e.g., RELIANCE.NS)
            company_name: Company name for matching
            limit: Max articles to return
        
        Returns:
            List of relevant news articles
        """
        # Clean symbol for matching
        symbol_base = symbol.replace('.NS', '').replace('.BO', '').upper()
        company_lower = company_name.lower()
        
        all_articles = []
        
        # Fetch from multiple feeds
        for category in ["companies", "markets", "news"]:
            articles = self._parse_feed(self.FEEDS[category], limit=30)
            all_articles.extend(articles)
        
        # Filter for relevant articles
        relevant = []
        for article in all_articles:
            text = f"{article['title']} {article['summary']}".lower()
            
            # Check if article mentions the stock
            if (symbol_base.lower() in text or 
                company_lower in text or
                any(word in text for word in company_lower.split()[:2])):
                
                # Add sentiment
                sentiment_data = self._analyze_sentiment(text)
                article["sentiment"] = sentiment_data["sentiment"]
                article["sentiment_score"] = sentiment_data["score"]
                relevant.append(article)
                
                if len(relevant) >= limit:
                    break
        
        print(f"[MINT] Found {len(relevant)} relevant articles for {symbol}")
        return relevant


# Singleton instance
_mint_client = None

def get_mint_client() -> MintClient:
    """Get or create singleton MintClient instance"""
    global _mint_client
    if _mint_client is None:
        _mint_client = MintClient()
    return _mint_client
