"""
News API Client
Fetches news articles and performs sentiment analysis
"""

import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests


class NewsClient:
    """
    News API client with sentiment analysis
    Fetches news articles and analyzes sentiment
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        self.base_url = api_key or os.getenv("NEWSAPI_URL")
        
        # Sentiment keywords
        self.positive_words = {
            'surge', 'soar', 'jump', 'rally', 'gain', 'rise', 'climb', 'advance',
            'strong', 'positive', 'bullish', 'optimistic', 'growth', 'profit',
            'success', 'breakthrough', 'innovation', 'record', 'beat', 'exceed',
            'outperform', 'upgrade', 'boost', 'momentum', 'recovery', 'expansion'
        }
        
        self.negative_words = {
            'plunge', 'tumble', 'drop', 'fall', 'decline', 'crash', 'sink',
            'weak', 'negative', 'bearish', 'pessimistic', 'loss', 'deficit',
            'failure', 'concern', 'worry', 'risk', 'miss', 'disappoint',
            'underperform', 'downgrade', 'cut', 'slump', 'recession', 'crisis'
        }
    
    def get_stock_news(
        self, 
        symbol: str, 
        company_name: str,
        days: int = 7
    ) -> List[Dict]:
        """
        Get recent news articles about a stock
        
        Args:
            symbol: Stock ticker symbol
            company_name: Full company name for better search
            days: Number of days to look back
        
        Returns:
            List of news articles with sentiment
        """
        if not self.api_key:
            return self._get_fallback_news(symbol, company_name)
        
        try:
            # Calculate date range
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            # Search query
            query = f"{company_name} OR {symbol}"
            
            # API request
            url = f"{self.base_url}/everything"
            params = {
                "q": query,
                "from": from_date,
                "sortBy": "publishedAt",
                "language": "en",
                "apiKey": self.api_key,
                "pageSize": 20
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get("articles", [])
            
            # Process and add sentiment
            results = []
            for article in articles:
                title = article.get("title", "")
                description = article.get("description", "")
                
                # Analyze sentiment
                sentiment_data = self.analyze_sentiment_simple(f"{title} {description}")
                
                results.append({
                    "title": title,
                    "description": description,
                    "url": article.get("url", ""),
                    "published_at": article.get("publishedAt", ""),
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "sentiment": sentiment_data["sentiment"],
                    "sentiment_score": sentiment_data["score"]
                })
            
            return results
            
        except requests.exceptions.RequestException as e:
            print(f"NewsAPI error: {e}, falling back to alternative")
            return self._get_fallback_news(symbol, company_name)
        except Exception as e:
            raise Exception(f"Failed to fetch news for {symbol}: {str(e)}")
    
    def _get_fallback_news(self, symbol: str, company_name: str) -> List[Dict]:
        """
        Fallback news source (Yahoo Finance RSS or web scraping)
        Used when NewsAPI is unavailable
        """
        try:
            # Use Yahoo Finance news as fallback
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            results = []
            for article in news[:10]:  # Limit to 10
                title = article.get("title", "")
                
                # Simple sentiment on title
                sentiment_data = self.analyze_sentiment_simple(title)
                
                results.append({
                    "title": title,
                    "description": article.get("summary", "")[:200],
                    "url": article.get("link", ""),
                    "published_at": datetime.fromtimestamp(
                        article.get("providerPublishTime", 0)
                    ).isoformat() if article.get("providerPublishTime") else "",
                    "source": article.get("publisher", "Yahoo Finance"),
                    "sentiment": sentiment_data["sentiment"],
                    "sentiment_score": sentiment_data["score"]
                })
            
            return results
            
        except Exception as e:
            # If all fails, return empty list
            print(f"Fallback news also failed: {e}")
            return []
    
    def analyze_sentiment_simple(self, text: str) -> Dict:
        """
        Simple keyword-based sentiment analysis
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment, score, and confidence
        """
        if not text:
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": "low"
            }
        
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Count positive and negative words
        positive_count = len(words.intersection(self.positive_words))
        negative_count = len(words.intersection(self.negative_words))
        
        # Calculate score (-1 to +1)
        total = positive_count + negative_count
        if total == 0:
            score = 0.0
            sentiment = "neutral"
            confidence = "low"
        else:
            score = (positive_count - negative_count) / total
            
            # Determine sentiment
            if score > 0.3:
                sentiment = "positive"
            elif score < -0.3:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            # Confidence based on total keyword matches
            if total >= 5:
                confidence = "high"
            elif total >= 2:
                confidence = "medium"
            else:
                confidence = "low"
        
        return {
            "sentiment": sentiment,
            "score": round(score, 2),
            "confidence": confidence,
            "positive_words": positive_count,
            "negative_words": negative_count
        }
    
    def get_market_news(self, limit: int = 10) -> List[Dict]:
        """
        Get general market news and trends
        
        Args:
            limit: Number of articles to fetch
        
        Returns:
            List of market news articles
        """
        if not self.api_key:
            return []
        
        try:
            url = f"{self.base_url}/top-headlines"
            params = {
                "category": "business",
                "language": "en",
                "apiKey": self.api_key,
                "pageSize": limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get("articles", [])
            
            results = []
            for article in articles:
                title = article.get("title", "")
                description = article.get("description", "")
                
                sentiment_data = self.analyze_sentiment_simple(f"{title} {description}")
                
                results.append({
                    "title": title,
                    "description": description,
                    "url": article.get("url", ""),
                    "published_at": article.get("publishedAt", ""),
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "sentiment": sentiment_data["sentiment"]
                })
            
            return results
            
        except Exception as e:
            print(f"Failed to fetch market news: {e}")
            return []
    
    def analyze_news_sentiment(
        self, 
        symbol: str, 
        company_name: str
    ) -> Dict:
        """
        Analyze overall sentiment from recent news
        
        Args:
            symbol: Stock ticker symbol
            company_name: Company name
        
        Returns:
            Dictionary with overall sentiment analysis
        """
        # Get recent news
        articles = self.get_stock_news(symbol, company_name, days=7)
        
        if not articles:
            return {
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "confidence": "low",
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "total_articles": 0,
                "summary": "No recent news available"
            }
        
        # Aggregate sentiments
        sentiments = [a["sentiment"] for a in articles]
        scores = [a["sentiment_score"] for a in articles]
        
        positive_count = sentiments.count("positive")
        negative_count = sentiments.count("negative")
        neutral_count = sentiments.count("neutral")
        
        # Calculate overall score
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Determine overall sentiment
        if avg_score > 0.2:
            overall = "positive"
        elif avg_score < -0.2:
            overall = "negative"
        else:
            overall = "neutral"
        
        # Confidence based on article count and agreement
        if len(articles) >= 10:
            if max(positive_count, negative_count, neutral_count) / len(articles) > 0.6:
                confidence = "high"
            else:
                confidence = "medium"
        elif len(articles) >= 5:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Generate summary
        summary = self._generate_sentiment_summary(
            overall, positive_count, negative_count, neutral_count
        )
        
        return {
            "overall_sentiment": overall,
            "sentiment_score": round(avg_score, 2),
            "confidence": confidence,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "total_articles": len(articles),
            "summary": summary,
            "recent_headlines": [a["title"] for a in articles[:5]]
        }
    
    def _generate_sentiment_summary(
        self, 
        overall: str, 
        positive: int, 
        negative: int, 
        neutral: int
    ) -> str:
        """Generate human-readable sentiment summary"""
        total = positive + negative + neutral
        
        if overall == "positive":
            return f"Positive sentiment across {total} articles ({positive} positive, {negative} negative, {neutral} neutral)"
        elif overall == "negative":
            return f"Negative sentiment across {total} articles ({positive} positive, {negative} negative, {neutral} neutral)"
        else:
            return f"Mixed/neutral sentiment across {total} articles ({positive} positive, {negative} negative, {neutral} neutral)"
    
    def get_trending_topics(self) -> List[str]:
        """
        Get trending financial topics
        
        Returns:
            List of trending topics/keywords
        """
        # Simple implementation - can be enhanced
        news = self.get_market_news(limit=20)
        
        # Extract common words from titles (simple approach)
        words = []
        for article in news:
            title_words = article["title"].lower().split()
            words.extend([w for w in title_words if len(w) > 4])
        
        # Count frequency
        from collections import Counter
        word_counts = Counter(words)
        
        # Return top 10
        return [word for word, count in word_counts.most_common(10)]


# Singleton instance
_news_client = None

def get_news_client() -> NewsClient:
    """Get singleton News client instance"""
    global _news_client
    if _news_client is None:
        _news_client = NewsClient()
    return _news_client