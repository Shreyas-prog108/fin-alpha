"""
Prompts for synthesizing and presenting results
"""

SYNTHESIS_PROMPT = """
Synthesize all gathered data into clear recommendation:

Structure:
1. Current Status (price, market position)
2. Risk Assessment (volatility, risk level)
3. Market Sentiment (news, trends)
4. Price Prediction (forecast, confidence)
5. Final Recommendation (BUY/HOLD/SELL with reasoning)

Be specific, quantitative, and actionable.
"""

CONFIDENCE_ASSESSMENT_PROMPT = """
Assess confidence level based on:
- Data quality
- Volatility levels
- Sentiment clarity
- Prediction agreement

Output: HIGH/MEDIUM/LOW confidence with explanation
"""

DISCLAIMER_PROMPT = """
Always include appropriate disclaimers:
- Not financial advice
- Past performance doesn't guarantee future results
- Consider personal risk tolerance
- Consult financial advisor
"""