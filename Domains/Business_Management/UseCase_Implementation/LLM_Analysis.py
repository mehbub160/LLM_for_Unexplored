# ============================================================================
# LLM-ASSISTED INVESTMENT DECISION SUPPORT IN GOLD FOREX TRADING
# Complete implementation with news validation and market analysis
# ============================================================================

import json
import os
import numpy as np
from datetime import datetime, timedelta
import random
import math
import pandas as pd
from collections import defaultdict
import requests
import re

# ============================================================================
# 1. MARKET DATA SIMULATOR
# ============================================================================

class FinancialDataSimulator:
    """Simulates comprehensive financial market data for gold trading analysis"""
    
    def __init__(self):
        # Market indicators and their typical ranges
        self.market_indicators = {
            "USD_index": {"min": 90, "max": 110, "current": 103.5},
            "inflation_rate": {"min": 1.0, "max": 8.0, "current": 3.2},
            "interest_rate": {"min": 0.0, "max": 6.0, "current": 5.25},
            "oil_price": {"min": 60, "max": 120, "current": 78.5},
            "VIX_volatility": {"min": 10, "max": 80, "current": 18.2},
            "bond_yield_10y": {"min": 1.0, "max": 5.0, "current": 4.3}
        }
        
        # Gold price parameters
        self.gold_base_price = 2000  # USD per ounce
        
        print("💰 Financial Data Simulator initialized")
        print(f"   Current gold base price: ${self.gold_base_price}/oz")
    
    def generate_historical_gold_prices(self, days=365):
        """Generate realistic historical gold price series"""
        
        prices = []
        current_price = self.gold_base_price
        
        # Generate daily prices with realistic volatility
        for day in range(days):
            # Market factors influence
            daily_volatility = random.uniform(-0.03, 0.03)  # ±3% daily volatility
            
            # Add some trend patterns
            if day < days * 0.3:  # First 30% - upward trend
                trend = random.uniform(-0.005, 0.015)
            elif day < days * 0.7:  # Middle 40% - sideways
                trend = random.uniform(-0.008, 0.008)
            else:  # Last 30% - mixed trend
                trend = random.uniform(-0.012, 0.010)
            
            # Calculate price change
            price_change = current_price * (daily_volatility + trend)
            current_price = max(current_price + price_change, 1500)  # Floor at $1500
            
            date = (datetime.now() - timedelta(days=days-day-1)).strftime("%Y-%m-%d")
            
            prices.append({
                "date": date,
                "price": round(current_price, 2),
                "change": round(price_change, 2),
                "change_percent": round((price_change / current_price) * 100, 3)
            })
        
        return prices
    
    def generate_current_market_indicators(self, market_scenario="neutral"):
        """Generate current market indicators based on scenario"""
        
        indicators = {}
        
        # Scenario-based adjustments
        scenario_multipliers = {
            "bullish": {"USD_index": 0.95, "inflation_rate": 1.2, "interest_rate": 0.9, "VIX_volatility": 0.8},
            "bearish": {"USD_index": 1.1, "inflation_rate": 0.8, "interest_rate": 1.15, "VIX_volatility": 1.5},
            "neutral": {"USD_index": 1.0, "inflation_rate": 1.0, "interest_rate": 1.0, "VIX_volatility": 1.0},
            "volatile": {"USD_index": 1.05, "inflation_rate": 1.3, "interest_rate": 0.85, "VIX_volatility": 2.0}
        }
        
        multipliers = scenario_multipliers.get(market_scenario, scenario_multipliers["neutral"])
        
        for indicator, params in self.market_indicators.items():
            base_value = params["current"]
            multiplier = multipliers.get(indicator, 1.0)
            noise = random.uniform(-0.1, 0.1)  # ±10% noise
            
            new_value = base_value * multiplier * (1 + noise)
            new_value = max(min(new_value, params["max"]), params["min"])  # Clamp to range
            
            indicators[indicator] = round(new_value, 2)
        
        # Add calculated indicators
        indicators["dollar_strength_index"] = self._calculate_dollar_strength(indicators)
        indicators["gold_sentiment_score"] = self._calculate_gold_sentiment(indicators)
        indicators["market_stress_level"] = self._calculate_market_stress(indicators)
        
        return indicators
    
    def _calculate_dollar_strength(self, indicators):
        """Calculate dollar strength composite index"""
        # Higher USD index + higher interest rates = stronger dollar = bearish for gold
        strength = (indicators["USD_index"] / 100) * (1 + indicators["interest_rate"] / 10)
        return round(strength, 3)
    
    def _calculate_gold_sentiment(self, indicators):
        """Calculate gold market sentiment score"""
        # Higher inflation + higher volatility + lower USD = bullish for gold
        sentiment = (
            indicators["inflation_rate"] / 5 +
            indicators["VIX_volatility"] / 50 +
            (110 - indicators["USD_index"]) / 20
        ) / 3
        return round(max(0, min(sentiment, 1)), 3)
    
    def _calculate_market_stress(self, indicators):
        """Calculate overall market stress level"""
        stress = (
            indicators["VIX_volatility"] / 80 +
            indicators["inflation_rate"] / 8 +
            indicators["interest_rate"] / 6
        ) / 3
        return round(max(0, min(stress, 1)), 3)

# ============================================================================
# 2. NEWS DATA SIMULATOR AND FAKE NEWS CLASSIFIER
# ============================================================================

class NewsAnalysisEngine:
    """Simulates financial news and provides fake news classification"""
    
    def __init__(self):
        # Real news templates for different market scenarios
        self.real_news_templates = {
            "bullish": [
                "Federal Reserve signals potential pause in interest rate hikes amid economic uncertainty",
                "Central banks increase gold reserves as inflation concerns persist globally",
                "Geopolitical tensions in Middle East drive safe-haven demand for precious metals",
                "Dollar weakens against major currencies following disappointing economic data",
                "Institutional investors increase gold allocation amid market volatility"
            ],
            "bearish": [
                "Federal Reserve maintains hawkish stance, signals continued rate increases",
                "Strong economic data reduces safe-haven demand for gold",
                "Dollar strengthens to multi-month highs against global currencies",
                "Tech sector rally diverts investment flows from precious metals",
                "Central bank officials express confidence in inflation control measures"
            ],
            "neutral": [
                "Gold prices consolidate in narrow range as markets await economic data",
                "Mixed economic signals create uncertainty in precious metals markets",
                "Analysts divided on gold price direction amid conflicting indicators",
                "Trading volumes remain moderate in gold futures markets",
                "Market participants await key economic announcements this week"
            ],    "volatile": [  # ADD THIS
        "Market volatility spikes as conflicting economic data creates uncertainty",
        "Gold prices swing wildly amid mixed signals from central banks",
        "Precious metals see increased trading volumes during market turbulence",
        "Investors seek clarity as gold experiences heightened price swings",
        "Market participants brace for continued volatility in precious metals"
    ]
        }
        
        # Fake news patterns (to be filtered out)
        self.fake_news_templates = [
            "BREAKING: Secret government memo reveals gold price manipulation scheme",
            "Exclusive: Insider trading scandal rocks precious metals markets",
            "Anonymous source claims major bank planning gold market crash",
            "Unverified report suggests hidden gold reserves discovered in [random country]",
            "Social media rumor spreads about cryptocurrency replacing gold standard"
        ]
        
        # News credibility indicators
        self.credible_sources = [
            "Reuters Financial", "Bloomberg Markets", "Wall Street Journal", 
            "Financial Times", "MarketWatch", "CNBC Markets", "Yahoo Finance"
        ]
        
        self.suspicious_sources = [
            "GoldTruthRevealed.com", "SecretFinanceNews", "MarketConspiracy.net",
            "CryptoGoldRevolution", "UndergroundTrading.org"
        ]
        
        print("📰 News Analysis Engine initialized")
        print(f"   Monitoring {len(self.credible_sources)} credible sources")
    
    def generate_news_articles(self, market_scenario="neutral", num_articles=10):
        """Generate realistic news articles for analysis"""
        
        articles = []
        
        # Generate mix of real and fake news
        real_count = int(num_articles * 0.8)  # 80% real news
        fake_count = num_articles - real_count
        
        # Generate real news
        for i in range(real_count):
            article = self._generate_real_article(market_scenario, i)
            articles.append(article)
        
        # Generate fake news
        for i in range(fake_count):
            article = self._generate_fake_article(i)
            articles.append(article)
        
        # Shuffle articles
        random.shuffle(articles)
        
        return articles
    
    def _generate_real_article(self, scenario, article_id):
        """Generate a real news article"""
        
        headline = random.choice(self.real_news_templates[scenario])
        source = random.choice(self.credible_sources)
        
        # Generate article content based on headline
        content = self._generate_article_content(headline, scenario, is_fake=False)
        
        article = {
            "id": f"real_{article_id}",
            "headline": headline,
            "source": source,
            "content": content,
            "timestamp": (datetime.now() - timedelta(hours=random.randint(1, 48))).strftime("%Y-%m-%d %H:%M:%S"),
            "credibility_score": random.uniform(0.8, 0.95),
            "is_fake": False,
            "market_impact": scenario
        }
        
        return article
    
    def _generate_fake_article(self, article_id):
        """Generate a fake news article"""
        
        headline = random.choice(self.fake_news_templates)
        # Replace placeholder with random country
        if "[random country]" in headline:
            countries = ["Bolivia", "Madagascar", "Kazakhstan", "Myanmar", "Estonia"]
            headline = headline.replace("[random country]", random.choice(countries))
        
        source = random.choice(self.suspicious_sources)
        content = self._generate_article_content(headline, "sensational", is_fake=True)
        
        article = {
            "id": f"fake_{article_id}",
            "headline": headline,
            "source": source,
            "content": content,
            "timestamp": (datetime.now() - timedelta(hours=random.randint(1, 12))).strftime("%Y-%m-%d %H:%M:%S"),
            "credibility_score": random.uniform(0.1, 0.4),
            "is_fake": True,
            "market_impact": "misleading"
        }
        
        return article
    
    def _generate_article_content(self, headline, scenario, is_fake=False):
        """Generate article content based on headline and scenario"""
        
        if is_fake:
            content_templates = [
                "According to anonymous sources within major financial institutions, [CLAIM]. "
                "This unprecedented development could dramatically impact precious metals markets. "
                "However, official sources have not yet confirmed these reports.",
                
                "Unverified documents obtained exclusively by our sources suggest [CLAIM]. "
                "Market analysts are reportedly concerned about the potential implications. "
                "The story is developing and more details are expected soon."
            ]
            
            claims = [
                "secret manipulation schemes targeting gold prices",
                "hidden reserves that could flood the market",
                "insider trading coordinated across multiple institutions"
            ]
            
            template = random.choice(content_templates)
            claim = random.choice(claims)
            content = template.replace("[CLAIM]", claim)
            
        else:
            # Generate realistic financial news content
            if "Federal Reserve" in headline:
                content = ("The Federal Reserve's latest policy statement has significant implications "
                          "for precious metals markets. Market participants are closely monitoring "
                          "monetary policy signals that could affect gold demand and pricing dynamics.")
            elif "inflation" in headline.lower():
                content = ("Rising inflation concerns continue to support safe-haven demand for gold. "
                          "Economic data suggests persistent price pressures that may influence "
                          "central bank policy and precious metals allocation strategies.")
            elif "geopolitical" in headline.lower():
                content = ("Heightened geopolitical tensions are driving increased demand for safe-haven "
                          "assets including gold. Investors are seeking portfolio protection amid "
                          "uncertainty in global markets and currency fluctuations.")
            else:
                content = ("Market dynamics in precious metals continue to evolve amid changing "
                          "economic conditions. Technical analysis and fundamental factors suggest "
                          "continued volatility in gold pricing and trading volumes.")
        
        return content
    
    def classify_news_authenticity(self, articles):
        """
        Algorithm Step 1: Classify each news article as real or fake
        """
        
        print("📰 Classifying news articles for authenticity...")
        
        real_articles = []
        fake_articles = []
        
        for article in articles:
            # Simulate fake news classifier
            is_real = self._fake_news_classifier(article)
            
            if is_real:
                real_articles.append(article)
            else:
                fake_articles.append(article)
        
        print(f"   ✅ Real articles: {len(real_articles)}")
        print(f"   ❌ Fake articles filtered: {len(fake_articles)}")
        
        return real_articles, fake_articles
    
    def _fake_news_classifier(self, article):
        """Simulated fake news classifier"""
        
        # Check source credibility
        source_credible = article["source"] in self.credible_sources
        
        # Check credibility score
        score_threshold = article["credibility_score"] > 0.6
        
        # Check for sensational language
        sensational_keywords = ["secret", "exclusive", "anonymous", "unverified", "breaking scandal"]
        has_sensational = any(keyword in article["headline"].lower() for keyword in sensational_keywords)
        
        # Classification logic
        if source_credible and score_threshold and not has_sensational:
            return True  # Real news
        elif article["credibility_score"] < 0.5 or has_sensational:
            return False  # Fake news
        else:
            # Borderline case - use credibility score
            return article["credibility_score"] > 0.7

# ============================================================================
# 3. LLM NEWS IMPACT ANALYSIS
# ============================================================================

class NewsImpactAnalyzer:
    """LLM-powered analysis of news impact on gold markets"""
    
    def __init__(self, llm_type="rule_based"):
        self.llm_type = llm_type
        self.model_available = False
        
        if llm_type == "transformers":
            self.setup_transformers()
        elif llm_type == "ollama":
            self.setup_ollama()
        else:
            # Default to financial expert system
            self.model_available = True
            print("💼 Using financial markets expert analysis system")
    
    def setup_transformers(self):
        """Setup Hugging Face Transformers"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            model_name = "microsoft/DialoGPT-medium"
            print(f"📥 Loading {model_name} for financial analysis...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model_available = True
            print("✅ Transformers model loaded successfully")
            
        except Exception as e:
            print(f"⚠️  Transformers error: {e}")
            print("   Using financial expert system")
    
    def setup_ollama(self):
        """Setup Ollama LLM"""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                
                preferred_models = ['llama2', 'mistral', 'phi']
                self.selected_model = None
                
                for model in preferred_models:
                    if any(model in m for m in available_models):
                        self.selected_model = model
                        break
                
                if self.selected_model:
                    self.model_available = True
                    print(f"✅ Using Ollama model: {self.selected_model}")
                else:
                    print("⚠️  No suitable Ollama models found")
        except:
            print("❌ Ollama not available")
    
    def analyze_news_impact(self, real_articles):
        """
        Algorithm Step 2: Use LLM to analyze real news and extract influencing factors
        """
        
        print("📊 Analyzing news impact on gold markets...")
        
        # Create comprehensive news analysis prompt
        prompt = self._create_news_analysis_prompt(real_articles)
        
        # Generate impact analysis
        if self.llm_type == "transformers" and self.model_available:
            analysis = self._transformers_analyze(prompt)
        elif self.llm_type == "ollama" and self.model_available:
            analysis = self._ollama_analyze(prompt)
        else:
            analysis = self._expert_financial_analysis(real_articles)
        
        # Extract structured factors
        impact_factors = self._extract_impact_factors(analysis, real_articles)
        
        return impact_factors
    
    def _create_news_analysis_prompt(self, articles):
        """Create structured prompt for news impact analysis"""
        
        # Format articles for analysis
        articles_text = ""
        for i, article in enumerate(articles, 1):
            articles_text += f"\nArticle {i}: {article['headline']}\n"
            articles_text += f"Source: {article['source']}\n"
            articles_text += f"Content: {article['content'][:200]}...\n"
        
        prompt = f"""You are a senior gold market analyst with 15 years of experience in precious metals trading and market analysis.

FINANCIAL NEWS ANALYSIS REQUEST:
Analyze the following {len(articles)} news articles for their impact on gold prices and investment decisions.

NEWS ARTICLES:
{articles_text}

ANALYSIS REQUIREMENTS:
Please provide a comprehensive analysis focusing on:

1. MARKET SENTIMENT IMPACT:
   - Overall sentiment direction (bullish/bearish/neutral)
   - Confidence level of sentiment assessment
   - Key sentiment drivers from the news

2. PRICE IMPACT FACTORS:
   - Short-term price pressures (next 1-7 days)
   - Medium-term trends (next 1-3 months)
   - Specific factors driving price expectations

3. RISK ASSESSMENT:
   - Market volatility indicators
   - Geopolitical risk factors
   - Economic uncertainty levels

4. INVESTMENT IMPLICATIONS:
   - Trading opportunities identified
   - Risk management considerations
   - Portfolio allocation suggestions

Please provide specific, actionable insights based on the news content."""

        return prompt
    
    def _transformers_analyze(self, prompt):
        """Analyze using Transformers"""
        try:
            simplified_prompt = "Financial news analysis for gold market impact and investment decisions:"
            
            inputs = self.tokenizer.encode(simplified_prompt, return_tensors="pt")
            
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + 200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(simplified_prompt):].strip()
            
            if len(response) < 50:
                return self._expert_financial_analysis([])
            
            return response
            
        except Exception as e:
            print(f"❌ Transformers error: {e}")
            return self._expert_financial_analysis([])
    
    def _ollama_analyze(self, prompt):
        """Analyze using Ollama"""
        try:
            payload = {
                "model": self.selected_model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post("http://localhost:11434/api/generate", json=payload)
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return self._expert_financial_analysis([])
                
        except Exception as e:
            print(f"❌ Ollama error: {e}")
            return self._expert_financial_analysis([])
    
    def _expert_financial_analysis(self, articles):
        """Expert financial analysis system"""
        
        print("💼 Using financial markets expert analysis")
        
        # Analyze articles for sentiment and impact
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        # Keywords for sentiment analysis
        bullish_keywords = ["inflation", "uncertainty", "tensions", "safe-haven", "demand", "reserves"]
        bearish_keywords = ["hawkish", "rate hikes", "strong economy", "dollar strengthens", "confidence"]
        
        risk_factors = []
        sentiment_drivers = []
        
        for article in articles:
            text = (article["headline"] + " " + article["content"]).lower()
            
            bullish_score = sum(1 for keyword in bullish_keywords if keyword in text)
            bearish_score = sum(1 for keyword in bearish_keywords if keyword in text)
            
            if bullish_score > bearish_score:
                bullish_count += 1
                sentiment_drivers.append("Inflation concerns and safe-haven demand")
            elif bearish_score > bullish_score:
                bearish_count += 1
                sentiment_drivers.append("Monetary policy tightening and dollar strength")
            else:
                neutral_count += 1
                sentiment_drivers.append("Mixed economic signals")
            
            # Extract risk factors
            if "geopolitical" in text or "tensions" in text:
                risk_factors.append("Geopolitical uncertainty")
            if "inflation" in text:
                risk_factors.append("Inflation volatility")
            if "rate" in text:
                risk_factors.append("Interest rate changes")
        
        # Determine overall sentiment
        if bullish_count > bearish_count:
            overall_sentiment = "bullish"
            confidence = round(bullish_count / len(articles), 2)
        elif bearish_count > bullish_count:
            overall_sentiment = "bearish"
            confidence = round(bearish_count / len(articles), 2)
        else:
            overall_sentiment = "neutral"
            confidence = 0.5
        
        # Generate analysis
        analysis = f"""
GOLD MARKET NEWS IMPACT ANALYSIS

MARKET SENTIMENT ASSESSMENT:
Overall Sentiment: {overall_sentiment.upper()}
Confidence Level: {confidence:.2f}/1.0
Articles Analyzed: {len(articles)}

Sentiment Breakdown:
• Bullish signals: {bullish_count} articles
• Bearish signals: {bearish_count} articles  
• Neutral signals: {neutral_count} articles

KEY SENTIMENT DRIVERS:
{chr(10).join(f"• {driver}" for driver in set(sentiment_drivers))}

PRICE IMPACT FACTORS:
Short-term (1-7 days):
• {"Upward pressure from safe-haven demand" if overall_sentiment == "bullish" else "Downward pressure from dollar strength" if overall_sentiment == "bearish" else "Sideways consolidation expected"}
• {"Increased volatility due to news flow" if len(risk_factors) > 2 else "Moderate volatility expected"}

Medium-term (1-3 months):
• {"Sustained inflation concerns support higher prices" if overall_sentiment == "bullish" else "Monetary policy headwinds limit upside" if overall_sentiment == "bearish" else "Range-bound trading likely"}
• {"Technical levels become more important" if overall_sentiment == "neutral" else "Fundamental factors driving direction"}

RISK ASSESSMENT:
Market Volatility: {"High" if len(risk_factors) > 3 else "Medium" if len(risk_factors) > 1 else "Low"}
Risk Factors Identified:
{chr(10).join(f"• {factor}" for factor in set(risk_factors)) if risk_factors else "• Minimal risk factors detected"}

Economic Uncertainty Level: {"Elevated" if len(risk_factors) > 2 else "Moderate"}

INVESTMENT IMPLICATIONS:
Trading Opportunities:
• {"Long gold positions favored on dips" if overall_sentiment == "bullish" else "Short-term trading opportunities on rallies" if overall_sentiment == "bearish" else "Range trading strategies applicable"}
• {"Options strategies may benefit from volatility" if len(risk_factors) > 2 else "Direct position taking preferred"}

Risk Management:
• {"Tight stops recommended due to volatility" if len(risk_factors) > 2 else "Standard risk management protocols"}
• {"Position sizing should reflect uncertainty level" if overall_sentiment == "neutral" else "Conviction trades appropriate"}

Portfolio Allocation:
• {"Increase gold allocation as portfolio hedge" if overall_sentiment == "bullish" else "Reduce gold exposure temporarily" if overall_sentiment == "bearish" else "Maintain current gold allocation"}
"""
        
        return analysis.strip()
    
    def _extract_impact_factors(self, analysis, articles):
        """Extract structured impact factors from analysis"""
        
        # Extract sentiment
        sentiment = "neutral"
        if "bullish" in analysis.lower():
            sentiment = "bullish"
        elif "bearish" in analysis.lower():
            sentiment = "bearish"
        
        # Extract confidence level
        confidence = 0.5
        try:
            if "confidence level:" in analysis.lower():
                conf_line = [line for line in analysis.split('\n') if 'confidence level:' in line.lower()][0]
                confidence = float(re.search(r'(\d+\.?\d*)', conf_line).group(1))
        except:
            pass
        
        # Extract risk factors
        risk_factors = []
        risk_keywords = ["geopolitical", "inflation", "interest rate", "volatility", "uncertainty"]
        for keyword in risk_keywords:
            if keyword in analysis.lower():
                risk_factors.append(keyword)
        
        impact_factors = {
            "overall_sentiment": sentiment,
            "sentiment_confidence": confidence,
            "news_count": len(articles),
            "risk_factors": risk_factors,
            "volatility_expectation": "high" if len(risk_factors) > 2 else "medium",
            "time_horizon_impact": {
                "short_term": "bullish" if sentiment == "bullish" else "bearish" if sentiment == "bearish" else "neutral",
                "medium_term": sentiment
            },
            "investment_bias": sentiment,
            "news_analysis": analysis
        }
        
        return impact_factors

# ============================================================================
# 4. PRICE PREDICTION ENGINE
# ============================================================================

class GoldPricePredictionEngine:
    """ML-based price prediction for gold markets"""
    
    def __init__(self):
        # Technical indicators for price prediction
        self.technical_indicators = [
            "moving_average_20", "moving_average_50", "RSI", "MACD", 
            "bollinger_upper", "bollinger_lower", "volume_trend"
        ]
        
        print("📈 Gold Price Prediction Engine initialized")
    
    def predict_gold_prices(self, historical_prices, market_indicators, news_factors):
        """
        Algorithm Steps 3-4: Feature aggregation and price forecasting
        """
        
        print("🔮 Predicting short-term and long-term gold prices...")
        
        # Step 3: Feature Aggregation
        features = self._aggregate_features(historical_prices, market_indicators, news_factors)
        
        # Step 4: Price Forecasting
        short_term_price, long_term_price = self._forecast_prices(features, historical_prices)
        
        predictions = {
            "current_price": historical_prices[-1]["price"],
            "short_term_prediction": short_term_price,
            "long_term_prediction": long_term_price,
            "short_term_change": round(short_term_price - historical_prices[-1]["price"], 2),
            "long_term_change": round(long_term_price - historical_prices[-1]["price"], 2),
            "short_term_change_percent": round(((short_term_price - historical_prices[-1]["price"]) / historical_prices[-1]["price"]) * 100, 2),
            "long_term_change_percent": round(((long_term_price - historical_prices[-1]["price"]) / historical_prices[-1]["price"]) * 100, 2),
            "prediction_confidence": self._calculate_prediction_confidence(features),
            "technical_indicators": self._calculate_technical_indicators(historical_prices),
            "aggregated_features": features
        }
        
        return predictions
    
    def _aggregate_features(self, historical_prices, market_indicators, news_factors):
        """Algorithm Step 3: Aggregate all features for ML prediction"""
        
        # Price-based features
        prices = [p["price"] for p in historical_prices[-30:]]  # Last 30 days
        price_features = {
            "price_trend_30d": (prices[-1] - prices[0]) / prices[0],
            "price_volatility": np.std(prices) / np.mean(prices),
            "price_momentum": (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0,
            "price_sma_20": np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices),
            "price_sma_50": np.mean(prices) if len(prices) >= 30 else np.mean(prices)
        }
        
        # Market indicator features
        market_features = {
            "usd_strength": market_indicators.get("dollar_strength_index", 1.0),
            "inflation_pressure": market_indicators.get("inflation_rate", 3.0) / 10,
            "interest_rate_level": market_indicators.get("interest_rate", 5.0) / 10,
            "market_stress": market_indicators.get("market_stress_level", 0.5),
            "gold_sentiment": market_indicators.get("gold_sentiment_score", 0.5),
            "vix_volatility": market_indicators.get("VIX_volatility", 20.0) / 100,
            "oil_price_impact": market_indicators.get("oil_price", 80.0) / 100,
            "bond_yield_pressure": market_indicators.get("bond_yield_10y", 4.0) / 10
        }
        
        # News sentiment features
        news_features = {
            "news_sentiment": 1.0 if news_factors["overall_sentiment"] == "bullish" else -1.0 if news_factors["overall_sentiment"] == "bearish" else 0.0,
            "news_confidence": news_factors.get("sentiment_confidence", 0.5),
            "news_volume": news_factors.get("news_count", 5) / 10,
            "risk_factor_count": len(news_factors.get("risk_factors", [])) / 5,
            "volatility_expectation": 1.0 if news_factors.get("volatility_expectation") == "high" else 0.5
        }
        
        # Technical analysis features
        technical_features = self._calculate_technical_features(historical_prices)
        
        # Combine all features
        aggregated_features = {
            **price_features,
            **market_features,
            **news_features,
            **technical_features
        }
        
        return aggregated_features
    
    def _calculate_technical_features(self, historical_prices):
        """Calculate technical analysis indicators"""
        
        prices = [p["price"] for p in historical_prices[-50:]]  # Last 50 days for technical analysis
        
        if len(prices) < 14:
            return {
                "rsi": 50.0,
                "macd_signal": 0.0,
                "bollinger_position": 0.5,
                "price_above_sma20": 0.5,
                "price_above_sma50": 0.5
            }
        
        # RSI calculation
        rsi = self._calculate_rsi(prices)
        
        # MACD calculation
        macd_signal = self._calculate_macd_signal(prices)
        
        # Bollinger Bands position
        bollinger_position = self._calculate_bollinger_position(prices)
        
        # Moving average positions
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
        sma_50 = np.mean(prices) if len(prices) >= 50 else np.mean(prices)
        current_price = prices[-1]
        
        return {
            "rsi": rsi,
            "macd_signal": macd_signal,
            "bollinger_position": bollinger_position,
            "price_above_sma20": 1.0 if current_price > sma_20 else 0.0,
            "price_above_sma50": 1.0 if current_price > sma_50 else 0.0
        }
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return round(rsi, 2)
    
    def _calculate_macd_signal(self, prices):
        """Calculate MACD signal"""
        if len(prices) < 26:
            return 0.0
        
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        macd_line = ema_12 - ema_26
        
        return round(macd_line / prices[-1] * 100, 3)  # Normalized MACD
    
    def _calculate_ema(self, prices, period):
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_bollinger_position(self, prices, period=20):
        """Calculate position within Bollinger Bands"""
        if len(prices) < period:
            return 0.5
        
        recent_prices = prices[-period:]
        sma = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        current_price = prices[-1]
        
        if upper_band == lower_band:
            return 0.5
        
        position = (current_price - lower_band) / (upper_band - lower_band)
        return round(max(0, min(1, position)), 3)
    
    def _forecast_prices(self, features, historical_prices):
        """Algorithm Step 4: Forecast short-term and long-term prices"""
        
        current_price = historical_prices[-1]["price"]
        
        # Weighted scoring system for price prediction
        short_term_multiplier = self._calculate_short_term_multiplier(features)
        long_term_multiplier = self._calculate_long_term_multiplier(features)
        
        # Apply price forecasting with volatility adjustment
        base_volatility = features.get("price_volatility", 0.02)
        
        short_term_price = current_price * short_term_multiplier
        long_term_price = current_price * long_term_multiplier
        
        # Add some realistic bounds
        short_term_price = max(current_price * 0.85, min(current_price * 1.15, short_term_price))
        long_term_price = max(current_price * 0.70, min(current_price * 1.40, long_term_price))
        
        return round(short_term_price, 2), round(long_term_price, 2)
    
    def _calculate_short_term_multiplier(self, features):
        """Calculate short-term price multiplier based on features"""
        
        # News sentiment impact (40% weight)
        news_impact = features.get("news_sentiment", 0.0) * 0.02  # ±2% max
        news_confidence = features.get("news_confidence", 0.5)
        weighted_news = news_impact * news_confidence
        
        # Technical indicators (30% weight)
        rsi_impact = (features.get("rsi", 50) - 50) / 1000  # RSI normalization
        macd_impact = features.get("macd_signal", 0.0) / 1000
        technical_impact = (rsi_impact + macd_impact) * 0.5
        
        # Market stress (20% weight)
        stress_impact = -features.get("market_stress", 0.5) * 0.01  # Negative correlation
        
        # USD strength (10% weight)
        usd_impact = -(features.get("usd_strength", 1.0) - 1.0) * 0.02
        
        total_impact = weighted_news * 0.4 + technical_impact * 0.3 + stress_impact * 0.2 + usd_impact * 0.1
        multiplier = 1.0 + total_impact
        
        return max(0.95, min(1.05, multiplier))  # Cap at ±5%
    
    def _calculate_long_term_multiplier(self, features):
        """Calculate long-term price multiplier based on features"""
        
        # Fundamental factors have more weight in long-term
        
        # Inflation impact (35% weight)
        inflation_impact = (features.get("inflation_pressure", 0.3) - 0.3) * 0.1
        
        # Interest rate impact (25% weight) - negative correlation
        rate_impact = -(features.get("interest_rate_level", 0.5) - 0.5) * 0.08
        
        # News sentiment (20% weight)
        news_impact = features.get("news_sentiment", 0.0) * 0.03
        news_confidence = features.get("news_confidence", 0.5)
        weighted_news = news_impact * news_confidence
        
        # Market stress (10% weight)
        stress_impact = features.get("market_stress", 0.5) * 0.02
        
        # Price momentum (10% weight)
        momentum_impact = features.get("price_momentum", 0.0) * 0.5
        
        total_impact = (inflation_impact * 0.35 + rate_impact * 0.25 + 
                       weighted_news * 0.20 + stress_impact * 0.10 + momentum_impact * 0.10)
        
        multiplier = 1.0 + total_impact
        
        return max(0.85, min(1.25, multiplier))  # Cap at ±15%
    
    def _calculate_prediction_confidence(self, features):
        """Calculate confidence level in predictions"""
        
        # Higher confidence with more consistent signals
        news_conf = features.get("news_confidence", 0.5)
        
        # Technical consistency
        rsi = features.get("rsi", 50)
        rsi_strength = abs(rsi - 50) / 50  # Distance from neutral
        
        # Market clarity (inverse of stress)
        market_clarity = 1.0 - features.get("market_stress", 0.5)
        
        # News volume adequacy
        news_volume = min(features.get("news_volume", 0.5), 1.0)
        
        confidence = (news_conf * 0.4 + rsi_strength * 0.3 + 
                     market_clarity * 0.2 + news_volume * 0.1)
        
        return round(min(confidence, 0.95), 3)  # Cap at 95%
    
    def _calculate_technical_indicators(self, historical_prices):
        """Calculate current technical indicators for display"""
        
        prices = [p["price"] for p in historical_prices[-50:]]
        
        if len(prices) < 20:
            return {
                "SMA_20": prices[-1] if prices else 2000,
                "SMA_50": prices[-1] if prices else 2000,
                "RSI": 50.0,
                "MACD": 0.0,
                "Bollinger_Upper": prices[-1] * 1.02 if prices else 2040,
                "Bollinger_Lower": prices[-1] * 0.98 if prices else 1960
            }
        
        current_price = prices[-1]
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices) if len(prices) >= 50 else np.mean(prices)
        
        rsi = self._calculate_rsi(prices)
        macd = self._calculate_macd_signal(prices)
        
        # Bollinger Bands
        std_20 = np.std(prices[-20:])
        bollinger_upper = sma_20 + (2 * std_20)
        bollinger_lower = sma_20 - (2 * std_20)
        
        return {
            "SMA_20": round(sma_20, 2),
            "SMA_50": round(sma_50, 2),
            "RSI": rsi,
            "MACD": macd,
            "Bollinger_Upper": round(bollinger_upper, 2),
            "Bollinger_Lower": round(bollinger_lower, 2)
        }

# ============================================================================
# 5. INVESTMENT RECOMMENDATION ENGINE
# ============================================================================

class InvestmentRecommendationEngine:
    """LLM-powered investment recommendation system"""
    
    def __init__(self, llm_type="rule_based"):
        self.llm_type = llm_type
        self.model_available = False
        
        if llm_type == "transformers":
            self.setup_transformers()
        elif llm_type == "ollama":
            self.setup_ollama()
        else:
            self.model_available = True
            print("💡 Using expert investment recommendation system")
    
    def setup_transformers(self):
        """Setup Hugging Face Transformers for recommendations"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            model_name = "microsoft/DialoGPT-medium"
            print(f"📥 Loading {model_name} for investment recommendations...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model_available = True
            print("✅ Transformers recommendation model loaded")
            
        except Exception as e:
            print(f"⚠️  Transformers error: {e}")
            print("   Using expert recommendation system")
    
    def setup_ollama(self):
        """Setup Ollama LLM for recommendations"""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                
                preferred_models = ['llama2', 'mistral', 'phi']
                self.selected_model = None
                
                for model in preferred_models:
                    if any(model in m for m in available_models):
                        self.selected_model = model
                        break
                
                if self.selected_model:
                    self.model_available = True
                    print(f"✅ Using Ollama model for recommendations: {self.selected_model}")
                else:
                    print("⚠️  No suitable Ollama models found")
        except:
            print("❌ Ollama not available for recommendations")
    
    def generate_investment_recommendation(self, news_factors, price_predictions, market_indicators):
        """
        Algorithm Step 5: Generate comprehensive investment recommendation
        """
        
        print("💰 Generating investment recommendation...")
        
        # Create structured prompt
        prompt = self._format_recommendation_prompt(news_factors, price_predictions, market_indicators)
        
        # Generate recommendation using selected method
        if self.llm_type == "transformers" and self.model_available:
            recommendation = self._transformers_recommend(prompt)
        elif self.llm_type == "ollama" and self.model_available:
            recommendation = self._ollama_recommend(prompt)
        else:
            recommendation = self._expert_investment_analysis(news_factors, price_predictions, market_indicators)
        
        # Structure the recommendation
        structured_recommendation = self._structure_recommendation(
            recommendation, news_factors, price_predictions, market_indicators
        )
        
        return structured_recommendation
    
    def _format_recommendation_prompt(self, news_factors, price_predictions, market_indicators):
        """Format comprehensive prompt for investment recommendation"""
        
        prompt = f"""You are a senior portfolio manager and gold trading expert with 20 years of experience in precious metals investment and risk management.

INVESTMENT ANALYSIS REQUEST:
Based on the comprehensive market analysis below, provide specific investment recommendations for gold trading.

=== MARKET ANALYSIS SUMMARY ===

NEWS SENTIMENT ANALYSIS:
• Overall sentiment: {news_factors['overall_sentiment'].upper()}
• Confidence level: {news_factors['sentiment_confidence']:.2f}/1.0
• Risk factors: {', '.join(news_factors.get('risk_factors', ['None identified']))}
• Volatility expectation: {news_factors.get('volatility_expectation', 'medium')}

PRICE PREDICTIONS:
• Current price: ${price_predictions['current_price']}/oz
• Short-term forecast (1-7 days): ${price_predictions['short_term_prediction']}/oz ({price_predictions['short_term_change']:+.2f}, {price_predictions['short_term_change_percent']:+.1f}%)
• Long-term forecast (1-3 months): ${price_predictions['long_term_prediction']}/oz ({price_predictions['long_term_change']:+.2f}, {price_predictions['long_term_change_percent']:+.1f}%)
• Prediction confidence: {price_predictions['prediction_confidence']:.1%}

MARKET INDICATORS:
• USD Strength Index: {market_indicators.get('dollar_strength_index', 'N/A')}
• Inflation Rate: {market_indicators.get('inflation_rate', 'N/A')}%
• Interest Rate: {market_indicators.get('interest_rate', 'N/A')}%
• Market Stress Level: {market_indicators.get('market_stress_level', 'N/A')}
• VIX Volatility: {market_indicators.get('VIX_volatility', 'N/A')}

TECHNICAL INDICATORS:
• RSI: {price_predictions['technical_indicators'].get('RSI', 'N/A')}
• SMA 20: ${price_predictions['technical_indicators'].get('SMA_20', 'N/A')}/oz
• SMA 50: ${price_predictions['technical_indicators'].get('SMA_50', 'N/A')}/oz

=== RECOMMENDATION REQUIREMENTS ===

Please provide a comprehensive investment recommendation including:

1. INVESTMENT DECISION: Clear buy/sell/hold recommendation with rationale
2. POSITION SIZE: Suggested allocation (% of portfolio)
3. ENTRY STRATEGY: Optimal entry points and timing
4. RISK MANAGEMENT: Stop-loss levels and risk mitigation
5. TIME HORIZON: Recommended holding period
6. CONFIDENCE LEVEL: Your confidence in this recommendation (1-10)
7. ALTERNATIVE SCENARIOS: What could change the recommendation

Provide specific, actionable guidance suitable for both institutional and retail investors."""

        return prompt
    
    def _transformers_recommend(self, prompt):
        """Generate recommendation using Transformers"""
        try:
            simplified_prompt = f"Gold investment recommendation based on market analysis: {prompt[:200]}..."
            
            inputs = self.tokenizer.encode(simplified_prompt, return_tensors="pt")
            
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + 300,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(simplified_prompt):].strip()
            
            if len(response) < 100:
                return self._expert_investment_analysis({}, {}, {})
            
            return response
            
        except Exception as e:
            print(f"❌ Transformers recommendation error: {e}")
            return self._expert_investment_analysis({}, {}, {})
    
    def _ollama_recommend(self, prompt):
        """Generate recommendation using Ollama"""
        try:
            payload = {
                "model": self.selected_model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post("http://localhost:11434/api/generate", json=payload)
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return self._expert_investment_analysis({}, {}, {})
                
        except Exception as e:
            print(f"❌ Ollama recommendation error: {e}")
            return self._expert_investment_analysis({}, {}, {})
    
    def _expert_investment_analysis(self, news_factors, price_predictions, market_indicators):
        """Expert investment recommendation system"""
        
        print("💡 Using expert investment recommendation system")
        
        # Determine primary recommendation
        recommendation_score = self._calculate_recommendation_score(news_factors, price_predictions, market_indicators)
        
        if recommendation_score > 0.6:
            primary_action = "BUY"
            action_strength = "STRONG" if recommendation_score > 0.8 else "MODERATE"
        elif recommendation_score < -0.6:
            primary_action = "SELL"
            action_strength = "STRONG" if recommendation_score < -0.8 else "MODERATE"
        else:
            primary_action = "HOLD"
            action_strength = "NEUTRAL"
        
        # Calculate position sizing
        position_size = self._calculate_position_size(recommendation_score, news_factors, price_predictions)
        
        # Determine entry strategy
        entry_strategy = self._determine_entry_strategy(primary_action, price_predictions, market_indicators)
        
        # Risk management
        risk_management = self._calculate_risk_management(primary_action, price_predictions, news_factors)
        
        # Time horizon
        time_horizon = self._determine_time_horizon(news_factors, price_predictions)
        
        # Confidence level
        confidence = self._calculate_recommendation_confidence(news_factors, price_predictions, market_indicators)
        
        # Generate comprehensive recommendation text
        recommendation = f"""
GOLD INVESTMENT RECOMMENDATION

=== PRIMARY RECOMMENDATION ===
Action: {action_strength} {primary_action}
Rationale: {self._get_recommendation_rationale(primary_action, recommendation_score, news_factors, price_predictions)}

=== POSITION SIZING ===
Suggested Allocation: {position_size}% of portfolio
Risk Justification: {self._get_position_rationale(position_size, recommendation_score)}

=== ENTRY STRATEGY ===
{entry_strategy}

=== RISK MANAGEMENT ===
{risk_management}

=== TIME HORIZON ===
Recommended Holding Period: {time_horizon}
Review Points: Weekly price action and monthly fundamental review

=== CONFIDENCE ASSESSMENT ===
Recommendation Confidence: {confidence}/10
Confidence Factors: {self._get_confidence_factors(news_factors, price_predictions)}

=== SCENARIO ANALYSIS ===
Bull Case: {self._get_bull_case(news_factors, price_predictions)}
Bear Case: {self._get_bear_case(news_factors, price_predictions)}
Base Case: Current recommendation represents most likely scenario

=== MONITORING TRIGGERS ===
Reassess if:
• Gold moves ±5% from current levels
• Major Fed policy announcements
• Significant geopolitical developments
• USD index moves beyond 100-108 range
• VIX spikes above 30 or falls below 15

=== EXECUTION RECOMMENDATIONS ===
For Retail Investors:
• Consider ETFs (GLD, IAU) for easy exposure
• Use dollar-cost averaging for large positions
• Maintain 5-15% gold allocation maximum

For Institutional Investors:
• Consider futures for leverage and hedging
• Use options strategies in high volatility periods
• Coordinate with existing commodity exposures
"""
        
        return recommendation.strip()
    
    def _calculate_recommendation_score(self, news_factors, price_predictions, market_indicators):
        """Calculate overall recommendation score (-1 to +1)"""
        
        # News sentiment (40% weight)
        news_score = 0
        if news_factors.get('overall_sentiment') == 'bullish':
            news_score = news_factors.get('sentiment_confidence', 0.5)
        elif news_factors.get('overall_sentiment') == 'bearish':
            news_score = -news_factors.get('sentiment_confidence', 0.5)
        
        # Price prediction (35% weight)
        short_term_change = price_predictions.get('short_term_change_percent', 0) / 100
        long_term_change = price_predictions.get('long_term_change_percent', 0) / 100
        price_score = (short_term_change * 0.4 + long_term_change * 0.6)
        
        # Market conditions (25% weight)
        market_stress = market_indicators.get('market_stress_level', 0.5)
        usd_strength = market_indicators.get('dollar_strength_index', 1.0)
        inflation = market_indicators.get('inflation_rate', 3.0)
        
        # Market score favors gold in stressed/inflationary environments
        market_score = (market_stress + (inflation - 2) / 6 - (usd_strength - 1.0)) / 3
        
        total_score = news_score * 0.4 + price_score * 0.35 + market_score * 0.25
        
        return max(-1.0, min(1.0, total_score))
    
    def _calculate_position_size(self, recommendation_score, news_factors, price_predictions):
        """Calculate suggested position size"""
        
        base_size = abs(recommendation_score) * 15  # Base: 0-15%
        
        # Adjust for volatility
        volatility = news_factors.get('volatility_expectation', 'medium')
        if volatility == 'high':
            base_size *= 0.7  # Reduce position in high volatility
        elif volatility == 'low':
            base_size *= 1.2  # Increase position in low volatility
        
        # Adjust for prediction confidence
        confidence = price_predictions.get('prediction_confidence', 0.5)
        base_size *= (0.5 + confidence)
        
        return round(max(2, min(20, base_size)), 1)
    
    def _determine_entry_strategy(self, action, price_predictions, market_indicators):
        """Determine optimal entry strategy"""
        
        current_price = price_predictions.get('current_price', 2000)
        short_term_price = price_predictions.get('short_term_prediction', 2000)
        
        if action == "BUY":
            if short_term_price > current_price:
                return f"Entry Strategy: Immediate entry at market price (${current_price}/oz) or on any dips below ${current_price * 0.995:.2f}/oz. Use dollar-cost averaging over 3-5 trading days to reduce timing risk."
            else:
                return f"Entry Strategy: Wait for pullback to ${short_term_price:.2f}/oz area before entering. Set limit orders between ${short_term_price * 0.98:.2f}-${short_term_price * 1.01:.2f}/oz."
        
        elif action == "SELL":
            if short_term_price < current_price:
                return f"Entry Strategy: Immediate exit at market price (${current_price}/oz) or on any rallies above ${current_price * 1.005:.2f}/oz. Consider staged exit over 2-3 days."
            else:
                return f"Entry Strategy: Wait for rally to ${short_term_price:.2f}/oz area before selling. Set limit orders between ${short_term_price * 0.99:.2f}-${short_term_price * 1.02:.2f}/oz."
        
        else:  # HOLD
            return f"Entry Strategy: Maintain current positions. Consider rebalancing if allocation drifts significantly from target. Monitor ${current_price * 0.97:.2f}/oz support and ${current_price * 1.03:.2f}/oz resistance levels."
    
    def _calculate_risk_management(self, action, price_predictions, news_factors):
        """Calculate risk management parameters"""
        
        current_price = price_predictions.get('current_price', 2000)
        volatility = news_factors.get('volatility_expectation', 'medium')
        
        if action == "BUY":
            if volatility == 'high':
                stop_loss = current_price * 0.95  # 5% stop loss
            else:
                stop_loss = current_price * 0.92  # 8% stop loss
            
            return f"Risk Management: Set stop-loss at ${stop_loss:.2f}/oz ({((stop_loss/current_price - 1) * 100):+.1f}%). Consider taking partial profits if price rises 10-15%. Maximum risk per trade: 2% of portfolio."
        
        elif action == "SELL":
            if volatility == 'high':
                stop_loss = current_price * 1.05  # 5% stop loss for short
            else:
                stop_loss = current_price * 1.08  # 8% stop loss for short
            
            return f"Risk Management: Set stop-loss at ${stop_loss:.2f}/oz ({((stop_loss/current_price - 1) * 100):+.1f}%). Cover position if price breaks above resistance. Maximum risk per trade: 2% of portfolio."
        
        else:  # HOLD
            return f"Risk Management: Monitor positions closely. Consider reducing exposure if gold breaks below ${current_price * 0.90:.2f}/oz or above ${current_price * 1.10:.2f}/oz. Maintain diversification across asset classes."
    
    def _determine_time_horizon(self, news_factors, price_predictions):
        """Determine recommended time horizon"""
        
        volatility = news_factors.get('volatility_expectation', 'medium')
        confidence = price_predictions.get('prediction_confidence', 0.5)
        
        if volatility == 'high' or confidence < 0.6:
            return "1-2 weeks (short-term tactical trade)"
        elif confidence > 0.8:
            return "1-3 months (high-confidence medium-term position)"
        else:
            return "3-6 weeks (standard medium-term trade)"
    
    def _calculate_recommendation_confidence(self, news_factors, price_predictions, market_indicators):
        """Calculate confidence in recommendation (1-10 scale)"""
        
        # Base confidence from predictions
        pred_confidence = price_predictions.get('prediction_confidence', 0.5)
        
        # News confidence
        news_confidence = news_factors.get('sentiment_confidence', 0.5)
        
        # Market clarity (inverse of stress)
        market_clarity = 1.0 - market_indicators.get('market_stress_level', 0.5)
        
        # Combined confidence
        combined = (pred_confidence * 0.4 + news_confidence * 0.4 + market_clarity * 0.2)
        
        return round(combined * 10, 1)
    
    def _get_recommendation_rationale(self, action, score, news_factors, price_predictions):
        """Get rationale for recommendation"""
        
        sentiment = news_factors.get('overall_sentiment', 'neutral')
        short_change = price_predictions.get('short_term_change_percent', 0)
        
        if action == "BUY":
            return f"Market sentiment is {sentiment} with {abs(short_change):.1f}% expected short-term price appreciation. Safe-haven demand and technical factors support upward momentum."
        elif action == "SELL":
            return f"Market sentiment is {sentiment} with {abs(short_change):.1f}% expected short-term price decline. Risk-off conditions and dollar strength present headwinds."
        else:
            return f"Mixed signals with {sentiment} sentiment. Price consolidation expected with {abs(short_change):.1f}% near-term movement. Wait for clearer directional signals."
    
    def _get_position_rationale(self, position_size, score):
        """Get rationale for position sizing"""
        
        if position_size > 12:
            return "High conviction trade with strong technical and fundamental alignment"
        elif position_size > 8:
            return "Moderate conviction with multiple supporting factors"
        elif position_size > 5:
            return "Conservative position due to mixed signals or higher uncertainty"
        else:
            return "Minimal exposure recommended due to unclear market direction"
    
    def _get_confidence_factors(self, news_factors, price_predictions):
        """Get factors affecting confidence"""
        
        factors = []
        
        if news_factors.get('sentiment_confidence', 0) > 0.7:
            factors.append("Strong news sentiment consensus")
        
        if price_predictions.get('prediction_confidence', 0) > 0.7:
            factors.append("High technical indicator alignment")
        
        if len(news_factors.get('risk_factors', [])) <= 2:
            factors.append("Limited risk factors identified")
        
        if not factors:
            factors.append("Moderate confidence due to mixed signals")
        
        return ", ".join(factors)
    
    def _get_bull_case(self, news_factors, price_predictions):
        """Generate bull case scenario"""
        
        return f"Inflation concerns intensify, driving safe-haven demand. Fed dovish pivot supports precious metals. Target: ${price_predictions.get('long_term_prediction', 2000) * 1.1:.0f}/oz (+{((price_predictions.get('long_term_prediction', 2000) * 1.1 / price_predictions.get('current_price', 2000)) - 1) * 100:.0f}%)"
    
    def _get_bear_case(self, news_factors, price_predictions):
        """Generate bear case scenario"""
        
        return f"Fed maintains hawkish stance, dollar strengthens significantly. Economic growth reduces safe-haven demand. Target: ${price_predictions.get('long_term_prediction', 2000) * 0.9:.0f}/oz ({((price_predictions.get('long_term_prediction', 2000) * 0.9 / price_predictions.get('current_price', 2000)) - 1) * 100:.0f}%)"
    
    def _structure_recommendation(self, recommendation_text, news_factors, price_predictions, market_indicators):
        """Structure the recommendation into a comprehensive format"""
        
        # Extract key components
        recommendation_score = self._calculate_recommendation_score(news_factors, price_predictions, market_indicators)
        
        if recommendation_score > 0.6:
            primary_action = "BUY"
            action_strength = "STRONG" if recommendation_score > 0.8 else "MODERATE"
        elif recommendation_score < -0.6:
            primary_action = "SELL"
            action_strength = "STRONG" if recommendation_score < -0.8 else "MODERATE"
        else:
            primary_action = "HOLD"
            action_strength = "NEUTRAL"
        
        structured = {
            "recommendation_summary": {
                "primary_action": primary_action,
                "action_strength": action_strength,
                "confidence_score": self._calculate_recommendation_confidence(news_factors, price_predictions, market_indicators),
                "recommendation_score": round(recommendation_score, 3)
            },
            "position_details": {
                "suggested_allocation": f"{self._calculate_position_size(recommendation_score, news_factors, price_predictions)}%",
                "time_horizon": self._determine_time_horizon(news_factors, price_predictions),
                "entry_price_target": price_predictions.get('current_price', 2000)
            },
            "risk_management": {
                "stop_loss_level": self._calculate_stop_loss(primary_action, price_predictions),
                "take_profit_level": self._calculate_take_profit(primary_action, price_predictions),
                "max_risk_per_trade": "2% of portfolio"
            },
            "market_analysis": {
                "news_sentiment": news_factors.get('overall_sentiment', 'neutral'),
                "sentiment_confidence": news_factors.get('sentiment_confidence', 0.5),
                "price_targets": {
                    "short_term": price_predictions.get('short_term_prediction', 2000),
                    "long_term": price_predictions.get('long_term_prediction', 2000)
                },
                "key_risk_factors": news_factors.get('risk_factors', [])
            },
            "detailed_analysis": recommendation_text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "next_review_date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        }
        
        return structured
    
    def _calculate_stop_loss(self, action, price_predictions):
        """Calculate stop loss level"""
        current_price = price_predictions.get('current_price', 2000)
        
        if action == "BUY":
            return round(current_price * 0.95, 2)  # 5% stop loss
        elif action == "SELL":
            return round(current_price * 1.05, 2)  # 5% stop loss for short
        else:
            return round(current_price * 0.90, 2)  # 10% for hold positions

    def _calculate_take_profit(self, action, price_predictions):
        """Calculate take profit level"""
        current_price = price_predictions.get('current_price', 2000)
        
        if action == "BUY":
            return round(current_price * 1.10, 2)  # 10% take profit
        elif action == "SELL":
            return round(current_price * 0.90, 2)  # 10% take profit for short
        else:
            return round(current_price * 1.05, 2)  # 5% for hold positions

# ============================================================================
# 6. MAIN ALGORITHM ORCHESTRATOR
# ============================================================================

class GoldTradingAlgorithm:
    """Main orchestrator for the complete gold trading algorithm"""
    
    def __init__(self, llm_type="rule_based"):
        print("🚀 Initializing LLM-Assisted Gold Trading Algorithm...")
        print("=" * 60)
        
        # Initialize all components
        self.data_simulator = FinancialDataSimulator()
        self.news_engine = NewsAnalysisEngine()
        self.news_analyzer = NewsImpactAnalyzer(llm_type)
        self.price_predictor = GoldPricePredictionEngine()
        self.recommendation_engine = InvestmentRecommendationEngine(llm_type)
        
        print("=" * 60)
        print("✅ All components initialized successfully!")
    
    def run_complete_analysis(self, market_scenario="neutral", num_articles=12, days_history=90):
        """
        Execute the complete 5-step algorithm for gold trading analysis
        """
        
        print(f"\n🔄 Starting complete gold market analysis...")
        print(f"   Market scenario: {market_scenario}")
        print(f"   News articles: {num_articles}")
        print(f"   Historical data: {days_history} days")
        print("=" * 60)
        
        # Generate market data
        print("\n📊 STEP 0: Generating Market Data")
        historical_prices = self.data_simulator.generate_historical_gold_prices(days_history)
        market_indicators = self.data_simulator.generate_current_market_indicators(market_scenario)
        news_articles = self.news_engine.generate_news_articles(market_scenario, num_articles)
        
        print(f"   Generated {len(historical_prices)} price points")
        print(f"   Generated {len(market_indicators)} market indicators")
        print(f"   Generated {len(news_articles)} news articles")
        
        # STEP 1: News Validation
        print("\n📰 STEP 1: News Validation")
        real_articles, fake_articles = self.news_engine.classify_news_authenticity(news_articles)
        
        # STEP 2: News Impact Analysis
        print("\n📊 STEP 2: News Impact Analysis")
        news_factors = self.news_analyzer.analyze_news_impact(real_articles)
        
        # STEPS 3-4: Feature Aggregation and Price Forecasting
        print("\n🔮 STEPS 3-4: Feature Aggregation and Price Forecasting")
        price_predictions = self.price_predictor.predict_gold_prices(
            historical_prices, market_indicators, news_factors
        )
        
        # STEP 5: Investment Feasibility via LLM
        print("\n💰 STEP 5: Investment Recommendation Generation")
        investment_recommendation = self.recommendation_engine.generate_investment_recommendation(
            news_factors, price_predictions, market_indicators
        )
        
        # Compile complete results
        complete_analysis = {
            "analysis_metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "market_scenario": market_scenario,
                "analysis_type": "Complete 5-Step Algorithm",
                "total_news_articles": len(news_articles),
                "real_articles_used": len(real_articles),
                "fake_articles_filtered": len(fake_articles)
            },
            "market_data": {
                "current_price": historical_prices[-1]["price"],
                "price_change_24h": historical_prices[-1]["change"],
                "market_indicators": market_indicators,
                "historical_data_points": len(historical_prices)
            },
            "news_analysis": news_factors,
            "price_predictions": price_predictions,
            "investment_recommendation": investment_recommendation,
            "algorithm_steps_completed": [
                "✅ Step 1: News Validation",
                "✅ Step 2: News Impact Analysis", 
                "✅ Step 3: Feature Aggregation",
                "✅ Step 4: Price Forecasting",
                "✅ Step 5: Investment Recommendation"
            ]
        }
        
        # Print summary
        self._print_analysis_summary(complete_analysis)
        
        return complete_analysis
    
    def _print_analysis_summary(self, analysis):
        """Print a comprehensive summary of the analysis"""
        
        print("\n" + "=" * 60)
        print("📋 ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Market Overview
        market_data = analysis["market_data"]
        print(f"💰 Current Gold Price: ${market_data['current_price']}/oz")
        print(f"📈 24h Change: ${market_data['price_change_24h']:+.2f}")
        
        # Price Predictions
        predictions = analysis["price_predictions"]
        print(f"\n🔮 PRICE FORECASTS:")
        print(f"   Short-term (1-7 days): ${predictions['short_term_prediction']}/oz ({predictions['short_term_change_percent']:+.1f}%)")
        print(f"   Long-term (1-3 months): ${predictions['long_term_prediction']}/oz ({predictions['long_term_change_percent']:+.1f}%)")
        print(f"   Prediction confidence: {predictions['prediction_confidence']:.1%}")
        
        # News Analysis
        news = analysis["news_analysis"]
        print(f"\n📰 NEWS SENTIMENT:")
        print(f"   Overall sentiment: {news['overall_sentiment'].upper()}")
        print(f"   Confidence: {news['sentiment_confidence']:.1%}")
        print(f"   Risk factors: {len(news.get('risk_factors', []))}")
        
        # Investment Recommendation
        recommendation = analysis["investment_recommendation"]["recommendation_summary"]
        print(f"\n💡 INVESTMENT RECOMMENDATION:")
        print(f"   Action: {recommendation['action_strength']} {recommendation['primary_action']}")
        print(f"   Confidence: {recommendation['confidence_score']}/10")
        print(f"   Suggested allocation: {analysis['investment_recommendation']['position_details']['suggested_allocation']}")
        
        # Market Indicators
        indicators = market_data["market_indicators"]
        print(f"\n📊 KEY MARKET INDICATORS:")
        print(f"   USD Strength: {indicators.get('dollar_strength_index', 'N/A')}")
        print(f"   Inflation Rate: {indicators.get('inflation_rate', 'N/A')}%")
        print(f"   Interest Rate: {indicators.get('interest_rate', 'N/A')}%")
        print(f"   Market Stress: {indicators.get('market_stress_level', 'N/A')}")
        
        print("\n" + "=" * 60)
        print("✅ Complete analysis finished successfully!")
        print("=" * 60)

# ============================================================================
# 7. DEMO AND TESTING FUNCTIONS
# ============================================================================

def run_algorithm_demo():
    """Run a complete demonstration of the algorithm"""
    
    print("🎯 GOLD TRADING ALGORITHM DEMONSTRATION")
    print("=" * 80)
    
    # Test different market scenarios
    scenarios = ["bullish", "bearish", "neutral", "volatile"]
    
    for scenario in scenarios:
        print(f"\n🧪 Testing scenario: {scenario.upper()}")
        print("-" * 40)
        
        # Initialize algorithm
        algorithm = GoldTradingAlgorithm(llm_type="rule_based")
        
        # Run analysis
        results = algorithm.run_complete_analysis(
            market_scenario=scenario,
            num_articles=10,
            days_history=60
        )
        
        # Brief summary for demo
        rec = results["investment_recommendation"]["recommendation_summary"]
        pred = results["price_predictions"]
        
        print(f"📊 Result: {rec['action_strength']} {rec['primary_action']}")
        print(f"💰 Price forecast: ${pred['short_term_prediction']}/oz ({pred['short_term_change_percent']:+.1f}%)")
        print(f"🎯 Confidence: {rec['confidence_score']}/10")
        
        print("-" * 40)
    
    print("\n✅ Algorithm demonstration completed!")

def save_analysis_results(analysis_results, filename=None):
    """Save analysis results to JSON file"""
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gold_analysis_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print(f"💾 Analysis results saved to: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return None

# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        # Run the complete algorithm demonstration
        run_algorithm_demo()
        
        # Run a detailed analysis
        print(f"\n🎯 RUNNING DETAILED ANALYSIS")
        print("=" * 80)
        
        algorithm = GoldTradingAlgorithm(llm_type="rule_based")
        
        detailed_results = algorithm.run_complete_analysis(
            market_scenario="neutral",
            num_articles=15,
            days_history=120
        )
        
        # Save results
        save_analysis_results(detailed_results)
        
        # Print detailed recommendation
        print(f"\n📋 DETAILED INVESTMENT RECOMMENDATION:")
        print("=" * 80)
        print(detailed_results["investment_recommendation"]["detailed_analysis"])
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\n🎉 Algorithm execution completed!")
        print("=" * 80)