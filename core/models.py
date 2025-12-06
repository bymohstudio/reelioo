from django.db import models

class Prediction(models.Model):
    MARKET_CHOICES = [
        ("EQUITY", "Equity"),
        ("FUTURES", "Futures"),
        ("OPTIONS", "Options"),
        ("CRYPTO", "Crypto"),
        ("FOREX", "Forex"),
        ("METALS", "Metals"),
    ]
    DIRECTION_CHOICES = [
        ("UP", "Up"),
        ("DOWN", "Down"),
        ("FLAT", "Flat"),
    ]
    OUTCOME_CHOICES = [
        ("PENDING", "Pending"),
        ("WIN", "Win"),
        ("LOSS", "Loss"),
        ("BREAKEVEN", "Breakeven"),
    ]

    created_at = models.DateTimeField(auto_now_add=True)
    symbol = models.CharField(max_length=50)
    market = models.CharField(max_length=20, choices=MARKET_CHOICES)
    trade_type = models.CharField(max_length=20)  # INTRADAY/SWING/LONG_TERM

    model_probability = models.FloatField()
    predicted_direction = models.CharField(max_length=10, choices=DIRECTION_CHOICES)
    
    entry_price = models.FloatField()
    target_price = models.FloatField()
    stop_loss_price = models.FloatField()

    # Outcome tracking
    actual_outcome = models.CharField(
        max_length=10, choices=OUTCOME_CHOICES, default="PENDING"
    )
    pnl_percent = models.FloatField(null=True, blank=True) # Realized PnL
    is_correct = models.BooleanField(null=True, blank=True)

    def __str__(self):
        return f"{self.symbol} {self.market} ({self.created_at:%Y-%m-%d})"


class Feedback(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    market = models.CharField(max_length=20, blank=True, default="GLOBAL")
    symbol = models.CharField(max_length=50, blank=True)
    
    # 1-5 Star Ratings
    accuracy_rating = models.IntegerField(default=0)
    ux_rating = models.IntegerField(default=0)
    speed_rating = models.IntegerField(default=0)
    
    # Text
    comment = models.TextField(blank=True)
    
    # Flags
    hallucination_report = models.BooleanField(default=False)
    
    def __str__(self):
        return f"Feedback: {self.market} - {self.accuracy_rating}/5"