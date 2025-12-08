from django.shortcuts import render
from django.db.models import Count, Q, Avg
from .models import Prediction, Feedback

def home(request):
    return render(request, "core/home.html")

def about_reelioo(request):
    return render(request, "core/about_reelioo.html")

def feedback_page(request):
    return render(request, "core/feedback_page.html")

def why_reelioo(request):
    return render(request, "core/why_reelioo.html")


def dashboard(request):
    """
    Superuser Dashboard.
    Calculates Win Rates per Market, Total Feedback, and Recent Activity.
    """
    # 1. Prediction Stats
    preds = Prediction.objects.all().order_by("-created_at")
    total_preds = preds.count()
    
    # Completed trades
    completed = preds.exclude(actual_outcome="PENDING")
    evaluated_count = completed.count()
    
    # Wins
    wins = completed.filter(is_correct=True).count()
    accuracy = round((wins / evaluated_count * 100), 1) if evaluated_count > 0 else 0
    
    # 2. Market Breakdown
    markets = ["EQUITY", "CRYPTO", "FOREX", "FUTURES", "OPTIONS", "METALS"]
    market_stats = []
    
    for m in markets:
        m_preds = completed.filter(market=m)
        m_total = m_preds.count()
        m_wins = m_preds.filter(is_correct=True).count()
        m_acc = round((m_wins / m_total * 100), 1) if m_total > 0 else 0
        market_stats.append({
            "name": m,
            "total": m_total,
            "wins": m_wins,
            "accuracy": m_acc
        })

    # 3. Feedback Stats
    feedbacks = Feedback.objects.all().order_by("-created_at")
    avg_accuracy_rating = feedbacks.aggregate(Avg('accuracy_rating'))['accuracy_rating__avg'] or 0
    hallucinations = feedbacks.filter(hallucination_report=True).count()

    context = {
        "total_predictions": total_preds,
        "evaluated_count": evaluated_count,
        "overall_accuracy": accuracy,
        "market_stats": market_stats,
        "recent_predictions": preds[:20],
        "recent_feedback": feedbacks[:10],
        "avg_user_rating": round(avg_accuracy_rating, 1),
        "hallucination_count": hallucinations
    }
    
    return render(request, "core/dashboard.html", context)