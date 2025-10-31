import os
import math
import random
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional


# =============================
# Data Models
# =============================


@dataclass
class SensorReading:
    index: int
    value: float


@dataclass
class Anomaly:
    index: int
    value: float
    score: float


@dataclass
class ForecastResult:
    forecasts: List[float]
    level: float
    trend: float
    alpha: float
    beta: float
    damped: bool = False
    phi: Optional[float] = None


# =============================
# Anomaly Detection (Robust)
# =============================


def _median_absolute_deviation(values: List[float]) -> float:
    if not values:
        return 0.0
    med = median(values)
    abs_devs = [abs(v - med) for v in values]
    return median(abs_devs)


def detect_anomalies(
    series: List[float],
    window: int = 24,
    threshold: float = 3.5,
    min_mad: float = 1e-9,
) -> List[Anomaly]:
    """Detect anomalies using robust z-scores with rolling median and MAD."""
    anomalies: List[Anomaly] = []
    if not series:
        return anomalies

    mad_scale = 1.4826  # to approximate std under normality

    for i, value in enumerate(series):
        start = max(0, i - window)
        window_values = series[start:i] if i > 0 else []
        if len(window_values) < max(5, window // 4):
            continue

        med = median(window_values)
        mad = _median_absolute_deviation(window_values)
        if mad < min_mad:
            mad = min_mad
        robust_z = (value - med) / (mad_scale * mad)
        if abs(robust_z) >= threshold:
            anomalies.append(Anomaly(index=i, value=float(value), score=abs(robust_z)))

    return anomalies


# =============================
# Forecasting (Holt Linear)
# =============================


def holt_linear_forecast(
    series: List[float],
    steps: int = 12,
    alpha: float = 0.3,
    beta: float = 0.1,
    damped: bool = False,
    phi: float = 0.98,
) -> ForecastResult:
    """Holt's linear trend method with optional damping (no seasonality)."""
    if not series:
        return ForecastResult(forecasts=[], level=0.0, trend=0.0, alpha=alpha, beta=beta, damped=damped, phi=phi)

    if len(series) == 1:
        level = float(series[0])
        trend = 0.0
    else:
        level = float(series[0])
        trend = float(series[1] - series[0])

    for t in range(1, len(series)):
        value = float(series[t])
        prev_level = level
        level = alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend

    forecasts: List[float] = []
    if damped:
        pow_phi = 1.0
        for h in range(1, steps + 1):
            pow_phi *= phi
            if phi != 1.0:
                forecasts.append(level + (pow_phi - 1.0) / (phi - 1.0) * trend)
            else:
                forecasts.append(level + h * trend)
    else:
        for h in range(1, steps + 1):
            forecasts.append(level + h * trend)

    return ForecastResult(
        forecasts=forecasts,
        level=level,
        trend=trend,
        alpha=alpha,
        beta=beta,
        damped=damped,
        phi=phi,
    )


# =============================
# AI Assistant (LLM with Fallback)
# =============================


def _fallback_summary(anomalies: List[Anomaly], forecast: ForecastResult, series: List[float]) -> str:
    num_anom = len(anomalies)
    max_score = max((a.score for a in anomalies), default=0.0)

    trend_direction = "flat"
    if forecast.trend > 0.05:
        trend_direction = "upward"
    elif forecast.trend < -0.05:
        trend_direction = "downward"

    last_val = series[-1] if series else 0.0
    next_val = forecast.forecasts[0] if forecast.forecasts else last_val
    delta = next_val - last_val

    suggestions: List[str] = []
    if num_anom > 0:
        suggestions.append("Validate sensor calibration and check for sudden environmental changes.")
    if abs(delta) > 0.5:
        suggestions.append("Prepare mitigation plan for short-term fluctuations.")
    if abs(forecast.trend) > 0.1:
        suggestions.append("Investigate root causes for sustained trend and adjust thresholds.")
    if not suggestions:
        suggestions.append("No immediate action required; continue routine monitoring.")

    return (
        f"Detected {num_anom} anomalies (max robust z-score={max_score:.2f}). "
        f"Trend appears {trend_direction} (trend={forecast.trend:.3f}). "
        f"Next value change ~{delta:.2f}. Suggested actions: " + " ".join(suggestions)
    )


def summarize_findings(
    anomalies: List[Anomaly],
    forecast: ForecastResult,
    series: List[float],
    model_name: str = "gpt-4o-mini",
) -> str:
    """Summarize findings using OpenAI when available; otherwise local fallback."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return _fallback_summary(anomalies, forecast, series)

    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        prompt = (
            "You are an environmental monitoring assistant. Given anomaly stats "
            "and short-term forecasts, produce a concise, actionable summary with "
            "1-3 bullet recommendations. Keep to 80-120 words.\n\n"
            f"Anomalies: {len(anomalies)}, max_score={max((a.score for a in anomalies), default=0.0):.2f}.\n"
            f"Trend: {forecast.trend:.4f}. Next forecasts (first 5): {forecast.forecasts[:5]}\n"
        )
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Be precise, technical, and concise."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=160,
        )
        content = resp.choices[0].message.content if resp and resp.choices else None
        return content or _fallback_summary(anomalies, forecast, series)
    except Exception:
        return _fallback_summary(anomalies, forecast, series)


# =============================
# Pipeline
# =============================


class EnvMonitorPipeline:
    def __init__(
        self,
        anomaly_window: int = 24,
        anomaly_threshold: float = 3.5,
        forecast_steps: int = 24,
        alpha: float = 0.3,
        beta: float = 0.1,
        damped: bool = False,
        phi: float = 0.98,
    ) -> None:
        self.anomaly_window = anomaly_window
        self.anomaly_threshold = anomaly_threshold
        self.forecast_steps = forecast_steps
        self.alpha = alpha
        self.beta = beta
        self.damped = damped
        self.phi = phi

    def run(self, series: List[float]) -> Dict[str, object]:
        clean_series = [float(x) for x in series if x is not None]
        anomalies: List[Anomaly] = detect_anomalies(
            clean_series,
            window=self.anomaly_window,
            threshold=self.anomaly_threshold,
        )
        forecast: ForecastResult = holt_linear_forecast(
            clean_series,
            steps=self.forecast_steps,
            alpha=self.alpha,
            beta=self.beta,
            damped=self.damped,
            phi=self.phi,
        )
        summary: str = summarize_findings(anomalies, forecast, clean_series)

        return {
            "anomalies": anomalies,
            "forecast": forecast,
            "summary": summary,
        }


# =============================
# Synthetic Data Test Runner
# =============================


def generate_synthetic_series(n: int = 24 * 7, seed: int = 42) -> List[float]:
    random.seed(seed)
    series = []
    for t in range(n):
        daily_cycle = 5.0 * math.sin(2 * math.pi * (t % 24) / 24.0)
        baseline = 25.0
        noise = random.gauss(0.0, 1.0)
        series.append(baseline + daily_cycle + noise)

    anomaly_indices = random.sample(range(24, n - 1), k=6)
    for idx in anomaly_indices:
        spike = random.choice([8.0, -8.0, 10.0, -10.0])
        series[idx] += spike
    return series


def main() -> None:
    series = generate_synthetic_series()
    pipeline = EnvMonitorPipeline(
        anomaly_window=24,
        anomaly_threshold=3.5,
        forecast_steps=24,
        alpha=0.35,
        beta=0.15,
        damped=False,
    )
    result = pipeline.run(series)
    anomalies = result["anomalies"]
    forecast = result["forecast"]
    summary = result["summary"]

    print("=== Environmental Monitoring AI Test ===")
    print(f"Total points: {len(series)}")
    print(f"Anomalies detected: {len(anomalies)}")
    print("Sample anomalies (up to 5):")
    for a in anomalies[:5]:
        print(f"  idx={a.index:3d} value={a.value:7.2f} score={a.score:5.2f}")
    print("\nFirst 5 forecasts:")
    print([round(x, 2) for x in forecast.forecasts[:5]])
    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()


