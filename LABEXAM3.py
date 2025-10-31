from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Optional, Iterable
import os
import inspect


# -----------------------------
# Models
# -----------------------------
AssetType = Literal["equity", "bond"]


@dataclass(frozen=True)
class Position:
    asset_type: AssetType
    symbol: str
    quantity: float
    price: float
    volatility: float  # daily stdev as decimal, e.g. 0.02 for 2%

    @property
    def market_value(self) -> float:
        return self.quantity * self.price


@dataclass(frozen=True)
class EquityPosition(Position):
    sector: str | None = None


@dataclass(frozen=True)
class BondPosition(Position):
    duration_years: float | None = None
    credit_rating: str | None = None


# -----------------------------
# Legacy risk (intentionally duplicated aggregation)
# -----------------------------

def _legacy_position_var(position: Position, confidence: float = 0.99) -> float:
    z = 2.33 if confidence >= 0.99 else 1.65
    return position.market_value * position.volatility * z


def _legacy_position_cvar(position: Position, confidence: float = 0.99) -> float:
    tail_factor = 1.2 if confidence >= 0.99 else 1.1
    return _legacy_position_var(position, confidence) * tail_factor


def legacy_portfolio_var(positions: Iterable[Position], confidence: float = 0.99) -> float:
    total_value = 0.0
    weighted_sum = 0.0
    for p in positions:
        mv = p.market_value
        total_value += mv
        weighted_sum += _legacy_position_var(p, confidence)
    if total_value == 0:
        return 0.0
    return weighted_sum


def legacy_portfolio_cvar(positions: Iterable[Position], confidence: float = 0.99) -> float:
    total_value = 0.0
    weighted_sum = 0.0
    for p in positions:
        mv = p.market_value
        total_value += mv
        weighted_sum += _legacy_position_cvar(p, confidence)
    if total_value == 0:
        return 0.0
    return weighted_sum


def legacy_portfolio_var_95(positions: Iterable[Position]) -> float:
    return legacy_portfolio_var(positions, confidence=0.95)


# -----------------------------
# Refactored calculators and portfolio service
# -----------------------------
class RiskCalculator(Protocol):
    def value_at_risk(self, position: Position, confidence: float) -> float:
        ...

    def conditional_var(self, position: Position, confidence: float) -> float:
        ...


@dataclass
class EquityRiskCalculator:
    def _z_score(self, confidence: float) -> float:
        return 2.33 if confidence >= 0.99 else 1.65

    def value_at_risk(self, position: Position, confidence: float) -> float:
        z = self._z_score(confidence)
        return position.market_value * position.volatility * z

    def conditional_var(self, position: Position, confidence: float) -> float:
        tail_factor = 1.2 if confidence >= 0.99 else 1.1
        return self.value_at_risk(position, confidence) * tail_factor


@dataclass
class BondRiskCalculator:
    def _z_score(self, confidence: float) -> float:
        return 2.33 if confidence >= 0.99 else 1.65

    def value_at_risk(self, position: Position, confidence: float) -> float:
        duration_dampen = 0.9
        z = self._z_score(confidence)
        return position.market_value * position.volatility * duration_dampen * z

    def conditional_var(self, position: Position, confidence: float) -> float:
        tail_factor = 1.18 if confidence >= 0.99 else 1.08
        return self.value_at_risk(position, confidence) * tail_factor


@dataclass
class PortfolioRiskService:
    equity_calculator: EquityRiskCalculator
    bond_calculator: BondRiskCalculator

    def _select_calculator(self, position: Position):
        if position.asset_type == "equity":
            return self.equity_calculator
        if position.asset_type == "bond":
            return self.bond_calculator
        raise ValueError(f"Unsupported asset type: {position.asset_type}")

    def _aggregate(self, positions: Iterable[Position], fn_name: str, confidence: float) -> float:
        total = 0.0
        for p in positions:
            calc = self._select_calculator(p)
            fn = getattr(calc, fn_name)
            total += fn(p, confidence)
        return total

    def portfolio_var(self, positions: Iterable[Position], confidence: float = 0.99) -> float:
        return self._aggregate(positions, "value_at_risk", confidence)

    def portfolio_cvar(self, positions: Iterable[Position], confidence: float = 0.99) -> float:
        return self._aggregate(positions, "conditional_var", confidence)


# -----------------------------
# AI analyzer with offline fallback (OpenAI optional)
# -----------------------------
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


def _offline_sample() -> str:
    return (
        "Refactor removes duplicated aggregation, centralizes risk aggregation via a service, "
        "and isolates asset-specific logic behind calculator classes. This improves SRP, "
        "testability, and extensibility (add asset types without touching aggregation)."
    )


def analyze_refactor_diff(old_code: str, new_code: str, model: Optional[str] = None) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return _offline_sample()

    client = OpenAI(api_key=api_key)
    chosen_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    prompt = (
        "You are a senior finance engineer. Analyze how the new code refactors the old. "
        "Identify removed duplication, separation of concerns, and extensibility. Keep it under 120 words.\n\n"
        f"OLD:\n{old_code[:4000]}\n\nNEW:\n{new_code[:4000]}"
    )

    try:
        completion = client.chat.completions.create(
            model=chosen_model,
            messages=[
                {"role": "system", "content": "You are a concise code reviewer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=180,
        )
        return completion.choices[0].message.content or _offline_sample()
    except Exception:
        return _offline_sample()


# -----------------------------
# Demo
# -----------------------------

def sample_positions() -> list[Position]:
    return [
        EquityPosition(asset_type="equity", symbol="AAPL", quantity=50, price=180.0, volatility=0.022, sector="Tech"),
        EquityPosition(asset_type="equity", symbol="MSFT", quantity=30, price=400.0, volatility=0.018, sector="Tech"),
        BondPosition(asset_type="bond", symbol="US10Y", quantity=10, price=1000.0, volatility=0.007, duration_years=9.0, credit_rating="AAA"),
    ]


def run_demo() -> None:
    positions = sample_positions()

    # Legacy computations
    legacy_var_99 = legacy_portfolio_var(positions, confidence=0.99)
    legacy_cvar_99 = legacy_portfolio_cvar(positions, confidence=0.99)

    # Refactored service
    service = PortfolioRiskService(EquityRiskCalculator(), BondRiskCalculator())
    ref_var_99 = service.portfolio_var(positions, confidence=0.99)
    ref_cvar_99 = service.portfolio_cvar(positions, confidence=0.99)

    print("=== Portfolio Risk (99%) ===")
    print(f"Legacy VaR:   {legacy_var_99:,.2f}")
    print(f"Refactored VaR: {ref_var_99:,.2f}")
    print(f"Legacy CVaR:  {legacy_cvar_99:,.2f}")
    print(f"Refactored CVaR: {ref_cvar_99:,.2f}")

    # AI analysis of refactor (offline fallback if API key absent)
    old_code = inspect.getsource(legacy_portfolio_var) + "\n" + inspect.getsource(legacy_portfolio_cvar)
    new_code = inspect.getsource(PortfolioRiskService)
    ai_summary = analyze_refactor_diff(old_code, new_code)

    print("\n=== AI Refactor Analysis ===")
    print(ai_summary)


if __name__ == "__main__":
    run_demo() 