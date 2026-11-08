"""Parâmetros do simulador econômico."""
from dataclasses import dataclass


@dataclass
class EconomyConfig:
    """Todos os parâmetros do mundo econômico.

    Os valores são didáticos: um passo representa um período (mês). A renda
    base é por passo, e a moeda é abstrata.
    """

    n_customers: int = 300
    n_firms: int = 6
    products_per_firm: int = 2
    initial_firm_cash: float = 3000.0
    initial_savings: float = 50.0

    # Trabalho e renda
    base_wage: float = 90.0
    productivity: float = 2.0  # unidades produzidas por funcionário/passo
    min_unit_cost: float = 2.0
    max_unit_cost: float = 6.0
    price_markup: float = 1.35
    initial_market_coverage: float = 0.9  # fração dos clientes atendida
    income_volatility: float = 0.08
    unemployment_benefit: float = 60.0

    # Consumo
    propensity_to_consume: float = 0.85
    price_sensitivity: float = 2.0
    quality_sensitivity: float = 3.0
    savings_spend_rate: float = 0.05

    # Governo (valores iniciais; a política pode ser alterada a cada passo)
    income_tax: float = 0.20
    corporate_tax: float = 0.15
    government_spending_ratio: float = 0.15  # fração do PIB
    interest_rate: float = 0.01  # por passo, sobre a dívida

    # Macroeconomia
    inflation_target: float = 0.02
    growth_mean: float = 0.005
    growth_volatility: float = 0.02
    shock_probability: float = 0.02
    shock_scale: float = 0.08
    inventory_decay: float = 0.15

    # Episódio
    max_steps: int = 120
    seed: int = 0
