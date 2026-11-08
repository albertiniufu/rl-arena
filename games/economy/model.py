"""Motor do simulador econômico.

O modelo é deliberadamente didático:

- clientes recebem renda (salário ou seguro-desemprego) e compram no máximo
  um produto por passo, escolhendo por utilidade logit (preço × qualidade);
- empresas empregam, produzem com estoque, definem preço por markup e
  ajustam o preço conforme o estoque, pagando salários, custos e impostos;
- o governo arrecada impostos, paga benefícios, faz compras públicas e paga
  juros sobre a dívida, com política ajustável a cada passo;
- choques macroeconômicos afetam crescimento, salários e preços.

Não há conservação exata de dinheiro: empresas com caixa negativo recebem um
"resgate" implícito (e demitem), o que mantém a simulação simples e estável.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import EconomyConfig


@dataclass
class Product:
    """Produto de uma empresa: preço, qualidade, custo e estoque."""

    firm: int
    quality: float
    price: float
    unit_cost: float
    inventory: float = 0.0
    sold_last: float = 0.0


@dataclass
class Firm:
    """Empresa concorrente."""

    cash: float
    employees: int = 0
    products: list[Product] = field(default_factory=list)
    revenue: float = 0.0
    costs: float = 0.0
    profit: float = 0.0
    sales_prev: float = 1.0


@dataclass
class Customer:
    """Consumidor/pessoa da população."""

    employed: bool = False
    savings: float = 0.0
    quality_pref: float = 0.5
    mpc: float = 0.85
    income: float = 0.0
    income_tax: float = 0.0
    net_income: float = 0.0
    spent: float = 0.0


class Economy:
    """Simulação econômica com população, empresas, governo e macro."""

    def __init__(self, config: EconomyConfig | None = None):
        self.cfg = config or EconomyConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.reset()

    # ------------------------------------------------------------------ #
    # Ciclo de vida
    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cfg = self.cfg

        self.step_count = 0
        self.growth = cfg.growth_mean
        self.inflation = cfg.inflation_target
        self.wage_level = cfg.base_wage
        self.benefit_level = cfg.unemployment_benefit
        self.price_index = 1.0
        self.prev_price_index = 1.0
        self.gdp = 0.0
        self.consumption = 0.0
        self.investment = 0.0
        self.government_spending = 0.0
        self.government_funds = 0.0
        self.debt = 0.0
        self.income_tax_revenue = 0.0
        self.corporate_tax_revenue = 0.0
        self.tax_revenue = 0.0
        self.transfers = 0.0
        self.interest_paid = 0.0
        self.bailouts = 0.0
        self.unemployment = 0.0

        # Política inicial (mutável em set_policy)
        self.income_tax = cfg.income_tax
        self.corporate_tax = cfg.corporate_tax
        self.spending_ratio = cfg.government_spending_ratio
        self.interest_rate = cfg.interest_rate

        self.firms: list[Firm] = []
        demand_per_firm = (
            cfg.n_customers * cfg.initial_market_coverage / cfg.n_firms)
        initial_stock = 1.2 * demand_per_firm / max(1, cfg.products_per_firm)
        for firm_id in range(cfg.n_firms):
            products = []
            for _ in range(cfg.products_per_firm):
                unit_cost = float(self.rng.uniform(
                    cfg.min_unit_cost, cfg.max_unit_cost))
                products.append(Product(
                    firm=firm_id,
                    quality=float(self.rng.uniform(0.3, 1.0)),
                    price=(unit_cost + cfg.base_wage / cfg.productivity)
                           * cfg.price_markup,
                    unit_cost=unit_cost,
                    inventory=initial_stock,
                ))
            firm = Firm(cash=cfg.initial_firm_cash, products=products)
            firm.employees = max(
                1, int(round(demand_per_firm / cfg.productivity)))
            firm.sales_prev = demand_per_firm
            self.firms.append(firm)

        self.customers: list[Customer] = [
            Customer(
                savings=cfg.initial_savings,
                quality_pref=float(self.rng.uniform(0.0, 1.0)),
                mpc=float(np.clip(
                    cfg.propensity_to_consume
                    + self.rng.normal(0.0, 0.05), 0.5, 0.99)),
            )
            for _ in range(cfg.n_customers)
        ]
        self._rebalance_jobs()
        self.price_index = self._avg_price()
        self.prev_price_index = self.price_index
        self.unemployment = self._unemployment_rate()
        return self.metrics()

    def step(self) -> dict:
        """Avança um período e devolve as métricas resultantes."""
        self._update_macro()
        self._produce()
        self._rebalance_jobs()
        self._compute_income()
        self._demand()
        self._settle_firms()
        self._settle_government()
        self._update_indicators()
        self.step_count += 1
        return self.metrics()

    # ------------------------------------------------------------------ #
    # Política
    # ------------------------------------------------------------------ #
    def set_policy(self, income_tax: float | None = None,
                   spending_ratio: float | None = None,
                   interest_rate: float | None = None) -> None:
        """Ajusta a política fiscal/monetária (valores são limitados)."""
        if income_tax is not None:
            self.income_tax = float(np.clip(income_tax, 0.0, 0.6))
        if spending_ratio is not None:
            self.spending_ratio = float(np.clip(spending_ratio, 0.05, 0.5))
        if interest_rate is not None:
            self.interest_rate = float(np.clip(interest_rate, 0.0, 0.05))

    # ------------------------------------------------------------------ #
    # Fases do passo
    # ------------------------------------------------------------------ #
    def _update_macro(self) -> None:
        cfg = self.cfg
        shock = float(self.rng.normal(0.0, cfg.growth_volatility))
        if self.rng.random() < cfg.shock_probability:
            shock += float(self.rng.choice([-1.0, 1.0])) * cfg.shock_scale
        # Crescimento com inércia (AR(1)); choques raros amplificam o ruído.
        self.growth = 0.8 * self.growth + shock
        # Salários nominais acompanham a inflação do período anterior.
        indexation = 0.5 * self.inflation + 0.5 * self.cfg.inflation_target
        self.wage_level *= 1.0 + indexation
        self.benefit_level *= 1.0 + indexation

    def _produce(self) -> None:
        cfg = self.cfg
        for firm in self.firms:
            desired_sales = max(2.0, firm.sales_prev * 1.05 + 2.0)
            desired_employees = max(
                1, int(round(desired_sales / cfg.productivity)))
            if firm.cash < 0:
                desired_employees = max(1, desired_employees // 2)
            # O mercado de trabalho se ajusta no mesmo passo (didático).
            firm.employees = desired_employees

            total_inventory = sum(p.inventory for p in firm.products)
            capacity = firm.employees * cfg.productivity
            produced = max(0.0, min(capacity, desired_sales - total_inventory))
            avg_cost = (sum(p.unit_cost for p in firm.products)
                        / max(1, len(firm.products)))
            wage_bill = firm.employees * self.wage_level
            production_cost = produced * avg_cost
            firm.cash -= wage_bill + production_cost
            firm.costs = wage_bill + production_cost
            firm.revenue = 0.0
            firm.profit = 0.0

            for product in firm.products:
                product.inventory *= 1.0 - cfg.inventory_decay
                product.inventory += produced / max(1, len(firm.products))
                product.sold_last = 0.0

    def _rebalance_jobs(self) -> None:
        """Acerta o número de clientes empregados ao total de vagas."""
        jobs = min(sum(f.employees for f in self.firms), len(self.customers))
        employed = [i for i, c in enumerate(self.customers) if c.employed]
        unemployed = [i for i, c in enumerate(self.customers) if not c.employed]

        while len(employed) < jobs and unemployed:
            i = unemployed.pop(int(self.rng.integers(len(unemployed))))
            self.customers[i].employed = True
            employed.append(i)
        while len(employed) > jobs and employed:
            i = employed.pop(int(self.rng.integers(len(employed))))
            self.customers[i].employed = False

    def _compute_income(self) -> None:
        wage = self.wage_level
        self.income_tax_revenue = 0.0
        for customer in self.customers:
            if customer.employed:
                noise = float(self.rng.normal(0.0, self.cfg.income_volatility))
                gross = (wage * max(0.2, 1.0 + noise)
                         * (1.0 + 0.5 * self.growth))
                tax = gross * self.income_tax
            else:
                gross = self.benefit_level
                tax = 0.0
            customer.income = gross
            customer.income_tax = tax
            customer.net_income = gross - tax
            customer.spent = 0.0
            self.income_tax_revenue += tax

    def _demand(self) -> None:
        cfg = self.cfg
        products = [(firm, product)
                    for firm in self.firms
                    for product in firm.products
                    if product.inventory > 1e-6]
        base_utility = np.array([
            cfg.quality_sensitivity * product.quality
            - cfg.price_sensitivity * (product.price
                                       / max(self.price_index, 1e-6))
            for _, product in products
        ], dtype=float)

        self.consumption = 0.0
        for customer in self.customers:
            budget = (customer.net_income
                      + customer.savings * cfg.savings_spend_rate)
            affordable = [
                k for k, (_, product) in enumerate(products)
                if product.inventory >= 1.0 and product.price <= budget
            ]
            if affordable and self.rng.random() < customer.mpc:
                utilities = np.array([
                    base_utility[k]
                    + (customer.quality_pref - 0.5) * products[k][1].quality
                    for k in affordable
                ])
                utilities += self.rng.gumbel(size=len(affordable))
                chosen = affordable[int(np.argmax(utilities))]
                firm, product = products[chosen]
                product.inventory -= 1.0
                product.sold_last += 1.0
                firm.revenue += product.price
                customer.spent += product.price
                self.consumption += product.price
            customer.savings = max(
                0.0, customer.savings + customer.net_income - customer.spent)

    def _settle_firms(self) -> None:
        cfg = self.cfg
        self.corporate_tax_revenue = 0.0
        for firm in self.firms:
            gross_profit = firm.revenue - firm.costs
            tax = max(0.0, gross_profit) * self.corporate_tax
            firm.cash += firm.revenue - tax
            firm.profit = gross_profit - tax
            firm.sales_prev = sum(p.sold_last for p in firm.products)
            self.corporate_tax_revenue += tax

            if firm.cash < 0.0:
                # Resgate implícito: cobre o caixa e demite para se ajustar.
                self.bailouts += -firm.cash
                firm.cash = 0.0
                if firm.employees > 1:
                    firm.employees = max(1, int(firm.employees * 0.8))

            target = max(2.0, firm.sales_prev * 1.05 + 2.0)
            inventory = sum(p.inventory for p in firm.products)
            for product in firm.products:
                if inventory > target * 1.5:
                    product.price *= 0.98
                elif inventory < max(1.0, target * 0.1):
                    product.price *= 1.01
                product.price *= 1.0 + cfg.inflation_target
                product.price = max(
                    product.price,
                    product.unit_cost + 0.8 * self.wage_level / cfg.productivity)
                if firm.profit > 0.0:
                    product.quality = min(1.0, product.quality + 0.002)

    def _settle_government(self) -> None:
        cfg = self.cfg
        base = self.gdp if self.gdp > 0.0 else (
            self.wage_level * sum(f.employees for f in self.firms))
        self.government_spending = self.spending_ratio * base
        self.transfers = (
            self._n_unemployed() * self.benefit_level)
        self.interest_paid = self.debt * self.interest_rate
        self.tax_revenue = self.income_tax_revenue + self.corporate_tax_revenue
        self.government_funds += (
            self.tax_revenue - self.government_spending
            - self.transfers - self.interest_paid)
        self.debt = max(0.0, -self.government_funds)

    def _update_indicators(self) -> None:
        new_index = self._avg_price()
        self.inflation = float(np.clip(
            new_index / max(self.prev_price_index, 1e-6) - 1.0, -0.5, 0.5))
        self.prev_price_index = new_index
        self.price_index = new_index
        self.investment = sum(f.costs for f in self.firms)
        self.gdp = self.consumption + self.government_spending + self.investment
        self.unemployment = self._unemployment_rate()

    # ------------------------------------------------------------------ #
    # Métricas e utilitários
    # ------------------------------------------------------------------ #
    def metrics(self) -> dict:
        n = len(self.customers)
        employed = sum(1 for c in self.customers if c.employed)
        incomes = [
            c.net_income if c.employed else self.benefit_level
            for c in self.customers
        ]
        return {
            "step": self.step_count,
            "gdp": self.gdp,
            "consumption": self.consumption,
            "investment": self.investment,
            "government_spending": self.government_spending,
            "government_funds": self.government_funds,
            "government_debt": self.debt,
            "tax_revenue": self.tax_revenue,
            "income_tax_revenue": self.income_tax_revenue,
            "corporate_tax_revenue": self.corporate_tax_revenue,
            "transfers": self.transfers,
            "interest_paid": self.interest_paid,
            "inflation": self.inflation,
            "unemployment": self.unemployment,
            "employed": employed,
            "n_customers": n,
            "price_index": self.price_index,
            "avg_savings": float(np.mean([c.savings for c in self.customers])),
            "firm_profit": sum(f.profit for f in self.firms),
            "firm_cash": sum(f.cash for f in self.firms),
            "gini": self._gini(incomes),
            "income_tax": self.income_tax,
            "spending_ratio": self.spending_ratio,
            "interest_rate": self.interest_rate,
            "growth": self.growth,
            "bailouts": self.bailouts,
        }

    def _n_unemployed(self) -> int:
        return sum(1 for c in self.customers if not c.employed)

    def _unemployment_rate(self) -> float:
        return self._n_unemployed() / max(1, len(self.customers))

    def _avg_price(self) -> float:
        prices = [p.price for firm in self.firms for p in firm.products]
        return float(np.mean(prices)) if prices else 1.0

    @staticmethod
    def _gini(values) -> float:
        x = np.sort(np.asarray(values, dtype=float))
        n = len(x)
        if n == 0 or x.sum() <= 0.0:
            return 0.0
        index = np.arange(1, n + 1)
        return float(2.0 * np.sum(index * x) / (n * x.sum()) - (n + 1) / n)
