"""Tabuleiro: territórios, continentes, bônus e adjacências.

Mapa didático com 6 continentes de 4 territórios (24 no total). As ligações
intercontinentais mantêm o grafo conectado e criam as rotas clássicas de
ataque (Alasca–Sibéria, Canadá–Reino Unido, México–Colômbia etc.).
"""
from __future__ import annotations

CONTINENTS: dict[str, tuple[str, ...]] = {
    "AmericaNorte": ("Alaska", "Canada", "Mexico", "California"),
    "AmericaSul": ("Colombia", "Brasil", "Peru", "Argentina"),
    "Europa": ("ReinoUnido", "Franca", "Alemanha", "Italia"),
    "Africa": ("Egito", "Nigeria", "Congo", "AfricaDoSul"),
    "Asia": ("Siberia", "China", "India", "Japao"),
    "Oceania": ("Australia", "Indonesia", "NovaZelandia", "NovaGuine"),
}

CONTINENT_BONUS: dict[str, int] = {
    "AmericaNorte": 3,
    "AmericaSul": 2,
    "Europa": 3,
    "Africa": 2,
    "Asia": 4,
    "Oceania": 2,
}

TERRITORIES: tuple[str, ...] = tuple(
    name for names in CONTINENTS.values() for name in names)
TERRITORY_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(TERRITORIES)}
CONTINENT_OF: dict[str, str] = {
    name: continent
    for continent, names in CONTINENTS.items()
    for name in names
}
CONTINENT_TERRITORIES: dict[str, tuple[int, ...]] = {
    continent: tuple(TERRITORY_INDEX[name] for name in names)
    for continent, names in CONTINENTS.items()
}

_ADJACENCY: dict[str, set[str]] = {name: set() for name in TERRITORIES}


def _link(a: str, b: str) -> None:
    if a not in _ADJACENCY or b not in _ADJACENCY:
        raise KeyError(f"território desconhecido: {a!r} ou {b!r}")
    if a == b:
        raise ValueError("um território não se liga a si mesmo")
    _ADJACENCY[a].add(b)
    _ADJACENCY[b].add(a)


# Anel dentro de cada continente + uma corda para dar alternativas.
for _names in CONTINENTS.values():
    for _i in range(len(_names)):
        _link(_names[_i], _names[(_i + 1) % len(_names)])
    _link(_names[0], _names[2])

# Rotas intercontinentais.
for _a, _b in (
    ("Alaska", "Siberia"),
    ("Canada", "ReinoUnido"),
    ("Mexico", "Colombia"),
    ("Brasil", "Nigeria"),
    ("Italia", "Egito"),
    ("Alemanha", "Siberia"),
    ("Egito", "India"),
    ("India", "Indonesia"),
):
    _link(_a, _b)

ADJACENCY: dict[str, tuple[str, ...]] = {
    name: tuple(sorted(neighbors)) for name, neighbors in _ADJACENCY.items()
}
ADJACENCY_INDEX: dict[int, tuple[int, ...]] = {
    TERRITORY_INDEX[name]: tuple(
        TERRITORY_INDEX[neighbor] for neighbor in neighbors)
    for name, neighbors in ADJACENCY.items()
}
