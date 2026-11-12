"""Tabuleiro do risk: 48 territórios/regiões em 6 continentes.

Mapa ampliado a pedido do projeto: América do Norte subdivide EUA e Canadá,
a Europa tem vários países, a África mais países, a Ásia inclui Dudinka,
Sibéria e Vladivostok, e a Oceania tem quatro países. As ligações
intracontinentais aproximam a geografia; as intercontinentais reproduzem as
rotas clássicas de ataque (Bering, Atlântico Norte, Panamá, Atlântico Sul,
Gibraltar, Mediterrâneo, Mar Vermelho, Ártico e Sudeste Asiático).
"""
from __future__ import annotations

CONTINENTS: dict[str, tuple[str, ...]] = {
    "AmericaNorte": (
        "Alaska", "Yukon", "Noroeste", "Ontario", "Quebec",
        "California", "Rockies", "Texas", "Leste", "Mexico",
    ),
    "AmericaSul": (
        "Colombia", "Venezuela", "Amazonia", "Nordeste", "Sudeste",
        "Peru", "Argentina",
    ),
    "Europa": (
        "ReinoUnido", "Franca", "Espanha", "Alemanha", "Italia",
        "Polonia", "Escandinavia", "Balcas", "Ucrania",
    ),
    "Africa": (
        "Egito", "Libia", "Nigeria", "AfricaOcidental", "Congo",
        "AfricaOriental", "Angola", "AfricaDoSul", "Madagascar",
    ),
    "Asia": (
        "Dudinka", "Siberia", "Vladivostok", "China", "India",
        "Japao", "OrienteMedio", "AsiaCentral", "SudesteAsiatico",
    ),
    "Oceania": (
        "Indonesia", "NovaGuine", "Australia", "NovaZelandia",
    ),
}

CONTINENT_BONUS: dict[str, int] = {
    "AmericaNorte": 5,
    "AmericaSul": 3,
    "Europa": 5,
    "Africa": 4,
    "Asia": 7,
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

_INTRA_LINKS = (
    # América do Norte
    ("Alaska", "Yukon"), ("Yukon", "Noroeste"), ("Noroeste", "Ontario"),
    ("Ontario", "Quebec"), ("Quebec", "Leste"), ("Leste", "Texas"),
    ("Texas", "Mexico"), ("Mexico", "California"),
    ("California", "Rockies"), ("Rockies", "Texas"),
    ("Rockies", "Ontario"), ("Ontario", "Leste"),
    # América do Sul
    ("Colombia", "Venezuela"), ("Venezuela", "Amazonia"),
    ("Amazonia", "Nordeste"), ("Nordeste", "Sudeste"),
    ("Sudeste", "Argentina"), ("Argentina", "Peru"),
    ("Peru", "Amazonia"), ("Peru", "Colombia"),
    ("Sudeste", "Amazonia"),
    # Europa
    ("ReinoUnido", "Franca"), ("Franca", "Espanha"),
    ("Franca", "Alemanha"), ("Franca", "Italia"),
    ("Alemanha", "Polonia"), ("Alemanha", "Italia"),
    ("Alemanha", "Escandinavia"), ("Polonia", "Ucrania"),
    ("Polonia", "Escandinavia"), ("Ucrania", "Balcas"),
    ("Balcas", "Italia"), ("Espanha", "Italia"),
    # África
    ("Egito", "Libia"), ("Libia", "Nigeria"),
    ("Nigeria", "AfricaOcidental"), ("AfricaOcidental", "Congo"),
    ("Congo", "Angola"), ("Congo", "AfricaOriental"),
    ("AfricaOriental", "Angola"), ("Angola", "AfricaDoSul"),
    ("AfricaDoSul", "Madagascar"), ("AfricaOriental", "Egito"),
    ("Nigeria", "Congo"),
    # Ásia
    ("Dudinka", "Siberia"), ("Siberia", "Vladivostok"),
    ("Vladivostok", "China"), ("Vladivostok", "Japao"),
    ("China", "India"), ("China", "SudesteAsiatico"),
    ("China", "Japao"), ("India", "SudesteAsiatico"),
    ("India", "OrienteMedio"), ("OrienteMedio", "AsiaCentral"),
    ("AsiaCentral", "Siberia"), ("AsiaCentral", "China"),
    ("AsiaCentral", "India"),
    # Oceania
    ("Indonesia", "NovaGuine"), ("NovaGuine", "Australia"),
    ("Australia", "NovaZelandia"), ("Indonesia", "Australia"),
)

_INTER_LINKS = (
    ("Alaska", "Dudinka"),              # estreito de Bering
    ("Quebec", "ReinoUnido"),           # Atlântico Norte
    ("Mexico", "Colombia"),             # América Central / Panamá
    ("Nordeste", "Nigeria"),            # Atlântico Sul
    ("Espanha", "AfricaOcidental"),     # Gibraltar
    ("Italia", "Libia"),                # Mediterrâneo
    ("Escandinavia", "Dudinka"),        # Ártico
    ("Ucrania", "AsiaCentral"),         # estepe euro-asiática
    ("OrienteMedio", "Egito"),          # Levante / Egito
    ("AfricaOriental", "OrienteMedio"),  # Mar Vermelho
    ("SudesteAsiatico", "Indonesia"),   # Sudeste Asiático / Oceania
)

_ADJACENCY: dict[str, set[str]] = {name: set() for name in TERRITORIES}


def _link(a: str, b: str) -> None:
    if a not in _ADJACENCY or b not in _ADJACENCY:
        raise KeyError(f"território desconhecido: {a!r} ou {b!r}")
    if a == b:
        raise ValueError("um território não se liga a si mesmo")
    _ADJACENCY[a].add(b)
    _ADJACENCY[b].add(a)


for _a, _b in _INTRA_LINKS + _INTER_LINKS:
    _link(_a, _b)

ADJACENCY: dict[str, tuple[str, ...]] = {
    name: tuple(sorted(neighbors)) for name, neighbors in _ADJACENCY.items()
}
ADJACENCY_INDEX: dict[int, tuple[int, ...]] = {
    TERRITORY_INDEX[name]: tuple(
        TERRITORY_INDEX[neighbor] for neighbor in neighbors)
    for name, neighbors in ADJACENCY.items()
}
