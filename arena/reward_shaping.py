"""Reward shaping baseado em potencial (PBRS) para o objetivo de moedas.

O objetivo de longo prazo do jogo é terminar o episódio com mais moedas que os
adversários. Em vez de um bônus terminal esparso (que o GAE quase não propaga),
o potencial abaixo transforma a *diferença* de moedas em recompensa densa:

    Phi(s) = moedas do meu time - moedas dos outros times
    r' = r + gamma * Phi(s') - Phi(s)

A forma é a de Ng et al. (1999): preserva a política ótima do MDP original, só
acelera o aprendizado. `gamma` deve ser o mesmo do agente (o default do
ArenaConfig casa com o default do MAPPO/IPPO).
"""


def coin_potential(engine, viewer, team_ids=None):
    """Diferença de moedas do time de `viewer` contra os demais.

    Sem `team_ids` (ambiente single-agent) o "time" é o próprio personagem e
    os demais são todos adversários.
    """
    characters = engine.characters
    if team_ids is None or len(team_ids) != len(characters):
        own = characters[viewer].coins
        return float(own - sum(c.coins for i, c in enumerate(characters)
                               if i != viewer))
    team = team_ids[viewer]
    own = sum(c.coins for i, c in enumerate(characters) if team_ids[i] == team)
    others = sum(c.coins for i, c in enumerate(characters)
                 if team_ids[i] != team)
    return float(own - others)
