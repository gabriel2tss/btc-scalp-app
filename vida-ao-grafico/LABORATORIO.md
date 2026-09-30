# Laboratório de testes

Ferramentas para testar ideias de trading **sem se enganar**: janelas às cegas, critérios travados antes,
custos reais (taxa, derrapagem, funding), execução realista com ticks e checagem de liquidação.
Dados: `data/` (só BTC; ver `data/LEIA-ME.md`). Ambiente: `.venv` (Python + PyTorch com CUDA).

## Regras da casa

1. **Travar os critérios antes de ver o resultado** (`config/success_criteria.yaml` + `python -m vag.criteria lock`).
   Se não passar, não se afrouxa a regra depois.
2. **Testar às cegas**: aprende no passado, escolhe na validação, a nota é só do período nunca visto.
3. **Custos reais**: taxa taker 0,05% + derrapagem 0,01% por lado, funding a cada 8 h; ordem limitada só executa
   se um negócio real passar do preço.
4. **Alavancagem sempre com liquidação simulada** (pior preço do dia/minuto contra a margem).
5. **Olhar a família, não a melhor variante**: com muitas variantes, uma sempre parece boa por sorte.
6. **Critério do Gabriel**: tem que ser positivo no ano mais recente.

## Ferramentas

Rodar a partir de `vida-ao-grafico`, com `$env:PYTHONPATH="src"` para os scripts de `scripts/lab`.

| Ferramenta | Para quê |
|---|---|
| `scripts/lab/simulador_binance.py` | US$ X na Binance com alavancagem, mês a mês, taxas, funding e **liquidação**. Ex.: `--estrategia "momentum 14d" --alavancagem 1 2 3 10 --inicio 2025-10-01` |
| `scripts/lab/tendencia_por_ano.py` | Seguir tendência (momentum 7/14/28/56 dias, fluxo de ticks) ano a ano, a partir dos ticks |
| `scripts/lab/martingale_btc.py` | Martingale diário e grid martingale, cada mês isolado, com liquidação |
| `scripts/lab/rompimento.py` | O tamanho do movimento é previsível? Rompimento (o preço escolhe o lado) dá lucro? |
| `scripts/lab/baixar_ticks_paralelo.py` | Baixar ticks da Binance com vários meses em paralelo |
| `python -m vag.walkforward` | Walk-forward completo (features, alfabeto, modelo de linguagem, dicionário, baselines) |
| `python -m vag.barrier_eval` / `vag.barrier_exec` | Modelo de barreiras (alvo e stop) e execução realista com ticks + veredito contra os critérios |
| `python -m vag.execution_eval` | Ordens limitadas simuladas segundo a segundo com os ticks |
| `python -m vag.criteria check / lock / verify` | Pré-registro dos critérios de sucesso |

## O que já foi testado (30/09/2026) — resumo

| Ideia | Resultado |
|---|---|
| Dialeto (alfabeto + modelo de linguagem) para prever direção, 5-60 min | Não passa dos custos; vantagem bruta real mas pequena (+3 a +10 bps) |
| Ticks como informação extra, alfabeto menor, horizontes de 1-24 h, treino mensal, modelo congelado | Não melhoram de forma consistente |
| Ordens limitadas (execução realista) | Levam ao zero a zero, nada significativo |
| Modelo de barreiras (alvo/stop), com e sem dialeto | Negativo em 2025/2 e 2026 |
| Cone de futuros (o modelo "imagina" continuações) | Acerta o tamanho tanto quanto a volatilidade simples; direção ~ acaso |
| Dialeto como filtro de EA (Fortune GRID, ouro) | Regra travada falhou em 2025 (inverteu) |
| Ouro (XAUUSD, RoboForex conta cent) | Custo ~0,45 bps; 2026 promissor, 2025 negativo; pausado |
| Martingale (diário e grid) | Grid quebrou em 8 de 12 meses; nunca melhora o resultado ajustado ao risco |
| **Seguir tendência, momentum 14 dias, 1-2x** | **Único consistente nos anos recentes** (2023 +56%, 2024 +26%, 2025 +13%, último ano +55% a 1x), mas com quedas de 25-60% no caminho; a 10x é liquidado (10/10/2025). Estratégia clássica e pública, não vem do dialeto |

Achado sólido: o **tamanho** do movimento é previsível (volatilidade; correlação ~0,7 com a amplitude da hora
seguinte). A **direção** em minutos/horas, não.

## Ideias em aberto

- Acompanhar o momentum 14d ao vivo com valor pequeno, 1-2x, antes de qualquer valor maior.
- Calendário de eventos (inflação/juros dos EUA, vencimentos, funding) e múltiplas escalas (semana/dia/hora).
- Entender por que um EA funciona num ano e não no outro (volatilidade, spread, horário).
- Usar a volatilidade para gestão de risco (tamanho de posição, distância de stop/take).
- Operar volatilidade com opções (usa o que já sabemos prever: o tamanho do movimento).
