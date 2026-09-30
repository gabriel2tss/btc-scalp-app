# Vida ao Gráfico

Dar ao mercado um dialeto próprio, **descoberto a partir dos dados** (não inventado por
humanos): alfabeto aprendido (VQ-VAE) → palavras (BPE) → modelo de linguagem do mercado →
dicionário de significados → validação walk-forward "verdade × mentira" → voz (LLM narrando
só o que a estatística comprovou).

Briefing completo: [`docs/briefing.pdf`](docs/briefing.pdf). Ele é a fonte da verdade do projeto.

## Status

| Fase | O quê | Status |
|---|---|---|
| 0 | Reconhecimento (disco, GPU, histórico disponível) | código pronto |
| 1 | Download de dados reais (Dukascopy / Binance) → Parquet+zstd + manifesto | **código pronto — aguardando liberação de rede** |
| 2 | Estado rico por instante (features sem look-ahead) | — |
| 3 | Alfabeto (VQ-VAE) | — |
| 4 | Palavras (BPE) + modelo de linguagem | — |
| 5 | Dicionário de significados | — |
| 6 | Verdade × mentira (walk-forward) | bloqueada até travar `config/success_criteria.yaml` |
| 7 | Voz | — |
| 8 | Integração com EAs / JEV | futuro |

## Pendências (do Gabriel)

- [ ] Liberar na rede do ambiente de nuvem: `datafeed.dukascopy.com` e `data.binance.vision`.
- [ ] Escolher o primeiro ativo.
- [ ] Preencher `config/success_criteria.yaml` e travar (`python -m vag.criteria lock`) antes da Fase 6.

## Como rodar (nuvem)

Tudo roda no ambiente de nuvem do Claude Code, com dados reais de fontes públicas:

- **Dukascopy** (forex, ouro, índices): velas M1 com bid **e** ask (spread real por minuto) e
  ticks com volume, desde ~2003, já em UTC.
- **Binance** (cripto spot): velas 1m com volume comprador agressor e aggTrades (cada negócio,
  com lado do agressor), conferidos pelo sha256 publicado pela Binance.

```bash
pip install -e ".[dev]"
python -m vag.data.download --source dukascopy --symbol EURUSD  --kind m1     --start 2015-01
python -m vag.data.download --source dukascopy --symbol EURUSD  --kind ticks  --start 2023-01
python -m vag.data.download --source binance   --symbol BTCUSDT --kind m1     --start 2018-01
python -m vag.data.download --source binance   --symbol BTCUSDT --kind trades --start 2024-01
```

O container da nuvem é efêmero: os dados (`data/`) não vão para o git e são baixados de novo
quando necessário (o download é retomável e o manifesto registra tudo).

## Alternativa: PC Windows com MT5

```powershell
cd vida-ao-grafico
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1

# Fase 0 — com o MT5 aberto e logado na corretora
.\.venv\Scripts\python -m vag.recon
.\.venv\Scripts\python -m vag.recon --symbols EURUSD XAUUSD --first-year 2010

# Fase 1 — depois de escolher o ativo e confirmar o fuso
.\.venv\Scripts\python -m vag.data.collect --symbol EURUSD --kind m1    --start 2015-01
.\.venv\Scripts\python -m vag.data.collect --symbol EURUSD --kind ticks --start 2023-01
```

A coleta é mês a mês, retomável (meses completos já baixados são pulados), grava direto em
Parquet+zstd (nunca CSV) e **para sozinha** se o disco livre cair abaixo de `disk.min_free_gb`.

## Rodada walk-forward (Fases 2–6)

```powershell
.\.venv\Scripts\python -m vag.data.download --source binance-um --symbol BTCUSDT --kind m1 --start 2019-09
.\.venv\Scripts\python -m vag.walkforward --smoke     # teste rápido de ponta a ponta
.\.venv\Scripts\python -m vag.walkforward             # rodada completa (retomável)
```

Resultados em `runs/<run_name>/report.md` (atualizado a cada janela concluída) e `log.txt`.
Parâmetros em `config/experiment.yaml`. Em cada janela: treino → validação (onde os dados escolhem
horizonte, ocorrências mínimas, FDR, limiares) → teste (visto uma única vez). Comparação obrigatória
com acaso, momentum/reversão, GBM nas features cruas e GBM com o dialeto.

## Decisões de implementação

- **LM sobre letras, dicionário sobre palavras.** A segmentação BPE clássica olha letras futuras para
  decidir onde uma palavra termina; aqui a palavra em t é a maior que *termina* em t (só passado).
  O mini-GPT lê letras (1 por minuto), o que mantém tudo causal e alinhado ao relógio.
- **Features só com janelas finitas** (sem EWM): calcular sobre as últimas `LOOKBACK` velas dá
  exatamente o mesmo que sobre o histórico todo — o código do backtest é o código ao vivo.
- **Operação executável:** decide no fechamento de t, entra na abertura de t+1, sai no fechamento de
  t+h, uma posição por vez; custos taker (0,05%+0,01% por lado) e cenário maker.

- **Fuso horário.** O pacote `MetaTrader5` devolve horários no relógio do *servidor* da
  corretora, não em UTC. Tudo é convertido para UTC na entrada. O padrão `ny_close`
  (servidor = Nova York + 7h, ou seja UTC+2/UTC+3 seguindo o horário de verão **dos EUA**)
  cobre a maioria das corretoras de forex; a Fase 0 estima o offset real para confirmar.
- **Layout em disco:** `data/<m1|ticks>/symbol=<SYM>/year=<AAAA>/<AAAA-MM>.parquet` +
  `data/manifest.json` (fonte, período, linhas, bytes, lacunas, fuso, info do símbolo
  incluindo `point`/`digits` para converter spread em preço).
- **Lacunas** são registradas por mês e separadas em fim de semana × dia útil.
- **Critérios pré-registrados:** `vag.criteria lock` grava um hash; a Fase 6 vai se recusar a
  rodar se o arquivo tiver mudado depois da trava (seção 8 do briefing).
- Dados ficam fora do git (`.gitignore`).

## Testes

```bash
pip install -e ".[dev]"
pytest -q
```

Os testes usam uma fonte MT5 falsa (`tests/fake_mt5.py`), então rodam em qualquer máquina.
