# Vida ao Gráfico

Dar ao mercado um dialeto próprio, **descoberto a partir dos dados** (não inventado por
humanos): alfabeto aprendido (VQ-VAE) → palavras (BPE) → modelo de linguagem do mercado →
dicionário de significados → validação walk-forward "verdade × mentira" → voz (LLM narrando
só o que a estatística comprovou).

Briefing completo: [`docs/briefing.pdf`](docs/briefing.pdf). Ele é a fonte da verdade do projeto.

## Status

| Fase | O quê | Status |
|---|---|---|
| 0 | Reconhecimento (disco, GPU, MT5, histórico da corretora) | **código pronto — rodar no PC** |
| 1 | Coleta MT5 → Parquet+zstd por ativo/ano + manifesto | **código pronto — aguardando ativo** |
| 2 | Estado rico por instante (features sem look-ahead) | — |
| 3 | Alfabeto (VQ-VAE) | — |
| 4 | Palavras (BPE) + modelo de linguagem | — |
| 5 | Dicionário de significados | — |
| 6 | Verdade × mentira (walk-forward) | bloqueada até travar `config/success_criteria.yaml` |
| 7 | Voz | — |
| 8 | Integração com EAs / JEV | futuro |

## Pendências (do Gabriel)

- [ ] Rodar a Fase 0 no PC e mandar o relatório (`reports/phase0_recon_*.md`).
- [ ] Escolher o primeiro ativo.
- [ ] Confirmar o fuso do servidor da corretora (`config/project.yaml` → `server_time`).
- [ ] Preencher `config/success_criteria.yaml` e travar (`python -m vag.criteria lock`) antes da Fase 6.

## Como rodar (PC Windows com MT5)

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

## Decisões de implementação

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
