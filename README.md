# OpenStat

Aplicativo desktop de análise estatística, de código aberto (MIT), escrito em Python com PySide6.

É uma implementação independente: os cálculos usam métodos estatísticos públicos e as bibliotecas
NumPy, SciPy, statsmodels, scikit-learn e matplotlib.

## Executar

```
python -m venv .venv
.venv\Scripts\pip install -e .[dev]
.venv\Scripts\python -m openstat
```

O idioma segue o sistema (pt-BR ou inglês). Para forçar: `set OPENSTAT_LANG=en`.

## Testes

```
.venv\Scripts\python -m pytest
```

## Estrutura

- `openstat/core` — planilha, projeto (`.ostat`), importação, resultados e registro de análises.
- `openstat/stats` — análises; funções puras `Worksheet -> Result`, sem dependência de Qt.
- `openstat/graphs` — gráficos matplotlib.
- `openstat/ui` — janela principal, planilha, saída e gerador de diálogos.

## Adicionar uma análise

Uma análise é uma função decorada com `@analysis`. O menu e o diálogo são gerados a partir da
declaração dos parâmetros:

```python
from openstat.core.registry import analysis, Param
from openstat.core.result import Result

@analysis("basic.example", menu=("Stat", "Basic Statistics"), title="Example",
          params=[Param("columns", "Variables", "columns")])
def example(ws, columns):
    res = Result("Example")
    res.table(ws.data(columns).describe().T)
    return res
```
