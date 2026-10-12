# OpenStat

Aplicativo desktop de análise estatística, de código aberto (MIT), escrito em Python com PySide6.

É uma implementação independente: os cálculos usam métodos estatísticos públicos e as bibliotecas
NumPy, SciPy, statsmodels, scikit-learn, lifelines, pyDOE3 e matplotlib.

## Executar

```
python -m venv .venv
.venv\Scripts\pip install -e .[dev]
.venv\Scripts\python -m openstat
```

Requer Python 3.11 ou mais novo. O idioma segue o sistema (pt-BR ou inglês) e pode ser trocado em
**Ferramentas > Idioma**, ou forçado com a variável `OPENSTAT_LANG=en`.

## O que tem

| Menu | Conteúdo |
|---|---|
| Dados | ordenar, subconjunto, mesclar e dividir planilhas, copiar colunas, excluir linhas, empilhar/desempilhar, transpor, concatenar, postos, padronizar, recodificar, alterar tipo, extrair partes de data/hora |
| Calc | calculadora de fórmulas, fórmulas vinculadas a colunas, estatísticas de colunas e linhas, dados padronizados, em malha e aleatórios, variáveis indicadoras, matrizes, distribuições de probabilidade |
| Estatística básica | descritiva, resumo gráfico, Z e t, t pareado, proporções, taxas de Poisson, variâncias, correlação, covariância, normalidade, outlier (Grubbs), ajuste de Poisson |
| Regressão | linear com diagnósticos, linha ajustada, stepwise, melhores subconjuntos, não linear, logística binária/ordinal/nominal, Poisson, PLS, ortogonal |
| ANOVA | um fator (Tukey, Fisher, Dunnett, Welch), modelo linear geral, igualdade de variâncias, aninhada, efeitos mistos, MANOVA, efeitos principais e interação |
| DOE | fatorial completo e fracionado, Plackett-Burman, fatorial geral, composto central, Box-Behnken, misturas, Taguchi; análise, superfície de resposta e otimizador |
| Cartas de controle | Xbarra-R, Xbarra-S, I-AM, P, NP, C, U, Laney, EWMA, CUSUM, média móvel, T², G, T, Box-Cox |
| Ferramentas da qualidade | gráfico de sequência, Pareto, causa e efeito, identificação de distribuição, Johnson, capacidade (normal, não normal, binomial, Poisson, sixpack), Gage R&R, tolerância, amostragem de aceitação |
| Confiabilidade | análise paramétrica com censura à direita e arbitrária, Kaplan-Meier, regressão com dados de vida, teste de vida acelerado, probit, curva de crescimento, plano de demonstração |
| Multivariada | componentes principais, análise fatorial, itens, clusters, discriminante, correspondência |
| Séries temporais | tendência, decomposição, média móvel, suavização exponencial, Winters, ACF, PACF, CCF, ARIMA |
| Tabelas | contagem, tabulação cruzada, testes qui-quadrado |
| Não paramétricos | sinais, Wilcoxon, Mann-Whitney, Kruskal-Wallis, Mood, Friedman, corridas |
| Equivalência | 1 amostra, 2 amostras, pareado, crossover 2x2 |
| Poder e tamanho de amostra | testes Z e t, proporções, variâncias, ANOVA, fatorial, equivalência, estimação, tolerância |
| Análise preditiva | CART, random forest e gradient boosting (classificação e regressão) |
| Gráfico | 21 tipos, de histograma a superfície 3D |

Projetos são salvos em `.ostat` (um zip com as planilhas em Parquet e a saída em HTML). Dados entram
por CSV, TXT ou Excel, ou colados da área de transferência. A saída pode ser exportada em HTML, PDF
ou Word.

As colunas são numéricas, de texto (`-T` no cabeçalho) ou de data/hora (`-D`). Datas são digitadas
como `aaaa-mm-dd` ou `dd/mm/aaaa` (`mm/dd/aaaa` em inglês), com hora opcional. Uma coluna com fórmula
(**Calc > Atribuir fórmula à coluna**, `=` no cabeçalho) é recalculada sempre que a planilha muda.

### Ainda não incluído

- Teste de Dixon, comparações de Hsu (MCB), experimentos definitive screening, análise de garantia
  e Gage R&R expandido.
- Leitura de arquivos de outros programas estatísticos.

## Linha de comando

**Ferramentas > Linha de comando** (Ctrl+L) abre um console Python ligado à planilha ativa:

```python
df['Razao'] = df['Peso'] / df['Altura']      # df é a planilha como DataFrame do pandas
analyses('capab')                             # lista análises cujo nome contém o texto
run('basic.t1', columns=['Peso'], mu0=70)     # roda uma análise e mostra o resultado na saída
```

## Plugins

Arquivos `.py` em `~/.openstat/plugins` (abra por **Ferramentas > Abrir pasta de plugins**) são
carregados na inicialização. Uma análise é uma função decorada com `@analysis`; o item de menu e o
diálogo são gerados a partir da declaração dos parâmetros:

```python
from openstat.core.registry import analysis, Param
from openstat.core.result import Result

@analysis("meu.resumo", menu=("Plugins",), title="Resumo rápido",
          params=[Param("columns", "Variáveis", "columns"),
                  Param("digits", "Casas decimais", "int", default=2, minimum=0)])
def resumo(ws, columns, digits=2):
    res = Result("Resumo rápido")
    res.table(ws.data(columns).describe().T, digits=digits)
    return res
```

Tipos de parâmetro: `columns`, `column`, `number`, `int`, `choice`, `bool`, `text`, `worksheet`.
Um `Result` aceita `text`, `heading`, `pre`, `table`, `figure` (matplotlib), `store` (nova coluna),
`formula` (fórmula de coluna) e `worksheet` (nova planilha).

## Testes

```
.venv\Scripts\python -m pytest
```

Os resultados são conferidos com SciPy, statsmodels, scikit-learn e lifelines, com os valores
certificados do NIST para o conjunto Longley, com constantes de tabelas publicadas (d2, d3, c4,
fatores de tolerância) e com exemplos de referência (Grubbs do manual NIST/SEMATECH, kappa de Fleiss).
Um teste de fumaça executa todas as análises registradas pelos diálogos gerados.

## Empacotar

```
.venv\Scripts\pip install pyinstaller
.venv\Scripts\pyinstaller packaging\openstat.spec --noconfirm
```

O executável fica em `dist\OpenStat\OpenStat.exe`.

## Estrutura

- `openstat/core` — planilha, projeto, importação, exportação, resultados, registro e avaliador de fórmulas.
- `openstat/stats` — análises; funções puras `Worksheet -> Result`, sem dependência de Qt.
- `openstat/graphs` — gráficos matplotlib.
- `openstat/ui` — janela principal, planilha, saída, console e gerador de diálogos.
- `openstat/i18n` — tabela de textos pt-BR. `tools/i18n_missing.py` lista o que falta traduzir.
