# OpenStat

Aplicativo desktop de análise estatística, de código aberto (MIT), escrito em Python com PySide6.

É uma implementação independente: os cálculos usam métodos estatísticos públicos e as bibliotecas
NumPy, SciPy, statsmodels, scikit-learn, lifelines, pyDOE3 e matplotlib.

## Baixar

Para Windows, sem instalar Python: baixe o [OpenStat-0.2.0-windows.zip](https://github.com/andre-carrasco/OpenStat/releases/download/v0.2.0/OpenStat-0.2.0-windows.zip)
(todas as versões em [Releases](https://github.com/andre-carrasco/OpenStat/releases)), extraia e abra `OpenStat.exe`.
O executável não é assinado, então o Windows pode pedir confirmação na primeira execução.

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
| Calc | calculadora de fórmulas, fórmulas vinculadas a colunas, estatísticas de colunas e linhas, dados padronizados, em malha e aleatórios, variáveis indicadoras, matrizes, distribuições de probabilidade, reamostragem (bootstrap e testes de aleatorização) |
| Estatística básica | descritiva, resumo gráfico, Z e t, t pareado, proporções, taxas de Poisson, variâncias, correlação, covariância, normalidade, outlier (Grubbs, Q de Dixon), ajuste de Poisson, armazenar estatísticas descritivas |
| Regressão | linear com diagnósticos, predição com intervalos, estudo de estabilidade (prazo de validade), linha ajustada (inclusive binária), stepwise, melhores subconjuntos, não linear, logística binária/ordinal/nominal, Poisson, PLS, ortogonal |
| ANOVA | um fator (Tukey, Fisher, Dunnett, Games-Howell, MCB de Hsu, Welch), análise de médias, ANOVA balanceada com fatores fixos e aleatórios, modelo linear geral, igualdade de variâncias, aninhada, efeitos mistos, MANOVA, efeitos principais e interação |
| DOE | fatorial completo e fracionado, Plackett-Burman, fatorial geral, definitive screening, split-plot, composto central, Box-Behnken, seleção D-ótima, misturas (simplex e vértices extremos), Taguchi; análise (inclusive resposta binária), gráficos de contorno, superfície e contorno sobreposto, otimizador com meta por resposta, predição Taguchi, modificar experimento |
| Cartas de controle | Xbarra-R, Xbarra-S, Xbarra, R, S, I-AM, I-AM-R/S (entre/dentro), Z-AM, zonas, P, NP, C, U, Laney, EWMA, CUSUM, média móvel, T², T²-variância generalizada, variância generalizada, EWMA multivariada, G, T, Box-Cox |
| Ferramentas da qualidade | gráfico de sequência, Pareto, causa e efeito, identificação de distribuição, Johnson, gráfico de simetria, multi-vari, gráfico de variabilidade, capacidade (normal, entre/dentro, múltiplas variáveis, não normal, binomial, Poisson, sixpack), Gage R&R (cruzado, aninhado, expandido), gráfico de corridas de medição, planilha de estudo de Gage R&R, estudo de medição por atributos (método analítico), intervalos de tolerância (normal, lognormal, unilaterais, não paramétricos), amostragem de aceitação por atributos e por variáveis |
| Confiabilidade | análise paramétrica com censura à direita e arbitrária, Kaplan-Meier, modos de falha múltiplos, regressão com dados de vida, regressão de Cox, teste de vida acelerado, probit, curvas de crescimento paramétrica e não paramétrica, análise de garantia (pré-processamento e predição), planos de teste de demonstração e de estimação |
| Multivariada | componentes principais, análise fatorial, itens, clusters, discriminante, correspondência |
| Séries temporais | tendência, decomposição, média móvel, suavização exponencial, Winters, diferenças, defasagens, ACF, PACF, CCF, Dickey-Fuller aumentado, ARIMA, previsão com o melhor ARIMA (AICc) |
| Tabelas | contagem, tabulação cruzada com medidas ordinais e kappa, testes qui-quadrado, McNemar, Cochran-Mantel-Haenszel, tabela de estatísticas descritivas |
| Não paramétricos | sinais, Wilcoxon, Mann-Whitney, Kruskal-Wallis, Mood, Friedman, corridas, médias, diferenças e inclinações pareadas |
| Equivalência | 1 amostra, 2 amostras, pareado, crossover 2x2 |
| Poder e tamanho de amostra | testes Z e t, proporções, taxas de Poisson, variâncias, ANOVA, fatorial de 2 níveis, Plackett-Burman, fatorial completo geral, equivalência (inclusive crossover 2x2), estimação, tolerância |
| Análise preditiva | CART, random forest e gradient boosting (classificação e regressão), regressão MARS, validação cruzada e conjunto de teste, comparação automática de modelos, descoberta de preditores-chave |
| Gráfico | 24 tipos, de histograma a superfície 3D, incluindo dispersão por faixas, coordenadas paralelas e distribuição de probabilidade |

Projetos são salvos em `.ostat` (um zip com as planilhas em Parquet e a saída em HTML). Dados entram
por CSV, TXT ou Excel, ou colados da área de transferência. A saída pode ser exportada em HTML, PDF
ou Word.

**Assistente** guia a escolha da análise por perguntas e abre o diálogo certo; o **Construtor de
gráficos** sugere tipos de gráfico a partir das colunas escolhidas. Na planilha, o menu do cabeçalho
da coluna aplica **formatação condicional** (maior, menor, igual, faltante, atípico). **Arquivo >
Importar de banco de dados** lê uma consulta SQL de um arquivo SQLite ou de uma conexão ODBC
(`pip install openstat[odbc]`). Itens selecionados no navegador podem ser exportados à parte, e
notas podem ser acrescentadas à saída. **Ferramentas > Salvar sessão como script** grava as análises
executadas como comandos `run(...)`, que **Executar script** repete.

As colunas são numéricas, de texto (`-T` no cabeçalho) ou de data/hora (`-D`). Datas são digitadas
como `aaaa-mm-dd` ou `dd/mm/aaaa` (`mm/dd/aaaa` em inglês), com hora opcional. Uma coluna com fórmula
(**Calc > Atribuir fórmula à coluna**, `=` no cabeçalho) é recalculada sempre que a planilha muda.

### Ainda não incluído

- Razões de Dixon além de r10, análise de médias para dados binomiais e de Poisson, diagnóstico das
  cartas P e U, Gage R&R expandido desbalanceado ou com fatores fixos, planos de teste de vida
  acelerado.
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
