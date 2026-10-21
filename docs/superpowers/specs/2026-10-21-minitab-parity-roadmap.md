# Roteiro: paridade com o Minitab

Data: 2026-10-21
Status: roteiro aprovado em linhas gerais; cada fase ainda passa por desenho e plano próprios
(superpowers:brainstorming → superpowers:writing-plans) antes de virar código.

## Por que

O OpenStat foi construído antes de usarmos o superpowers e o graphify. Comparamos o projeto com a
[lista oficial de recursos do Minitab](https://www.minitab.com/en-us/products/minitab/features/) e com as
novidades do Minitab 22 (22.0 a 22.5.1) para decidir o que acrescentar. A conclusão é que **o conteúdo
estatístico está quase todo coberto (253 análises); a distância maior está no software em volta das análises**:
edição da planilha, execução, saída e gráficos.

## Como o OpenStat funciona hoje

- `openstat/core/registry.py`: cada análise é declarada com `@analysis`; menus e diálogos
  (`ui/dialog_builder.py`) são gerados a partir dessas declarações.
- `openstat/core/result.py`: uma análise devolve um `Result` com texto, tabelas, figuras matplotlib,
  colunas a gravar (`store`), fórmulas e planilhas novas. As figuras viram PNG base64 em `to_html`.
- `openstat/ui/main_window.py`:
  - `run_analysis` abre o diálogo e roda a análise de forma síncrona, dentro de `_guarded`, sem thread.
  - `add_result` grava colunas e fórmulas na planilha, cria planilhas novas e acrescenta o HTML à saída.
  - `record` guarda cada análise rodada como comando de console, que é a base de "Salvar sessão como script".
- `openstat/ui/output_view.py`: a saída inteira é um documento HTML só, com o Navegador à esquerda.
- `openstat/ui/worksheet_view.py`: um `QTableView` sobre `Worksheet`. Tem copiar e colar pelo teclado,
  apagar com Delete, renomear coluna e formatação condicional pelo cabeçalho. Não tem desfazer, recortar,
  localizar, inserir ou mover.
- `openstat/core/worksheet.py`: colunas numéricas, texto ou data/hora; fórmulas vinculadas que se recalculam.
- Também existem: Assistente (árvore de decisão para escolher a análise), Construtor de gráficos,
  console Python, plugins, importação de arquivo e de banco de dados, exportação para HTML/PDF/DOCX, e i18n
  pt-BR/en com `QSettings` guardando só o idioma.

## Lacunas em relação ao Minitab (conferidas no código)

**Software em volta das análises**
- Desfazer/Refazer, menu Editar e barra de ferramentas. A referência é a barra padrão do Minitab: abrir,
  salvar, imprimir, recortar/copiar/colar, desfazer/refazer, localizar, localizar e substituir, cancelar,
  ajuda, fx, inserir células/linhas/colunas, mover colunas, atualizar gráfico e limpar formatação.
- Recortar, Localizar/Substituir, inserir e mover linhas, colunas e células pela interface.
- Propriedades de coluna: descrição e ordem dos valores.
- Preferências e perfis de usuário; menus e barras de ferramentas personalizáveis.
- Impressão.
- Ajuda por análise: o Minitab tem guia de interpretação e "Métodos e fórmulas"; o OpenStat só tem "Sobre".

**Execução**
- A análise trava a janela e não pode ser cancelada.
- Não tem "Editar última janela" (Ctrl+E).

**Saída e gráficos**
- Gráficos são PNG fixos.
- Faltam:
  - atualização automática quando os dados mudam e saída marcada como desatualizada;
  - brushing;
  - edição interativa (linhas de grade, linhas de referência, mostrar valores);
  - 3D girável;
  - exportação de gráfico em TIF, JPEG, PNG, BMP, GIF e EMF;
  - menu de botão direito por resultado (copiar como imagem, enviar para Word, editar a análise).

**Qualidade**
- Cartas de controle por estágios (histórico e mudança de processo).
- Capacidade em lotes (batch), não paramétrica e automática (Minitab 22).
- EMP cruzado (Evaluate Measurement Process, método de Wheeler).
- Cartas de controle interativas (22.5).

**DOE**
- Gráfico de efeitos meio-normal (o normal e o de Pareto já existem).
- Corridas perdidas (botched runs).
- Análise de variabilidade para fatoriais.
- Planos baseados em distância.
- Plano definido pelo usuário (Define Custom Design).

**Confiabilidade**
- Distribuições com parâmetro de limiar, de 3 parâmetros (`stats/distfit.py` hoje fixa `floc=0`).
- Análise Weibayes.
- Distribuição não paramétrica com censura arbitrária (Turnbull) e o gráfico de identificação de
  distribuição para censura arbitrária.

**Básica e regressão**
- Intervalos de Wilson e de Agresti-Coull para proporções.
- Stepwise com critério AICc e BIC.
- Otimizador de resposta para modelos de regressão e preditivos (hoje existe só para DOE).

**Assistente**
- O do Minitab roda a análise e entrega relatório guiado com cartões de diagnóstico para MSA, capacidade,
  gráficos, testes de hipótese, regressão, DOE e cartas de controle.
- O do OpenStat só escolhe a análise.

**Tabelas e gráficos**
- Construtor interativo de tabelas.
- Estatísticas tabuladas e Pareto dentro do Construtor de gráficos.

**Integração**
- Integração com R. A integração com Python já existe pelo console.

**Fora do escopo:** os Módulos de Soluções e o Minitab Workspace (mapa de fluxo de valor, FMEA, roteiro de
projeto). Seriam um produto separado.

## Fases

Os marcos continuam a numeração do histórico do git (o último é M16).

| Fase | Conteúdo | Por que nessa ordem |
|---|---|---|
| **M17: Edição da planilha** | Desfazer/Refazer, menu Editar, barra de ferramentas, recortar, localizar/substituir, inserir e mover, propriedades de coluna | Mais sentida no dia a dia; o desenho do Desfazer já começou (ver abaixo) |
| **M18: Execução** | Análise em segundo plano com cancelar, Ctrl+E, preferências | Pré-requisito do M19: atualizar sozinho exige rodar sem travar a tela |
| **M19: Saída viva** | Cada resultado guarda a análise e os parâmetros que o geraram (reaproveitando `record`), atualização automática e "desatualizado", exportação em vários formatos, menu de botão direito, edição básica, brushing, 3D girável | Muda o modelo da saída de "HTML pronto" para "resultado reexecutável" |
| **M20: Qualidade** | Estágios, capacidade em lotes, não paramétrica e automática, EMP, cartas interativas | Área mais forte do projeto |
| **M21: DOE e confiabilidade** | Itens das listas de DOE e Confiabilidade | Análises isoladas no padrão do registro |
| **M22: Básica e regressão** | Wilson e Agresti-Coull, AICc e BIC, otimizador geral | Mudanças pequenas |
| **M23: Assistente com relatórios** | Relatórios guiados com cartões de diagnóstico | Depende dos gráficos vivos do M19 |
| **M24: Personalização** | Menus e barras configuráveis, perfis, ponte com R, ajuda por análise | Menor impacto |

As fases M20 a M22 não dependem umas das outras e podem trocar de ordem.

## Decisões já tomadas (M17)

- **O Desfazer cobre também o que uma análise grava nas colunas**, não só as edições manuais
  (decisão do usuário).
- Propostas ainda **não confirmadas** pelo usuário:
  - Guardar uma cópia da planilha antes de cada alteração, em vez de registrar diferenças. O motivo é que as
    análises não declaram o que mudam, e algumas operações de Dados alteram a planilha diretamente. O custo
    é memória em planilhas grandes.
  - Histórico separado por planilha, como no Minitab.
  - Desfazer uma análise volta as colunas, mas mantém o resultado na saída e as planilhas novas que ela
    criou, como no Minitab.

## Próximo passo

Na nova sessão: abrir o brainstorming do M17 a partir deste documento. A primeira pergunta pendente é se o
resultado de uma análise desfeita continua no painel de saída ou some junto.
