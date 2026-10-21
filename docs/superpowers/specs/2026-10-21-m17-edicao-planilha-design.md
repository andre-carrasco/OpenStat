# M17: Edição da planilha — desenho

Data: 2026-10-21
Status: aprovado pelo usuário na conversa; falta o plano (superpowers:writing-plans).
Roteiro: `2026-10-21-minitab-parity-roadmap.md`.

## Por que

A edição da planilha é a lacuna mais sentida no dia a dia. Hoje a `WorksheetView` só copia, cola, apaga com
Delete, renomeia colunas e faz formatação condicional. Não tem desfazer, recortar, localizar, inserir ou mover.
O objetivo é editar a planilha como no Minitab.

**Regra geral do usuário: quando houver dúvida de comportamento, seguir o Minitab.**

## Objetivos

1. Desfazer/Refazer, com histórico separado por planilha.
2. Menu Editar e barra de ferramentas.
3. Recortar, Excluir células e Limpar células.
4. Localizar e Substituir.
5. Inserir células, linhas e colunas; Mover colunas.
6. Propriedades de coluna: descrição e ordem dos valores. A ordem dos valores vale em todas as análises e
   gráficos.

## Fora do escopo

- Imprimir, cancelar análise, fx, atualizar gráfico e ajuda, mesmo estando na barra padrão do Minitab. Ficam
  para o M18, M19 e M24.
- Localizar na saída: só a planilha.
- Desfazer o que acontece fora da planilha: saída, planilhas novas, renomear planilha, abrir projeto.

## Não pode acontecer

- Uma operação que altera a planilha e escapa do Desfazer. Depois dela, um Ctrl+Z voltaria um estado errado.
- Desfazer apagar resultados da saída ou planilhas criadas pela análise.
- Um projeto salvo antes do M17 deixar de abrir.

## Como funciona hoje (pontos que alteram uma planilha)

| Ponto | Arquivo |
|---|---|
| Digitar, colar, Delete | `WorksheetModel.set_cells` (`openstat/ui/worksheet_view.py`) |
| Renomear coluna | `WorksheetView.rename_column` |
| Formatação condicional | `WorksheetView.set_formats` |
| Colunas e fórmulas gravadas por análise | `MainWindow.add_result` (`openstat/ui/main_window.py`) |
| Comandos do console | `Console` (`openstat/ui/console.py`, reescreve `ws.df` e chama `recalculate`) |

As operações do menu Dados (ordenar, excluir linhas, mesclar...) não alteram a planilha atual: criam uma
planilha nova por `Result.worksheet`. Por isso ficam fora do Desfazer, como as planilhas novas no Minitab.

## Abordagens consideradas

1. **Cópia da planilha antes de cada alteração, numa pilha nossa.** É simples e cobre análises e console, que
   não declaram o que mudam. Mas teríamos que escrever à mão o menu, os rótulos e a troca de pilha por aba.
2. **Comandos com operação inversa** (`QUndoStack` clássico). Gasta pouca memória, mas análises e console não
   têm inverso e precisariam de cópia do mesmo jeito: seria o dobro de código.
3. **Cópia da planilha dentro da estrutura de desfazer do Qt.** ← **escolhida pelo usuário.** Um `QUndoStack`
   por planilha; um `QUndoGroup` na `MainWindow` ativa a pilha da aba atual. Cada comando guarda o estado de
   antes e o de depois. O Qt entrega Ctrl+Z/Ctrl+Y, o texto "Desfazer <ação>", a habilitação dos botões e o
   limite de níveis.

## Desenho

### Estado da planilha

- `Worksheet` ganha `snapshot()` e `restore(state)`.
- O estado traz a ordem das colunas, `formulas`, `formats` e as novas propriedades de coluna (`descriptions`,
  `value_orders`), mais os dados das colunas.
- `formula_errors` não entra: é recalculado em `restore`.
- **Só as colunas que mudaram são guardadas** (escolha minha, por memória). Ao fim de cada passo, cada coluna é
  comparada com a de antes, e o comando guarda os valores de antes e de depois só das colunas diferentes.
  Também guarda a ordem das colunas e os dicionários, que são pequenos.
  - Exemplo: numa planilha importada de 1 milhão de linhas por 20 colunas, cópias inteiras dariam cerca de
    160 MB por passo, ou vários GB em 50 passos. Editando uma célula, o passo guarda só uma coluna (cerca de
    16 MB, antes e depois).
  - Continua sendo o caminho 3, de cópias. A diferença é que cada cópia é por coluna.

### Desfazer

- Um comando do Qt (`QUndoCommand`) chamado, por exemplo, `SnapshotCommand(ws, text, before, after)`:
  - `undo` faz `restore(before)`;
  - `redo` faz `restore(after)`;
  - os dois atualizam a grade.
- Forma de uso nos pontos de alteração: tirar o estado antes, aplicar a mudança, tirar o estado depois e
  empilhar o comando.
  - Quando a mudança falha (`ValueError`), restaurar o estado de antes e não empilhar nada.
  - Quando o estado não mudou, não empilhar.
  - O `redo` da primeira execução não reaplica nada: o estado já é o de depois.
- Uma ação do usuário vira um único passo, mesmo quando altera muitas células (colar) ou várias colunas
  (análise).
- O limite é de 50 níveis por planilha (`setUndoLimit(50)`).
- Todos os pontos da tabela acima passam por esse caminho, além das operações novas deste marco.
- O console registra um passo por comando que alterou a planilha atual.
- A `MainWindow` liga `QUndoGroup.setActiveStack` à troca de aba e cria as ações Desfazer/Refazer com
  `createUndoAction`/`createRedoAction`.
- `mark_dirty` continua funcionando: desfazer também marca o projeto como alterado.
- Abrir, criar ou carregar um projeto começa com as pilhas vazias. O histórico não é salvo no `.ostat`.

### Editar: Excluir × Limpar × Recortar (como no Minitab)

- **Excluir células** (tecla Delete): remove as células selecionadas e sobe as de baixo, coluna por coluna.
  ⚠️ Isso muda o comportamento de hoje, em que o Delete só esvazia as células.
  - Quando a seleção são **colunas inteiras** (pelo cabeçalho), Excluir **remove as colunas**: as da direita
    vêm para a esquerda, e a fórmula, os formatos, a descrição e a ordem dos valores vão embora junto. É o que o
    Minitab faz.
  - Quando a seleção são **linhas inteiras**, Excluir remove as linhas.
- **Limpar células**: deixa as células selecionadas em branco (o Delete de hoje).
- **Recortar** (Ctrl+X): copia e depois exclui, num único passo de desfazer.
- **Copiar e colar** continuam como estão; colar passa a ser um passo de desfazer.
- **Selecionar tudo** (Ctrl+A): o padrão do `QTableView`, também no menu.
- Uma coluna com fórmula que recebe exclusão ou inserção: segue o que a `Worksheet` já faz com edição manual
  dessas colunas. Escolha do construtor, desde que seja igual para digitar, excluir e inserir.

### Inserir e mover

- **Inserir células**: insere células vazias acima da seleção, nas colunas selecionadas, e empurra as de baixo.
- **Inserir linhas**: insere linhas inteiras acima da seleção, tantas quantas estão selecionadas.
- **Inserir colunas**: insere colunas vazias à esquerda da seleção. Os nomes seguem `_unique_name`.
- **Mover colunas...**: um diálogo para escolher as colunas e o destino (início, fim, antes de uma coluna ou
  depois de uma coluna). As colunas mantêm a ordem relativa.
- Fórmulas e formatos acompanham o nome da coluna, então mover e inserir não os quebram.

### Localizar e Substituir

- Dois diálogos não modais (Ctrl+F e Ctrl+H), ou um com duas abas; escolha do construtor.
- Campos: localizar; substituir por (só no Substituir).
- Opções: diferenciar maiúsculas; célula inteira; só nas colunas selecionadas.
- A busca é feita no texto exibido (`get_text`), linha por linha, a partir da célula atual, e volta ao início
  quando chega ao fim.
- **Localizar próximo** seleciona a célula encontrada. **Substituir** troca a atual e avança.
  **Substituir tudo** faz um único passo de desfazer e informa quantas células trocou.
- O valor substituído passa por `set_cell` (com `parse_cell`), igual a digitar.

### Propriedades de coluna

- Diálogo **Propriedades da coluna...**, aberto pelo menu Editar e pelo menu do cabeçalho, com:
  - **Descrição**: texto livre. Aparece na dica do cabeçalho, junto com a fórmula se houver uma.
  - **Ordem dos valores**, só em colunas de texto e de data:
    - alfabética (padrão, o comportamento de hoje);
    - ordem em que aparecem na planilha;
    - definida pelo usuário: uma lista editável dos níveis, com botões para subir e descer.
- Ficam guardadas em `ws.descriptions: dict[str, str]` e `ws.value_orders: dict[str, list[str] | str]`.
  Acompanham `rename_column` como `formulas` e `formats`.
- No `.ostat`, entram em `manifest["descriptions"]` e `manifest["value_orders"]`. Arquivos antigos, sem essas
  chaves, abrem com os dicionários vazios, como já acontece com `formats`. `FORMAT_VERSION` continua 1, porque
  os campos são opcionais.
- **Ordem dos valores em todas as análises e gráficos**:
  - Uma função única, `levels(ws, column, values=None) -> list[str]`, junto da `Worksheet` (ou do
    `factor_groups`), devolve os níveis na ordem da coluna.
  - Níveis que surgirem depois de definida a ordem do usuário vão para o fim, em ordem alfabética.
  - Todos os pontos que hoje ordenam níveis por conta própria passam a usá-la: `sorted(... .unique())`,
    `groupby(..., sort=True)` e `np.unique` sobre colunas de categoria.
  - O levantamento inicial encontrou uns 40 pontos em `stats/` e `graphs/`, entre eles `anova.py`, `basic.py`,
    `data.py`, `doe.py`, `gage.py`, `quality.py`, `regression.py`, `reliability.py`, `nonparametric.py`,
    `equivalence.py`, `calc.py`, `_design.py`, `_life.py`, `_mars.py`, `graphs/basic.py`, `graphs/more.py` e
    `graphs/relation.py`.
  - Só trocar onde a ordem é de **níveis de uma coluna da planilha**. `np.unique` usado em cálculo numérico
    fica como está.
  - Colunas numéricas usadas como fator mantêm a ordem numérica. Colunas de data, sem ordem definida pelo
    usuário, ficam em ordem cronológica, como no Minitab. Hoje elas são ordenadas pelo texto exibido, o que
    erra em datas dd/mm/aaaa.
  - **A ordem dos valores muda resultados, não só a ordem das linhas**, como no Minitab:
    - o evento da regressão logística binária e do gráfico de linha ajustada binário é o último nível da ordem
      (`levels[-1]` em `regression.py`);
    - a ordem da resposta ordinal (`_category_codes`) segue a ordem dos valores;
    - o nível de referência dos preditores categóricos (`get_dummies` em `predictive.py`, `doe.py`) é o
      primeiro nível da ordem.
    - Onde isso acontece, a saída mostra uma linha "Evento: X" ou "Nível de referência: X", para quem
      reordenou uma coluna entender por que os coeficientes mudaram.
    - Há testes que reordenam a coluna e conferem que o evento ou a referência mudou.
  - Funções auxiliares que recebem só um DataFrame passam a receber também os níveis (ou o `ws`) de quem as
    chama.

### Menu Editar e barra de ferramentas

- Menu **Editar**, entre Arquivo e Assistente:
  1. Desfazer (Ctrl+Z), Refazer (Ctrl+Y)
  2. Recortar (Ctrl+X), Copiar (Ctrl+C), Colar (Ctrl+V)
  3. Excluir células (Delete), Limpar células, Selecionar tudo (Ctrl+A)
  4. Localizar... (Ctrl+F), Substituir... (Ctrl+H)
  5. Inserir células, Inserir linhas, Inserir colunas, Mover colunas...
  6. Propriedades da coluna...
- Os atalhos de teclado que já existem na grade (copiar e colar) passam a usar as mesmas ações, sem duplicar
  código.
- **Barra de ferramentas** padrão, com ícones do estilo do Qt (`QStyle.standardIcon` ou
  `QIcon.fromTheme`), na ordem: abrir, salvar | recortar, copiar, colar | desfazer, refazer | localizar,
  substituir | inserir células, linhas, colunas, mover colunas.
- O menu do cabeçalho de coluna ganha: Inserir colunas, Mover colunas... e Propriedades da coluna...
- Os textos novos entram em `openstat/i18n/en.json` e `pt_BR.json`, como os demais.

## Testes

- Seguir o padrão de `tests/test_app.py` (fixture `window` com `qapp`).
- Testes da `Worksheet` sem Qt: `snapshot`/`restore`, inserir, excluir, mover e `levels`.
- Para cada ponto de alteração, verificar que um Ctrl+Z volta exatamente o estado anterior e que um Ctrl+Y
  refaz: digitar, colar, excluir, limpar, recortar, renomear, formatar, propriedades, inserir, mover,
  substituir tudo, análise que grava coluna e comando do console.
- Desfazer uma análise mantém a saída e as planilhas novas.
- Histórico por aba: um Ctrl+Z na aba B não mexe na aba A.
- A ordem dos valores definida pelo usuário aparece na ordem das tabelas de pelo menos uma ANOVA, um gráfico
  por grupo e uma tabela cruzada.
- Um projeto salvo com as propriedades novas abre com elas; um `.ostat` sem as chaves abre normalmente.

## Decisões

**Do usuário**
- Seguir o Minitab em toda escolha de comportamento.
- O Desfazer cobre o que as análises gravam nas colunas.
- Abordagem 3: cópia da planilha com `QUndoStack` e `QUndoGroup`.
- Delete exclui e sobe as células, como no Minitab.
- A ordem dos valores vale em todas as análises e gráficos.

**Escolhas minhas, seguindo o Minitab**
- O histórico é separado por planilha.
- Desfazer uma análise mantém a saída e as planilhas novas.
- O limite é de 50 níveis.
- O histórico não é salvo no projeto.
- A lista exata do menu e da barra é a definida acima.
- Os campos novos no manifesto são opcionais, sem trocar a versão.
- O Desfazer guarda só as colunas que mudaram, por causa da memória em planilhas grandes.
- Excluir com colunas ou linhas inteiras selecionadas remove as colunas ou as linhas.
- Datas ficam em ordem cronológica por padrão.
- A ordem dos valores também define o evento, a ordem da resposta ordinal e o nível de referência, e a saída
  mostra qual foi usado.

## Fica para o construtor

- Nomes das classes, das funções e dos módulos novos, se `SnapshotCommand` não ficar bom.
- Um diálogo ou dois para Localizar e Substituir.
- Escolha dos ícones.
- Divisão do trabalho em tarefas e a ordem delas.
- Depois de mudar código, rodar `graphify update .`.
