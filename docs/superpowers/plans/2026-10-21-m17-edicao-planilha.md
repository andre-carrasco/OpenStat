# M17: Edição da planilha — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** dar à planilha do OpenStat Desfazer/Refazer, menu Editar, barra de ferramentas, recortar, excluir, limpar,
localizar/substituir, inserir e mover, além de propriedades de coluna, com a ordem dos valores valendo em todas as
análises e gráficos. Tudo como no Minitab.

**Architecture:**
- A `Worksheet` ganha operações de estrutura (inserir, excluir, mover), propriedades (`descriptions`,
  `value_orders`), `levels()` e `snapshot()`/`restore()`.
- Na interface, cada `WorksheetView` tem um `QUndoStack`, e a `MainWindow` tem um `QUndoGroup` que acompanha a aba
  ativa.
- Toda alteração passa por `WorksheetView.edit(text, action)`. Ela tira o estado antes e depois, reduz os dois às
  colunas que mudaram e empilha um `SnapshotCommand`.

**Tech Stack:** Python 3.11+, PySide6 (`QUndoStack`, `QUndoGroup`, `QUndoCommand`), pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-10-21-m17-edicao-planilha-design.md`

## Global Constraints

- Em toda escolha de comportamento que o spec não fixa, seguir o Minitab.
- Limite de **50** níveis de desfazer por planilha (`setUndoLimit(50)`).
- Uma ação do usuário é **um único** passo de desfazer.
- Mudança que falha: a planilha volta ao estado de antes e nenhum passo é empilhado.
- Mudança que não alterou nada: nenhum passo é empilhado.
- Desfazer nunca remove itens da saída nem planilhas criadas.
- O histórico não é salvo no `.ostat`.
- `FORMAT_VERSION` continua `1`. As chaves novas do manifesto são opcionais, e arquivos sem elas abrem.
- Todo texto novo de interface passa por `tr()` e ganha entrada em `openstat/i18n/pt_BR.json` e em `en.json`.
  `python tools/i18n_missing.py` não pode listar nenhum texto novo.
- Comando de testes: `.venv/Scripts/python -m pytest -q`. A base hoje é **491 passed**.
- Depois de cada tarefa que muda código: `graphify update .`.
- Trabalhar no ramo `m17-edicao-planilha`, criado a partir de `main`.

## Review Focus

1. **Digitar além do fim dos dados e desfazer.** O `ensure_size` cresce linhas e colunas. Um Ctrl+Z tem que devolver
   a planilha ao tamanho original (`nrows` e `ncols` de antes). Teste na Tarefa 4.
2. **Análise que falha no meio de `add_result`** (`add_column` funcionou, `set_formula` falhou). Não pode sobrar
   coluna pela metade nem passo de desfazer. Teste na Tarefa 4.
3. **Renomear uma coluna com fórmula, formato e descrição, e desfazer.** O nome volta, e a fórmula, o formato e a
   descrição voltam com ele. Teste na Tarefa 1.
4. **Ordem definida pelo usuário, e depois a coluna ganha um valor novo ou perde um nível.** O valor novo vai para o
   fim, em ordem alfabética; o nível sumido não aparece. Teste na Tarefa 3.
5. **Console que altera o `df` no lugar** (`df.loc[0, 'A'] = 99`). Tem que ser desfeito, mesmo sem reatribuir
   `df`. Teste na Tarefa 4.

---

### Task 1: Estado da planilha, propriedades de coluna e gravação no projeto

**Files:**
- Modify: `openstat/core/worksheet.py` (classe `Worksheet`, mais uma função de módulo)
- Modify: `openstat/core/project.py` (`save`, `load`)
- Test: `tests/test_editing.py` (novo)

**Interfaces:**
- Produces:
  - `Worksheet.descriptions: dict[str, str]`
  - `Worksheet.value_orders: dict[str, str | list[str]]`. O valor é `"occurrence"` ou a lista de níveis definida
    pelo usuário. A ausência da chave significa ordem padrão.
  - `Worksheet.snapshot() -> dict`, com as chaves `columns` (lista de nomes), `nrows`, `data` (nome →
    `pd.Series` copiada), `formulas`, `formats`, `descriptions` e `value_orders` (estas quatro como cópias profundas).
  - `changed_parts(before: dict, after: dict) -> tuple[dict, dict]` (função de módulo em `worksheet.py`): os mesmos
    dicionários, mas `data` mantém só as colunas que diferem. Uma coluna difere quando o nome falta no outro estado
    ou quando `Series.equals` dá falso.
  - `Worksheet.restore(state: dict) -> None`: reconstrói `df` na ordem de `state["columns"]` com `state["nrows"]`
    linhas. Cada coluna vem de `state["data"]` quando está lá; senão, mantém os valores atuais. Repõe os quatro
    dicionários e chama `recalculate()`.

- [ ] **Step 0: Criar o ramo e incluir o spec e o plano**

```bash
git checkout -b m17-edicao-planilha
git add docs/superpowers
git commit -m "docs: M17 design and plan"
```

- [ ] **Step 1: Escrever os testes que falham** em `tests/test_editing.py`:

```python
def ws_ab():
    return Worksheet("W", pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": ["x", "y", "x"]}))

def test_snapshot_restore_roundtrip():
    ws = ws_ab(); before = ws.snapshot()
    ws.set_cell(5, 3, "7"); ws.formats["A"] = [{"rule": "x"}]; ws.descriptions["A"] = "d"
    ws.restore(before)
    assert ws.column_names() == ["A", "B"] and ws.nrows == 3 and ws.formats == {} and ws.descriptions == {}

def test_changed_parts_keeps_only_changed_columns():
    ws = ws_ab(); before = ws.snapshot(); ws.set_cell(0, 0, "9"); after = ws.snapshot()
    b, a = changed_parts(before, after)
    assert set(b["data"]) == {"A"} and set(a["data"]) == {"A"}
    ws.restore(b); assert ws.df["A"].tolist() == [1.0, 2.0, 3.0] and ws.df["B"].tolist() == ["x", "y", "x"]
    ws.restore(a); assert ws.df["A"].tolist() == [9.0, 2.0, 3.0]

def test_rename_carries_properties_and_restores():          # Review Focus 3
    ws = ws_ab(); ws.set_formula("F", "A * 2"); ws.descriptions["A"] = "d"; ws.value_orders["B"] = ["y", "x"]
    before = ws.snapshot(); ws.rename_column(1, "Bee")
    assert ws.value_orders == {"Bee": ["y", "x"]}
    ws.restore(changed_parts(before, ws.snapshot())[0])
    assert ws.column_names() == ["A", "B", "F"] and ws.value_orders == {"B": ["y", "x"]} and ws.formulas == {"F": "A * 2"}

def test_project_saves_descriptions_and_value_orders(tmp_path):
    p = Project([ws_ab()]); p.worksheets[0].descriptions["A"] = "d"; p.worksheets[0].value_orders["B"] = "occurrence"
    p.save(tmp_path / "p.ostat"); ws = Project.load(tmp_path / "p.ostat").worksheets[0]
    assert ws.descriptions == {"A": "d"} and ws.value_orders == {"B": "occurrence"}

def test_old_project_without_new_keys_opens(tmp_path):
    # write a .ostat, then drop "descriptions"/"value_orders" from manifest.json and reload
    ...
    assert ws.descriptions == {} and ws.value_orders == {}
```

- [ ] **Step 2: Rodar e ver falhar.** `.venv/Scripts/python -m pytest tests/test_editing.py -q`. Esperado: FAIL
  (`AttributeError: snapshot` / `ImportError: changed_parts`).
- [ ] **Step 3: Implementar.**
  - Os dois dicionários novos são criados em `__init__`.
  - `rename_column` leva `descriptions` e `value_orders` junto, como já faz com `formats`.
  - Em `project.py`, `save` grava `manifest["descriptions"]` e `manifest["value_orders"]` (uma lista por planilha);
    `load` lê com `manifest.get(..., [])`, como `formats`.
  - Em `restore`, montar o `DataFrame` com `index=range(nrows)` e usar `.to_numpy()` de cada `Series`, para não
    alinhar por índice.
- [ ] **Step 4: Rodar e ver passar.** Mesmo comando; esperado PASS.
- [ ] **Step 5: Commit.** `git commit -am "M17: worksheet snapshot, column properties"` (adicionar
  `tests/test_editing.py`).

### Task 2: Operações de estrutura na Worksheet

**Files:**
- Modify: `openstat/core/worksheet.py`
- Test: `tests/test_editing.py`

**Interfaces:**
- Produces (todos os métodos são de `Worksheet`; `cells` é sempre um iterável de `(row, col)` com índices de grade a
  partir de 0, e o que cai fora dos dados é ignorado):
  - `delete_cells(cells) -> None`: em cada coluna, remove as linhas selecionadas e sobe as de baixo. O número de
    linhas da planilha não muda, porque o fim da coluna fica em branco.
  - `clear_cells(cells) -> None`: deixa as células em branco, com uma única chamada a `recalculate`.
  - `insert_cells(cells) -> None`: em cada coluna da seleção, insere `k` brancos na menor linha selecionada daquela
    coluna (`k` é quantas linhas foram selecionadas nela). A planilha cresce se for preciso.
  - `insert_rows(row: int, count: int) -> None`: insere linhas em branco antes de `row`.
  - `delete_rows(rows: list[int]) -> None`: remove as linhas inteiras.
  - `insert_columns(col: int, count: int) -> None`: insere colunas numéricas vazias antes de `col`, com nomes de
    `_unique_name(f"C{n}")`.
  - `delete_columns(cols: list[int]) -> None`: remove as colunas e, junto, `formulas`, `formats`, `descriptions` e
    `value_orders` delas.
  - `move_columns(names: list[str], where: str, anchor: str | None = None) -> None`: `where` é um de
    `"start"`, `"end"`, `"before"`, `"after"`. `anchor` é obrigatório para `before`/`after` e não pode estar em
    `names`; se estiver, levantar `ValueError("Choose a column that is not being moved")`. As colunas movidas mantêm a
    ordem relativa.
- Depois de mexer nos valores de uma coluna, recompor o tipo dela com `coerce_values`. Por exemplo, uma coluna de
  texto que perdeu a única célula de texto volta a ser numérica, como já acontece em `set_cell`. Terminar com
  `recalculate()`.

- [ ] **Step 1: Escrever os testes que falham**:

```python
def test_delete_cells_shifts_up_per_column():
    ws = ws_ab(); ws.delete_cells([(0, 0)])
    assert ws.df["A"].tolist()[:2] == [2.0, 3.0] and np.isnan(ws.df["A"][2]) and ws.df["B"].tolist() == ["x", "y", "x"]

def test_clear_cells_leaves_blanks():
    ws = ws_ab(); ws.clear_cells([(1, 0), (1, 1)])
    assert np.isnan(ws.df["A"][1]) and ws.df["B"][1] is None and ws.nrows == 3

def test_insert_cells_pushes_down():
    ws = ws_ab(); ws.insert_cells([(0, 0), (1, 0)])
    assert ws.nrows == 5 and ws.df["A"].tolist()[2:] == [1.0, 2.0, 3.0]

def test_insert_and_delete_rows_and_columns():
    ws = ws_ab(); ws.insert_rows(1, 2); assert ws.nrows == 5 and ws.df["A"].tolist()[3:] == [2.0, 3.0]
    ws.delete_rows([1, 2]); assert ws.df["A"].tolist() == [1.0, 2.0, 3.0]
    ws.insert_columns(1, 1); assert ws.column_names()[0] == "A" and ws.column_names()[2] == "B"
    ws.descriptions["B"] = "d"; ws.delete_columns([2]); assert "B" not in ws.column_names() and ws.descriptions == {}

def test_move_columns():
    ws = Worksheet("W", pd.DataFrame({c: [1.0] for c in "ABCD"}))
    ws.move_columns(["D", "B"], "start"); assert ws.column_names() == ["B", "D", "A", "C"]
    ws.move_columns(["B"], "after", "C"); assert ws.column_names() == ["D", "A", "C", "B"]
    with pytest.raises(ValueError): ws.move_columns(["A"], "before", "A")

def test_text_column_becomes_numeric_after_deleting_its_text():
    ws = Worksheet("W", pd.DataFrame({"A": ["a", "2", "3"]})); ws.delete_cells([(0, 0)])
    assert ws.is_numeric("A")
```

- [ ] **Step 2: Rodar e ver falhar.** `.venv/Scripts/python -m pytest tests/test_editing.py -q`. Esperado: FAIL
  (`AttributeError`).
- [ ] **Step 3: Implementar** os sete métodos acima.
- [ ] **Step 4: Rodar e ver passar.**
- [ ] **Step 5: Commit.** `git commit -am "M17: insert, delete, clear and move in the worksheet"`.

### Task 3: `levels()`, a ordem dos valores

**Files:**
- Modify: `openstat/core/worksheet.py`
- Test: `tests/test_editing.py`

**Interfaces:**
- Produces: `Worksheet.levels(name: str, labels: pd.Series | None = None) -> list[str]`. Devolve os valores
  distintos já formatados (`format_value`) de `labels`, ou da coluna `name` sem os ausentes quando `labels` é
  `None`, na ordem dos valores da coluna:
  - `value_orders[name]` é uma lista: os níveis presentes na ordem da lista, e depois os que não estão nela, em ordem
    alfabética.
  - `"occurrence"`: a ordem da primeira aparição em `ws.df[name]`.
  - Sem ordem definida:
    - coluna numérica: ordem numérica (`key=float`);
    - coluna de data: ordem cronológica;
    - texto: `sorted`.

- [ ] **Step 1: Escrever os testes que falham**:

```python
def test_levels_default_orders():
    ws = Worksheet("W", pd.DataFrame({"N": [10.0, 2.0, 1.0], "T": ["b", "a", "c"],
                                      "D": pd.to_datetime(["2024-02-01", "2023-12-31", "2024-01-15"])}))
    assert ws.levels("N") == ["1", "2", "10"] and ws.levels("T") == ["a", "b", "c"]
    assert ws.levels("D") == ["2023-12-31", "2024-01-15", "2024-02-01"]

def test_levels_occurrence_and_user_order():                  # Review Focus 4
    ws = Worksheet("W", pd.DataFrame({"T": ["b", "a", "c", "a"]}))
    ws.value_orders["T"] = "occurrence"; assert ws.levels("T") == ["b", "a", "c"]
    ws.value_orders["T"] = ["c", "z", "a"]; assert ws.levels("T") == ["c", "a", "b"]   # z missing, b new at end
    assert ws.levels("T", pd.Series(["a", "c"])) == ["c", "a"]
```

- [ ] **Step 2: Rodar e ver falhar.** Esperado: FAIL (`AttributeError: levels`).
- [ ] **Step 3: Implementar** `levels`.
- [ ] **Step 4: Rodar e ver passar.**
- [ ] **Step 5: Commit.** `git commit -am "M17: value order of a column"`.

### Task 4: Desfazer/Refazer na interface, ligado a todos os pontos que alteram a planilha

**Files:**
- Create: `openstat/ui/undo.py`
- Modify: `openstat/ui/worksheet_view.py`, `openstat/ui/main_window.py`, `openstat/ui/console.py`,
  `openstat/i18n/pt_BR.json`, `openstat/i18n/en.json`
- Test: `tests/test_editing_app.py` (novo, usa a fixture `qapp` e uma fixture `window` igual à de `tests/test_app.py`)

**Interfaces:**
- Consumes: `snapshot`, `changed_parts` e `restore` (Tarefa 1).
- Produces:
  - `SnapshotCommand(QUndoCommand)` em `openstat/ui/undo.py`, com
    `__init__(self, view: WorksheetView, text: str, before: dict, after: dict)`. `undo` faz `restore(before)` e
    `redo` faz `restore(after)`; os dois chamam `view.refresh()`.
  - `WorksheetView.undo_stack: QUndoStack`, com `setUndoLimit(50)`.
  - `WorksheetView.edit(self, text: str, action: Callable[[], object]) -> object`:
    - tira `snapshot()`, roda `action` e tira o `snapshot()` de depois;
    - se `action` levanta, faz `restore(before)`, `refresh()` e relança;
    - se `changed_parts` não achou coluna diferente e os dicionários são iguais, não empilha;
    - senão, empilha `SnapshotCommand(self, tr(text), *changed_parts(before, after))`;
    - devolve o retorno de `action`.
    - É reentrante: uma chamada dentro de outra (o console rodando `run(...)`, que chama `add_result`) só executa
      `action`, e o passo fica com a de fora.
  - `MainWindow.undo_group: QUndoGroup`. `_add_tab` adiciona a pilha; `tabs.currentChanged` chama `setActiveStack`.
  - `MainWindow.view_of(ws: Worksheet) -> WorksheetView`.
- Pontos ligados e o texto de cada passo:

| Ponto | Texto |
|---|---|
| `WorksheetModel.set_cells` via `setData` | `"Typing"` |
| `paste` | `"Paste"` |
| `rename_column` | `"Rename column"` |
| `set_formats` | `"Conditional Formatting"` |
| `MainWindow.add_result` (só a parte de `stored`/`formulas`) | o título traduzido do resultado |
| `Console._run` | `"Command Line"` |

  O modelo chama a view para editar (por exemplo, `WorksheetModel` recebe a view ou um callback `edit`).
- Menu **Editar** novo entre Arquivo e Assistente, já com `undo_group.createUndoAction(self, tr("Undo"))`
  (Ctrl+Z) e `createRedoAction(self, tr("Redo"))` (Ctrl+Y). As outras ações entram na Tarefa 5.
- `set_project` limpa o grupo, porque as abas são recriadas.
- Desfazer marca o projeto como alterado: ligar `undo_stack.indexChanged` a `mark_dirty`.

- [ ] **Step 1: Escrever os testes que falham** em `tests/test_editing_app.py`:

```python
def test_typing_beyond_data_undo_restores_size(window):        # Review Focus 1
    view = window.current_view(); ws = view.ws
    view.model_.setData(view.model_.index(9, 5), "1")
    view.undo_stack.undo(); assert (ws.nrows, ws.ncols) == (4, 2)
    view.undo_stack.redo(); assert ws.df.iat[9, 5] == 1.0

def test_each_point_is_one_undo_step(window):
    # paste, rename_column (monkeypatch QInputDialog.getText), set_formats, console "df['C'] = df['A']*2":
    # after each, undo_stack.count() grew by 1, undo() gives back the snapshot taken before it
    ...

def test_console_in_place_change_is_undone(window):           # Review Focus 5
    window.console.execute("df.loc[0, 'A'] = 99")
    window.current_view().undo_stack.undo()
    assert window.current_worksheet().df["A"][0] == 1.0

def test_undo_analysis_keeps_output_and_new_worksheets(window):
    window.console.execute("run('data.rank', column='A')")      # stores Rank_A
    window.console.execute("run('data.sort', by=['A'])")        # new worksheet, no column change
    view = window.view_of(window.project.worksheets[1]); view.undo_stack.undo()
    assert "Rank_A" not in view.ws.df and len(window.project.outputs) == 2 and len(window.project.worksheets) == 3

def test_failing_result_leaves_no_half_column(window):        # Review Focus 2
    res = Result("X"); res.stored["Good"] = [1, 2, 3, 4]; res.formulas["Bad"] = "Bad + 1"
    with pytest.raises(ValueError): window.add_result(res, window.current_worksheet())
    assert "Good" not in window.current_worksheet().df and window.current_view().undo_stack.count() == 0

def test_undo_history_is_per_worksheet(window):
    # type in tab A, switch to tab B, undo_group.undo() leaves tab A unchanged
    ...

def test_no_step_when_nothing_changed(window):
    window.current_view().model_.setData(window.current_view().model_.index(0, 0), "1")   # same value
    assert window.current_view().undo_stack.count() == 0
```

  Hoje `add_result` deixa a exceção sair para o `_guarded` de quem chama. O teste do Review Focus 2 confirma esse
  contrato: `edit` relança.
- [ ] **Step 2: Rodar e ver falhar.** `.venv/Scripts/python -m pytest tests/test_editing_app.py -q`. Esperado: FAIL.
- [ ] **Step 3: Implementar** `undo.py`, `edit`, as ligações da tabela e o menu Editar com Desfazer e Refazer.
  Acrescentar as traduções de `"Edit"` (Editar), `"Undo"` (Desfazer), `"Redo"` (Refazer), `"Typing"` (Digitação) e
  `"Paste"` (Colar).
- [ ] **Step 4: Rodar a suíte inteira.** `.venv/Scripts/python -m pytest -q`. Esperado: tudo PASS (≥ 491 mais os
  novos). Rodar também `python tools/i18n_missing.py`; esperado: nenhum texto novo listado.
- [ ] **Step 5: Commit.** `git commit -am "M17: undo and redo for every worksheet change"` (com o arquivo de teste
  novo e `undo.py`).

### Task 5: Ações do menu Editar, tecla Delete, cabeçalho e barra de ferramentas

**Files:**
- Modify: `openstat/ui/worksheet_view.py`, `openstat/ui/main_window.py`, os dois `.json` de i18n
- Create: `MoveColumnsDialog` em `openstat/ui/worksheet_view.py` (junto do `FormatDialog`)
- Test: `tests/test_editing_app.py`

**Interfaces:**
- Consumes: as operações da Tarefa 2 e `WorksheetView.edit` (Tarefa 4).
- Produces (métodos de `WorksheetView`; cada um é um passo com o texto entre parênteses):
  - `cut()` ("Cut"): `copy_selection()` e depois `delete_selection()`, num único passo.
  - `delete_selection()` ("Delete Cells"):
    - se toda coluna selecionada está selecionada inteira (`selectionModel().isColumnSelected`), chama
      `delete_columns`;
    - senão, se toda linha selecionada está selecionada inteira, chama `delete_rows`;
    - senão, chama `delete_cells`.
  - `clear_selection()` ("Clear Cells"): `clear_cells`.
  - `insert_cells()`, `insert_rows()` e `insert_columns()` ("Insert Cells", "Insert Rows", "Insert Columns"):
    a partir da seleção. As linhas entram acima da primeira linha selecionada, tantas quantas estão selecionadas;
    as colunas entram à esquerda da primeira coluna selecionada, tantas quantas estão selecionadas.
  - `move_columns()` ("Move Columns"): abre o `MoveColumnsDialog`, com:
    - lista de colunas de múltipla escolha, já com as colunas selecionadas marcadas;
    - destino: Início, Fim, Antes de, Depois de;
    - coluna de referência.
- `keyPressEvent`: a tecla Delete (fora do modo de edição) chama `delete_selection`; Backspace continua limpando
  (`clear_selection`). Copiar, colar e recortar usam os métodos acima.
- Menu Editar da `MainWindow`, na ordem do spec, com atalhos:

| Ação | Atalho |
|---|---|
| Recortar | `QKeySequence.Cut` |
| Copiar | `QKeySequence.Copy` |
| Colar | `QKeySequence.Paste` |
| Excluir células | `Delete` |
| Limpar células | sem atalho |
| Selecionar tudo | `QKeySequence.SelectAll` |

  Localizar, Substituir e Propriedades entram nas Tarefas 6 e 7. Cada ação chama o método em
  `self.current_view()`. Com a grade em foco, os atalhos de Recortar, Copiar, Colar e Delete devem disparar uma vez
  só: usar `Qt.WidgetWithChildrenShortcut` na ação do menu ou tirar o tratamento duplicado do `keyPressEvent`.
- Menu do cabeçalho de coluna: acrescentar "Insert Columns", "Delete Cells" (rótulo "Delete Columns") e
  "Move Columns...".
- `QToolBar` "Standard" na `MainWindow`, com ícones de `self.style().standardIcon(...)` ou
  `QIcon.fromTheme(...)` (escolha do construtor), na ordem:
  1. abrir, salvar
  2. recortar, copiar, colar
  3. desfazer, refazer
  4. localizar, substituir (as duas entram na Tarefa 6)
  5. inserir células, linhas, colunas, mover colunas

- [ ] **Step 1: Escrever os testes que falham**:

```python
def test_delete_key_deletes_and_shifts_up(window):
    view = window.current_view(); view.setCurrentIndex(view.model_.index(0, 0))
    QTest.keyClick(view, Qt.Key_Delete)
    assert view.ws.df["A"].tolist()[:3] == [2.0, 3.0, 4.0] and view.undo_stack.undoText() == tr("Delete Cells")

def test_whole_column_selected_delete_removes_column(window):
    view = window.current_view(); view.selectColumn(0); view.delete_selection()
    assert view.ws.column_names() == ["B"]

def test_cut_is_one_step_and_fills_clipboard(window):
    view = window.current_view(); view.setCurrentIndex(view.model_.index(0, 1)); view.cut()
    assert QGuiApplication.clipboard().text() == "2" and view.undo_stack.count() == 1
    view.undo_stack.undo(); assert view.ws.df["B"][0] == 2.0

def test_insert_rows_and_columns_from_selection(window):
    # select rows 1..2 → insert_rows → nrows == 6; select column 0 → insert_columns → names[1] == "A"
    ...

def test_edit_menu_and_toolbar_exist(window):
    titles = [a.text() for a in window.menuBar().actions()]
    assert titles[1] == tr("Edit") and window.findChild(QToolBar) is not None
```

- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** as ações, o diálogo, a tecla Delete, o menu do cabeçalho e a barra, com as
  traduções:

| Inglês | pt-BR |
|---|---|
| Cut | Recortar |
| Copy | Copiar |
| Delete Cells | Excluir células |
| Clear Cells | Limpar células |
| Select All | Selecionar tudo |
| Insert Cells | Inserir células |
| Insert Rows | Inserir linhas |
| Insert Columns | Inserir colunas |
| Move Columns | Mover colunas |
| Delete Columns | Excluir colunas |
| Start | Início |
| End | Fim |
| Before | Antes de |
| After | Depois de |
| Standard | Padrão |

- [ ] **Step 4: Rodar a suíte inteira e o `i18n_missing`.** Esperado: PASS e nenhum texto novo listado.
- [ ] **Step 5: Commit.** `git commit -am "M17: edit menu, toolbar, cut, delete, insert and move"`.

### Task 6: Localizar e Substituir

**Files:**
- Create: `openstat/ui/find_dialog.py` (`FindDialog`)
- Modify: `openstat/ui/main_window.py` (ações Ctrl+F e Ctrl+H no menu Editar e na barra), os `.json` de i18n
- Test: `tests/test_editing_app.py`

**Interfaces:**
- Produces:
  - `FindDialog(QDialog)`, não modal. Recebe `window: MainWindow` e `replace: bool` (Substituir mostra o campo
    "Replace with" e os botões de substituir). Campos: `find_text`, `replace_text`, `match_case`, `whole_cell` e
    `selected_only` (`QCheckBox`).
  - Uma função pura em `find_dialog.py`:

    ```python
    find_next(ws, text, start: tuple[int, int], match_case: bool, whole_cell: bool,
              columns: list[int] | None) -> tuple[int, int] | None
    ```

    Percorre linha por linha a partir da célula seguinte a `start`, compara com `ws.get_text`, volta ao início ao
    chegar ao fim e para quando dá a volta completa.
  - Botões do diálogo:
    - "Find Next" seleciona a célula achada na view atual;
    - "Replace" troca a célula atual, se ela casar, e avança;
    - "Replace All" faz um único `edit("Replace", ...)` e mostra "{n} cells replaced".
  - A troca usa `str.replace` no texto exibido (ou o texto inteiro, quando "whole cell" está marcado) e grava com
    `set_cell`.

- [ ] **Step 1: Escrever os testes que falham**:

```python
def test_find_next_wraps_and_respects_options():
    ws = Worksheet("W", pd.DataFrame({"A": ["ab", "x", "AB"], "B": ["b", "ab", "c"]}))
    assert find_next(ws, "ab", (0, 0), False, False, None) == (1, 1)
    assert find_next(ws, "ab", (1, 1), True, False, None) == (0, 0)          # wraps; "AB" skipped by case
    assert find_next(ws, "a", (0, 0), False, True, None) is None              # whole cell
    assert find_next(ws, "ab", (0, 0), False, False, [0]) == (2, 0)

def test_replace_all_is_one_step(window):
    dialog = FindDialog(window, replace=True); dialog.find_text.setText("2"); dialog.replace_text.setText("5")
    dialog.whole_cell.setChecked(True); dialog.replace_all()
    ws = window.current_worksheet(); assert ws.df["A"][1] == 5.0 and window.current_view().undo_stack.count() == 1
```

- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar**, com as traduções:

| Inglês | pt-BR |
|---|---|
| Find... | Localizar... |
| Replace... | Substituir... |
| Find what | Localizar |
| Replace with | Substituir por |
| Match case | Diferenciar maiúsculas |
| Match entire cell | Célula inteira |
| Selected columns only | Somente colunas selecionadas |
| Find Next | Localizar próxima |
| Replace | Substituir |
| Replace All | Substituir tudo |
| cells replaced | células substituídas |
| Not found | Não encontrado |

- [ ] **Step 4: Rodar a suíte inteira e o `i18n_missing`.**
- [ ] **Step 5: Commit.** `git commit -am "M17: find and replace in the worksheet"`.

### Task 7: Diálogo Propriedades da coluna e dica do cabeçalho

**Files:**
- Modify: `openstat/ui/worksheet_view.py` (`ColumnPropertiesDialog`, menu do cabeçalho, `headerData` com
  `ToolTipRole`), `openstat/ui/main_window.py` (item no menu Editar), os `.json` de i18n
- Test: `tests/test_editing_app.py`

**Interfaces:**
- Consumes: `descriptions`, `value_orders`, `levels` (Tarefas 1 e 3); `edit` (Tarefa 4).
- Produces:
  - `ColumnPropertiesDialog(ws, name, parent)`, com:
    - `description` (`QLineEdit`);
    - `order` (`QComboBox` com "Alphabetical", "Order of occurrence" e "User-defined"), habilitado só em colunas de
      texto ou data;
    - `levels_list` (`QListWidget`), com botões "Move Up" e "Move Down", habilitado só quando a ordem é
      "User-defined" e já preenchido com `ws.levels(name)`.
  - `values()` devolve `(description: str, order: str | list[str] | None)`; `None` é a ordem padrão.
  - `WorksheetView.column_properties(section: int)`: aplica o resultado do diálogo num `edit("Column Properties", ...)`.
    Descrição vazia remove a chave; ordem `None` remove a chave.
- A dica do cabeçalho mostra a descrição na primeira linha, quando houver, e a fórmula e o erro abaixo, como hoje.

- [ ] **Step 1: Escrever os testes que falham**:

```python
def test_column_properties_apply_and_undo(window, monkeypatch):
    view = window.current_view(); ws = view.ws; ws.set_cell(0, 2, "b"); ws.set_cell(1, 2, "a")
    monkeypatch.setattr(ColumnPropertiesDialog, "exec", lambda self: QDialog.Accepted)
    monkeypatch.setattr(ColumnPropertiesDialog, "values", lambda self: ("weight in kg", ["b", "a"]))
    view.column_properties(2)
    assert ws.descriptions == {"C3": "weight in kg"} and ws.levels("C3") == ["b", "a"]
    assert view.model_.headerData(2, Qt.Horizontal, Qt.ToolTipRole).startswith("weight in kg")
    view.undo_stack.undo(); assert ws.descriptions == {} and ws.value_orders == {}

def test_order_disabled_for_numeric_column(qapp):
    dialog = ColumnPropertiesDialog(Worksheet("W", pd.DataFrame({"A": [1.0]})), "A")
    assert not dialog.order.isEnabled()
```

- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar**, com as traduções:

| Inglês | pt-BR |
|---|---|
| Column Properties... | Propriedades da coluna... |
| Description | Descrição |
| Value Order | Ordem dos valores |
| Alphabetical | Alfabética |
| Order of occurrence | Ordem de ocorrência |
| User-defined | Definida pelo usuário |
| Move Up | Subir |
| Move Down | Descer |

- [ ] **Step 4: Rodar a suíte inteira e o `i18n_missing`.**
- [ ] **Step 5: Commit.** `git commit -am "M17: column properties"`.

### Task 8: Ordem dos valores nos gráficos e nas análises que só exibem por grupo

**Files:**
- Modify:
  - `openstat/graphs/basic.py:27`
  - `openstat/graphs/more.py:55`
  - `openstat/graphs/relation.py:25,117-118`
  - `openstat/stats/anova.py:35` (`factor_groups`)
  - `openstat/stats/basic.py:69,86`
  - `openstat/stats/calc.py:154`
  - `openstat/stats/data.py:62,204`
  - `openstat/stats/equivalence.py:96` (`crossover_effect`)
  - `openstat/stats/gage.py:123,128,230,347,483,486`
  - `openstat/stats/nonparametric.py:85` (`_by_factor`)
  - `openstat/stats/quality.py:732-734`
- Test: `tests/test_value_order.py` (novo)

**Interfaces:**
- Consumes: `Worksheet.levels(name, labels=None)` (Tarefa 3).
- Troca:
  - `sorted(labels.unique())` vira `ws.levels(column, labels)`.
  - `groupby(by, sort=True)` vira um laço em `ws.levels(by)`, filtrando pelo rótulo formatado (`format_value`).
  - Onde hoje se compara o valor cru (`frame["operator"] == o`, em `gage.py`), comparar o rótulo formatado.
  - Funções que recebem só a série, como `crossover_effect(period1, period2, sequence)`, ganham o parâmetro
    `order: list[str] | None = None`, que quem chama preenche com `ws.levels(...)`.
- **Não** trocar `np.unique` sobre valores numéricos de cálculo (`gage.py:291,417,424`, `nonparametric.py:57,104`,
  `reliability.py`, `_life.py` e `_mars.py`).

- [ ] **Step 1: Escrever os testes que falham** em `tests/test_value_order.py`, usando a fixture `sample_ws`
  (`Group` tem os níveis A, B, C e D) com `sample_ws.value_orders["Group"] = ["D", "C", "B", "A"]`:

```python
def test_one_way_anova_follows_value_order(sample_ws): ...   # first row label of the means table == "D"
def test_boxplot_by_group_follows_value_order(sample_ws): ... # graphs.basic.groups(ws, ["Y"], "Group")[0][0] == "D"
def test_cross_tabulation_follows_value_order(sample_ws): ...  # rows of the table in D, C, B, A order
def test_unstack_follows_value_order(sample_ws): ...           # data.unstack new worksheet columns in D..A order
def test_numeric_factor_in_numeric_order(): ...                # factor 1, 2, 10 → means table rows "1", "2", "10"
```

  O construtor confere a função de cada teste de tabela cruzada em `stats/tables.py`. Se ela ordena por
  `pd.crosstab`, entra na troca com `reindex(ws.levels(...))`.
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Trocar** os pontos listados.
- [ ] **Step 4: Rodar a suíte inteira.** Esperado: os testes antigos passam, porque a ordem padrão de texto continua
  alfabética. Se um teste antigo quebrar por causa de um fator numérico com 10 ou mais níveis, que antes era
  ordenado como texto, atualizar a expectativa dele: a ordem numérica é a decisão do spec.
- [ ] **Step 5: Commit.** `git commit -am "M17: value order in graphs and grouped analyses"`.

### Task 9: Ordem dos valores nos modelos (evento, resposta ordinal, nível de referência)

**Files:**
- Modify:
  - `openstat/stats/regression.py:239-259` (`binary_fitted_line`), `:654-660` (`binary_logistic`) e `:689-695`
    (`_category_codes`)
  - `openstat/stats/basic.py:270-273` (`_events`)
  - `openstat/stats/_design.py:40` (`code_factor`) e seu chamador `build_design`
  - `openstat/stats/doe.py:446`
  - `openstat/stats/predictive.py:39`
  - `openstat/stats/multivariate.py:397`, se o `get_dummies` for sobre colunas da planilha
  - os `.json` de i18n
- Test: `tests/test_value_order.py`

**Interfaces:**
- Consumes: `Worksheet.levels` (Tarefa 3).
- Regras, como no Minitab:
  - **Evento** = `levels[-1]`, sendo `levels = ws.levels(response, labels)`. Isso vale para a logística binária, o
    gráfico de linha ajustada binário e `_events` das proporções. A linha "Event:" já existe em `regression.py`;
    acrescentar uma igual em `_events` (proporções), se ainda não houver.
  - **Resposta ordinal**: a ordem de `ws.levels(response, labels)` (resposta numérica: ordem numérica, como já é).
  - **Nível de referência** = `levels[0]` na codificação `"reference"` de `code_factor`. `code_factor` ganha
    `order: list[str] | None = None`, e `build_design` passa `ws.levels(name, labels)`.
  - Em `predictive.py`, `get_dummies` passa a usar `pd.Categorical(labels, categories=ws.levels(name, labels))`,
    para as colunas de indicadores saírem na ordem dos valores.
  - Onde um nível de referência é escolhido, acrescentar uma linha `res.text(f"{tr('Reference level')}: {name} = {levels[0]}")`.
    O construtor escolhe o lugar, uma vez por fator categórico.

- [ ] **Step 1: Escrever os testes que falham**:

```python
def test_binary_logistic_event_follows_value_order(sample_ws):
    sample_ws.value_orders["Pass"] = ["yes", "no"]
    html = binary_logistic(sample_ws, "Pass", continuous=["X"]).to_html()
    assert "Pass = no" in html          # event is now the last level, "no"

def test_reference_level_follows_value_order(sample_ws):
    sample_ws.value_orders["Group"] = ["C", "A", "B", "D"]
    html = <general linear model / regression with categorical=["Group"]>.to_html()
    assert "Group = C" in html and "Group_C" not in html     # C is the reference, so it has no coefficient

def test_ordinal_response_follows_value_order(sample_ws): ...  # levels returned by _category_codes == user order
def test_proportion_event_follows_value_order(sample_ws): ...  # one-proportion on "Pass" with order ["yes","no"] counts "no"
```

- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar.** Traduções: "Reference level" → "Nível de referência" ("Event" já existe).
- [ ] **Step 4: Rodar a suíte inteira e o `i18n_missing`.** Esperado: PASS. Com ordem padrão, evento e referência
  continuam os de hoje para texto (alfabético). Para fatores numéricos, a ordem passa a ser numérica; atualizar
  testes antigos só nesse caso, como na Tarefa 8.
- [ ] **Step 5: Commit.** `git commit -am "M17: value order sets event, ordinal order and reference level"`.

### Task 10: Fechamento

- [ ] **Step 1:** `.venv/Scripts/python -m pytest -q`. Esperado: tudo PASS.
- [ ] **Step 2:** `python tools/i18n_missing.py`. Esperado: nenhum texto do M17 listado.
- [ ] **Step 3:** Abrir o app (`.venv/Scripts/python -m openstat`) e conferir à mão:
  1. digitar, colar e Ctrl+Z;
  2. Delete numa célula e numa coluna inteira;
  3. Ctrl+F e Ctrl+H;
  4. Propriedades da coluna com ordem definida e um boxplot por grupo;
  5. troca de aba com desfazer separado;
  6. salvar e reabrir o projeto.
- [ ] **Step 4:** `graphify update .` e commit final, se o grafo mudou.
