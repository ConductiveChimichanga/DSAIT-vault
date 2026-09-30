# DSAIT vault

Obsidian vault with study notes for the TU Delft DSAIT master.

> [!WARNING]
> **Disclaimer:** parts of these notes may be AI-generated. They can contain mistakes, so double-check anything important against the lecture slides, the course books or the lecturers before relying on it.

| Folder                                  | Course                              | Start at                                                                                                                        |
| --------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `MDL (Machine and Deep Learning)`       | DSAIT4005 Machine and Deep Learning | [MDL Course walkthrough](obsidian://open?vault=DSAIT&file=MDL%20(Machine%20and%20Deep%20Learning)%2FMDL%20Course%20walkthrough) |
| `PAIR (Probabilistic AI and Reasoning)` | Probabilistic AI and Reasoning      | `00 PAIR Index`                                                                                                                 |

Each course folder has the same layout: an index note, lecture notes grouped per week or lecture, a `Concepts` folder with one note per definition, and a `figures` folder.

## Opening the vault

1. Install [Obsidian](https://obsidian.md) (1.4 or newer, for note Properties and callouts).
2. Clone this repository and choose "Open folder as vault".
3. When asked, trust the author and enable community plugins. The plugins are included in `.obsidian/plugins`, so nothing has to be downloaded.

## Obsidian plugins

Built-in features the notes rely on (no plugin needed): LaTeX math, Mermaid diagrams, callouts, wikilinks, Properties, graph view.

| Plugin | Version in this vault | Needed for | Required? |
|---|---|---|---|
| Spaced Repetition | 1.15.4 | Reviewing the `## Flashcards` sections (`question::answer` cards tagged `#flashcards/...`) | Required for flashcards; notes read fine without it |
| Git | 2.40.0 | Committing and syncing the vault from inside Obsidian | Optional, plain `git` works too |
| Excalidraw | 2.27.3 | Hand-drawn sketches | Optional, no note depends on it yet |
| Dataview | 0.5.68 | Queries over note properties (`week`, `slides`, `lab`) | Optional, no note depends on it yet |
| Advanced Tables | 0.23.2 | Easier editing of Markdown tables | Optional |
| Tasks | 8.4.0 | Task tracking | Optional |
| Kanban | 2.0.51 | Boards | Optional |
| Claudian | 2.3.6 | Claude inside Obsidian | Optional; its session folder `.claudian/` is not in the repository |

Spaced Repetition settings that the cards assume: flashcard tag `#flashcards`, single-line separator `::`.

## Not in the repository

`.claudian/`, Obsidian workspace files and caches (see `.gitignore`). Course slides and assignments are not stored here either.
