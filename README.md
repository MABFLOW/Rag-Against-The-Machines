*This project has been created as part of the 42 curriculum by amaghafr.*

# RAG against the machine

## Description

**RAG against the machine** is a Retrieval-Augmented Generation (RAG) system built to
answer questions about a real-world codebase — [vLLM](https://github.com/vllm-project/vllm)
(version 0.10.1) — using only the content of that codebase as its source of truth.

The goal of the project is to build, end to end, the core pieces of a RAG pipeline:

- **Chunking**: turning raw source files (Python, Markdown, text) into indexable pieces
  of content.
- **Indexing**: building a lexical (BM25) search index over those chunks.
- **Retrieval**: given a natural-language question, finding the most relevant chunks.
- **Generation**: feeding the retrieved chunks to a local language model to produce a
  grounded answer.
- **Evaluation**: measuring retrieval quality with recall@k against a labeled dataset.

Everything is exposed through a single command-line entry point built with
[`fire`](https://github.com/google/python-fire), so each stage of the pipeline
(indexing, searching, answering, evaluating) can be run and inspected independently.

## Instructions

### Requirements

- Python >= 3.14
- [`uv`](https://docs.astral.sh/uv/) for dependency management and running the project

### Installation

```sh
make install
# equivalent to: uv sync
```

This installs all runtime dependencies declared in `pyproject.toml`
(`torch`, `transformers`, `accelerate`, `bm25s`, `pydantic`, `fire`, `tqdm`, `flake8`)
as well as the `mypy`/`types-tqdm` dev tooling.

### Running the project

The CLI is exposed via `python -m src`, wired to the `Engine` class through `fire`, so
every public method of `Engine` becomes a subcommand:

```sh
make run ARGS="<command> [args...]"
# equivalent to: uv run python3 -m src <command> [args...]
```

Available commands (see [Example usage](#example-usage) below for concrete calls):

| Command          | Purpose                                                          |
|-------------------|-------------------------------------------------------------------|
| `index`           | Chunk the source folder and build the BM25 index                |
| `search`          | Run a single query against the index and print matching sources |
| `search_dataset`  | Run every question of a dataset through search, save the results|
| `answer`          | Search + generate a grounded answer for a single question       |
| `answer_dataset`  | Generate answers for a whole dataset of search results           |
| `evaluate`        | Compute recall@k of a dataset's search results vs. ground truth |

### Linting and type checking

```sh
make lint
```

Runs `flake8` and `mypy` (with `--disallow-untyped-defs`, `--check-untyped-defs`,
`--warn-return-any`, `--warn-unused-ignores`) over the `src/` package.

### Cleaning

```sh
make clean
```

Removes `__pycache__`, `.mypy_cache`, `.pytest_cache`, `*.egg-info`, and compiled `.pyc`
files.

## System architecture

The pipeline is orchestrated by `Engine` (`src/engine.py`), which delegates to four
focused components:

```
                 ┌───────────┐
  raw files ───▶ │  Chunker  │──▶ chunks.json
                 └───────────┘
                       │
                       ▼
                 ┌───────────┐
                 │  Indexer  │──▶ BM25 index (data/processed/bm25_index)
                 └───────────┘
                       │
              query    ▼
        ─────────────▶ search() ──▶ retrieved sources
                       │
                       ▼
                 ┌───────────┐
                 │ Generator │──▶ grounded answer (local LLM)
                 └───────────┘
```

- **`Parser`** (`src/parser.py`) — shared I/O and validation utilities: reading files,
  loading/dumping JSON, and validating CLI arguments (numbers, strings, paths).
- **`Chunker`** (`src/chunker.py`) — walks the raw source folder and splits every
  eligible file into size-bounded chunks (see *Chunking strategy*).
- **`Indexer`** (`src/indexer.py`) — builds a BM25 index over the chunk contents using
  [`bm25s`](https://github.com/xhluca/bm25s) and persists/reloads it from disk.
- **`Generator`** (`src/generator.py`) — wraps a local Hugging Face
  `text-generation` pipeline (`Qwen/Qwen3-0.6B`, CPU, float32) and turns a question plus
  retrieved contexts into a grounded answer.
- **`Validation`** (`src/validation.py`) — validates raw JSON against the project's
  Pydantic schemas (`src/models.py`) before it enters the pipeline.
- **`Engine`** (`src/engine.py`) — the CLI-facing orchestrator: wires the above
  components together for indexing, searching, answering, and evaluation.

Data flows through the pipeline as plain JSON on disk between stages
(`data/processed/chunks.json`, the BM25 index directory, and
`StudentSearchResults`/`StudentSearchResultsAndAnswer` JSON files), so each stage can be
run, inspected, and re-run independently.

## Chunking strategy

Chunking is handled by `Chunker.run()`, which walks `data/raw/vllm-0.10.1`
(skipping `.git`, `.venv`, `__pycache__`) and processes three file types differently:

- **Python files** (`.py`): parsed with the `ast` module. Every top-level `class`,
  `def`, and `async def` becomes its own chunk, tagged with its type (`class`,
  `function`, `async_function`) and name. This keeps each chunk semantically coherent
  (a whole function/class) instead of splitting on arbitrary line counts.
- **Text/Markdown files** (`.txt`, `.md`): the whole file content is treated as a
  single logical block and split by `splitting_content`.

For any content larger than `max_chunk_size` (2000 characters by default),
`splitting_content` performs boundary-aware splitting:

1. Try to split at the last paragraph break (`\n\n`) before the size limit.
2. Fall back to the last line break (`\n`) if no paragraph break is found.
3. Fall back to a hard cut at the size limit if neither is found.
4. Start the next chunk 200 characters before the split point, so consecutive chunks
   overlap slightly and don't lose context at the boundary.

Each resulting chunk (`ChunkModel`) records its file path, type, name, part index,
total parts, content, and absolute character offsets in the source file — the offsets
are what let retrieval report exact source locations later.

On the current corpus (vLLM 0.10.1, 1965 source/doc files), this produces **30,509
chunks**: 20,534 functions, 8,102 classes, 1,069 async functions, and 804 text/Markdown
chunks.

## Retrieval method

Retrieval is purely lexical, using [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) via
the `bm25s` library (`Indexer`, `src/indexer.py`):

1. **Indexing** (`Indexer.index`): every chunk is rendered into a short text block
   (file name, path, type, name, content), the corpus is tokenized with
   `bm25s.tokenize`, and a `bm25s.BM25` retriever is built and persisted to
   `data/processed/bm25_index` alongside the original chunk metadata.
2. **Querying** (`Indexer.search`): one or more query strings are tokenized the same
   way, and `BM25.retrieve` returns the top-`k` scoring chunks per query, ranked purely
   by BM25 score (term-frequency / inverse-document-frequency weighting with length
   normalization — no re-ranking or embedding-based step is applied).
3. Results are converted into `MinimalSource` objects carrying the file path and the
   character span of the matched chunk, so downstream consumers (CLI output, the
   generator, evaluation) can locate the exact text that was retrieved.

BM25 was chosen over dense/embedding retrieval for this project because it requires no
model download or GPU, is fast to index (30k+ chunks in seconds) and to query, and is
easy to reason about and debug — all useful properties when the whole pipeline needs to
run locally and be evaluated iteratively.

## Performance analysis

Retrieval quality is measured with `Engine.evaluate`, computing **recall@k**: for each
question with known ground-truth source spans, the fraction of those spans that are
covered (overlapping file + character range) by *any* of the top-`k` retrieved chunks,
averaged over all questions.

Running `evaluate` at `k=5` against the public labeled datasets gives:

| Dataset                          | Questions | Recall@5 |
|-----------------------------------|-----------|----------|
| `dataset_code_public.json`        | 100       | **0.68** |
| `dataset_docs_public.json`        | 100       | **0.84** |
| **Overall (average)**             | 200       | **0.76** |

A few observations:

- Recall is noticeably higher on documentation questions than on code questions. BM25
  matches on surface tokens, and natural-language documentation tends to share more
  vocabulary with natural-language questions than source code does (code questions
  often rely on identifier names, APIs, or behavior that isn't stated verbatim in the
  matching chunk).
- Chunking Python by top-level `class`/`function` boundaries helps keep each chunk
  focused, which improves BM25's precision (fewer irrelevant tokens diluting a match)
  compared to fixed-size, boundary-agnostic chunks.
- Indexing the full corpus (~30.5k chunks) and querying it are both fast — BM25's
  sparse, inverted-index nature keeps this a CPU-only, seconds-scale operation, which
  is why iterating on chunking/indexing parameters was practical during development.

## Design decisions

- **Lexical (BM25) retrieval over dense embeddings.** Simpler to implement correctly,
  no extra model/GPU dependency for the retrieval step, and fast enough to iterate on
  for a corpus of this size. The tradeoff is weaker performance on
  paraphrased/semantic queries that don't share vocabulary with the target chunk.
- **AST-based chunking for Python instead of fixed-size windows.** Splitting on
  `class`/`function` boundaries keeps each chunk a coherent, self-contained unit of
  code, which both improves BM25 matching and gives more meaningful sources to show
  the user (a whole function, not an arbitrary slice of it).
- **Character-offset-based sources.** Every chunk and every retrieved/generated answer
  carries `(file_path, first_character, last_character)` instead of a chunk ID. This
  makes results independent of how a file was chunked and lets `evaluate` compare
  retrieved spans against ground-truth spans by simple overlap, rather than requiring
  identical chunk boundaries.
- **JSON-on-disk between stages.** `chunks.json`, the BM25 index directory, and the
  `StudentSearchResults`/`StudentSearchResultsAndAnswer` files decouple indexing,
  searching, answering, and evaluation, so any stage can be re-run, inspected, or
  swapped independently.
- **Small local generator model (`Qwen/Qwen3-0.6B`, CPU).** Keeps the whole pipeline
  runnable without external API calls or GPU access, at the cost of weaker generation
  quality than a larger hosted model.
- **Pydantic models for every JSON boundary.** All data crossing a file/CLI boundary
  (`ChunkModel`, `RagDataset`, `StudentSearchResults`, ...) is validated through
  `Validation`/`Parser` before use, so malformed inputs fail fast with a clear error
  instead of propagating silently.

## Challenges faced

- **Choosing a chunk granularity for code.** Fixed-size chunking splits functions and
  classes mid-body, hurting both retrieval relevance and answer readability. Switching
  to AST-based chunking (splitting at `class`/`def`/`async def` boundaries, with
  paragraph/line-aware overlap for oversized chunks) fixed this at the cost of extra
  parsing logic per language.
- **Comparing retrieved and ground-truth spans fairly.** Because chunk boundaries in the
  index don't necessarily match the ground-truth dataset's spans, an exact-match
  comparison would under-count correct retrievals. `evaluate` instead checks for
  *character-range overlap* within the same file (`Engine._sources_overlap`), which
  tolerates boundary differences while still requiring genuine overlap.
- **Type-checking a pipeline built on loosely-typed third-party libraries.**
  `bm25s` and `fire` ship without type stubs, `transformers`' pipeline objects can be
  statically `None`/union-typed (e.g. `tokenizer`), and Pydantic models validated from
  raw dataset JSON return a union type (`AnsweredQuestion | UnansweredQuestion`) that
  isn't guaranteed to carry ground-truth `sources`. Getting a clean `mypy` pass required
  targeted `ignore_missing_imports` overrides for untyped libraries, explicit `None`
  checks around the generator's tokenizer/output, and an `isinstance` guard before
  accessing `.sources` in `evaluate` — the last of which also fixed a latent crash on
  datasets containing unanswered questions.
- **Running everything locally.** Keeping the generator small (`Qwen3-0.6B`, CPU,
  `float32`) and the retriever lexical (BM25, no embedding model) was necessary to keep
  indexing and evaluation iterations fast without GPU access.

## Example usage

Build the index from the raw corpus:

```sh
uv run python3 -m src index --max_chunk_size=2000
```

Search the index for a single query:

```sh
uv run python3 -m src search --query="How does vLLM handle KV cache eviction?" --k=5
```

Generate a grounded answer to a single question:

```sh
uv run python3 -m src answer --query="What activation formats does the fused batched MoE layer return?" --k=5
```

Run search over a full labeled dataset and save the results:

```sh
uv run python3 -m src search_dataset \
  --dataset_path=data/datasets_public/public/AnsweredQuestions/dataset_code_public.json \
  --k=5 \
  --save_directory=data/results
```

Generate answers for a whole dataset of previously-saved search results:

```sh
uv run python3 -m src answer_dataset \
  --student_search_results_path="data/results/StudentSearchResult s.json" \
  --save_directory=data/results
```

Evaluate recall@k of a dataset's search results against ground truth:

```sh
uv run python3 -m src evaluate \
  --student_search_results_path="data/results/StudentSearchResult s.json" \
  --dataset_path=data/datasets_public/public/AnsweredQuestions/dataset_code_public.json
# Recall@k over 100 questions: 0.6800
```

## Resources

- [Okapi BM25 (Wikipedia)](https://en.wikipedia.org/wiki/Okapi_BM25) — background on the
  ranking function used for retrieval.
- [`bm25s` documentation](https://github.com/xhluca/bm25s) — the BM25 implementation
  used for indexing and retrieval.
- [Python `ast` module documentation](https://docs.python.org/3/library/ast.html) —
  used to parse Python source files for AST-based chunking.
- [Pydantic documentation](https://docs.pydantic.dev/) — schema validation for every
  JSON boundary in the pipeline.
- [Hugging Face `transformers` — Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines) —
  the `text-generation` pipeline used by `Generator`.
- [Retrieval-Augmented Generation (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401) —
  the original RAG paper motivating the retrieve-then-generate architecture.
- [`mypy` documentation](https://mypy.readthedocs.io/) — static type checking used in
  `make lint`.

### AI usage

An AI assistant was used to help add type hints and docstrings across `src/`, fix a
few of the type errors that surfaced along the way, and draft parts of this README. All
suggestions were reviewed and checked (`mypy`, `flake8`) before being committed.
