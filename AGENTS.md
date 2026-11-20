# Research Paper Semantic Search

A reproducible SciFact information-retrieval benchmark comparing BM25, dense encoders, hybrid fusion, and cross-encoder reranking.

## Build commands

```bash
# Verification commands.
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```

## Platform support

- macOS and Linux; Python 3.11-3.13 (Kaggle images ship 3.11).
- Exact retrieval and the offline fixture are CPU-safe; dense encoding, FAISS, and cross-encoder reranking need the `neural` extra and are intended to run on Kaggle (`scripts/kaggle_run.sh`).
- `uv sync --extra neural` downloads sentence-transformers, torch, and faiss-cpu; local tests never download models.

## Key conventions

- Preserve SciFact document, query, split, and qrel IDs; keep unjudged documents eligible for retrieval.
- Tune encoders, fusion, reranking, and cutoffs only on development queries, never final test qrels.
- Version index manifests with corpus hashes, preprocessing, model revisions, and deterministic tie handling.

## Rules

- Default branch is `main`; never force-push it or rewrite shared history.
- Ask before adding any dependency or package manager.
- Never fabricate data, benchmarks, logs or results.
- Done means the relevant tests actually ran and passed; never skip or mock them to finish.
- No emojis anywhere: code, comments, commits, READMEs, docs.
- No em dashes (the long dash, U+2014) anywhere: code, comments, commits, READMEs, docs. Use a comma, colon, parentheses, or a new sentence instead.
- Keep README in sync with code changes, following its existing template.
- Comments are brief and explain why, only when the code doesn't already say it.
- Keep dataset splits and eval leakage controls intact.

## Commits

- Conventional Commits: `type(scope): summary`. Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`, `ci`, `chore`; scope is optional.
- Summary in imperative mood, lowercase, no trailing period, at most 72 characters.
- One logical change per commit; keep formatting-only changes separate.
- Add a body only when the reason is not obvious: explain why, not what, wrapped at 72.
- Breaking changes: `type!:` in the summary plus a `BREAKING CHANGE:` footer.
- Commit only when asked. Never commit secrets, build output, or failing code.
