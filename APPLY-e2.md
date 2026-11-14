# A2 E2 first-version patch

Files to add:
- `src/replication/e2.py`
- `configs/exp/e2.yaml`
- `tests/test_e2.py`

Existing file to update:
- `src/replication/train.py` (register `e2.run`)

The included `src/replication/train.py` is a full replacement for the repository
version inspected on 2026-11-14. Alternatively, apply `patches/train.py.patch`.

Recommended low-cost validation before Kaggle training:

```bash
python -m pytest tests/test_e2.py -q
python -m pytest -q
```

Do not start the 30,000-iteration run until these pass.
