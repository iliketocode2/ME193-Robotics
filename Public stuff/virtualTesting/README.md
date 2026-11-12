# virtualTesting

A one-shot script for trying out `legoeducation` in a clean, disposable
environment — no manual venv setup needed.

## Usage

```bash
python3 setup_test_env.py
```

This will:

1. Create a `test/` folder next to the script (ignored by git — it's
   throwaway).
2. Create a virtual environment at `test/.venv`.
3. Install `legoeducation` into it.
4. (macOS only) Open a new Terminal window already `cd`'d into `test/`
   with the venv activated.

From there, just run `python3` to drop into a REPL with `legoeducation`
already importable:

```python
>>> import legoeducation
```

Re-running the script reuses the existing `test/.venv` instead of
recreating it.
