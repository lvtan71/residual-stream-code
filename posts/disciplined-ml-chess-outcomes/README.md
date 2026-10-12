# Predicting Chess Outcomes: the full code

The scripts behind [Predicting Chess Outcomes: A Disciplined ML Walkthrough](https://residual-stream.lvtan71.workers.dev/posts/disciplined-ml-chess-outcomes/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `data.py` | Downloads January 2015 from the Lichess database (a pinned parquet copy on Hugging Face), keeps the first 100,000 rated blitz games, and audits them. Writes `games.parquet`. | `huggingface_hub`, `pandas`, `pyarrow`; downloads about 400 MB |
| `features.py` | Replays the first ten moves of each game with python-chess and records ratings, time control and board features after moves 3 and 10. Writes `features.parquet`. | `chess`; about 10 s on 4 cores |
| `contract.py` | The leakage contract: which columns each model may see, and which games each is scored on. Imported by the rest. | |
| `test_contract.py` | The contract's rules as tests. | runs with `python` or `pytest` |
| `models.py` | The split, the gradient-boosted model, the Elo baselines and the metrics. Imported by the rest. | `scikit-learn` |
| `train.py` | Trains the models on one 80/20 split and scores them against baselines, with a bootstrap interval. | about 20 s |
| `seeds.py` | The same on five random splits, a split by time, and a split that keeps each player on one side. | a few minutes |
| `leak.py` | Adds one leaked column to the Elo model, shows the score it would report, and the test that catches it. | |
| `diagnose.py` | A win-rate table by rating gap and material, a castling check, a learning curve, permutation importance and error by segment. | about 15 s |
| `engine.py` | Stockfish's evaluation of every move-10 position. Writes `engine.parquet`. | a `stockfish` binary on PATH (the outputs used Stockfish 18); minutes on 4 cores |
| `engine_ab.py` | The move-10 model with and without the engine's evaluation, on the same five splits. | reads `engine.parquet` |

`engine.parquet` is committed, so `engine_ab.py` runs without Stockfish. The outputs came from an Apple M4 laptop (10 cores) with Python 3.12, pandas, scikit-learn 1.9.1 and python-chess 1.11.2.

```bash
pip install huggingface_hub pandas pyarrow chess scikit-learn
python data.py
python features.py
python test_contract.py
python train.py
python seeds.py
python leak.py
python diagnose.py
python engine.py        # optional: needs Stockfish
python engine_ab.py
```
