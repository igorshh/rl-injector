# InjecAgent data used by the RL-Hammer runs

These files are the exact inputs of every attacker training run and every attacker
evaluation in this repo. They are committed (the upstream .gitignore excluded them) so a
checkout is runnable and the numbers are reproducible.

| file | entries | sha256 |
|---|---|---|
| `raw/test_cases_dh_base.json` | 510 | `0a8186468d21389af432e8c7b399ae42264d1b93a07b65c7a489468508604305` |
| `dataset/train.json` | 310 | `bc4569d127ffbc272675af08ff51d0903b534b7ec204a48c892761d73a56c57b` |
| `dataset/eval.json` | 100 | `cd2a50dd98eae7d4ac20a36948227045c4cb9414caef708f09c30ba0dedbc30f` |
| `dataset/test.json` | 100 | `a93e2fcbf59baa546b69a900ff53daa0c1fa18785c689bfa2b8b5a258df6be1b` |
| `dataset/eval_smoke10.json` | 10 | `a7858691183946b3df6595da48181cefe572a0e0e3860ea378d55974c0997a1c` |
| `tools.json` | 38 | `e21a8f70b1d5de4677d6d52642936a322655d79b17a72c84f600550384083a1e` |

* `raw/test_cases_dh_base.json`: InjecAgent direct-harm base test cases (upstream source).
* `dataset/{train,eval,test}.json`: produced by `split_dataset.py` from the raw file
  (unseeded shuffle; 100 eval, 100 test, the rest train), frozen on 2026-09-04. Attackers
  train on `train.json`; every reported attack success rate is on **`eval.json`**
  (`--validation_data_path data/InjecAgent/dataset/eval.json`, cache key
  `saved_adv_prompts/<attacker>/data_InjecAgent_dataset_eval/`). `test.json` is held out.
* `dataset/eval_smoke10.json`: first 10 eval cases, for smoke tests only.
* `tools.json`: InjecAgent tool schemas, read by `utils.py` at import time.
