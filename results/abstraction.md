# To try

* Low Length Tax

```bash
python scripts/trainer.py --config-file configs/ablations/naive_boot.py --machine CCV --name LowLengthTax --cfg.OBJECTIVE.LENGTH_TAX -0.001
```

* Fast Turn around - PLAD MAX OUTER = 0 Longer epoch

```bash
python scripts/trainer.py --config-file configs/ablations/naive_boot.py --machine CCV --name FastTurn --cfg.PLAD.MAX_OUTER_ITER 0 --cfg.TRAIN.DATASET.EPOCH_SIZE 64000
```

* MACRO PER ERA 1

```bash
python scripts/trainer.py --config-file configs/ablations/naive_boot.py --machine CCV --name SingleMacro --cfg.ABSTRACTION.CS_SPLICER.MACRO_PER_ERA 1
```

* MAX_NEW_PERCENT = 1.0
  
```bash
python scripts/trainer.py --config-file configs/ablations/naive_boot.py --machine CCV --name SingleMacro --cfg.ABSTRACTION.CS_SPLICER.MAX_NEW_PERCENTAGE 1.0
```

* Transformer Reload  Longer epoch
  
```bash
python scripts/trainer.py --config-file configs/ablations/naive_boot.py --machine CCV --name SingleMacroTFReload --cfg.ABSTRACTION.CS_SPLICER.MAX_NEW_PERCENTAGE 1.0 --cfg.MODEL.RELOAD_ON_DSL_UPDATE True
```

* NO Splice Injection

```bash
python scripts/trainer.py --config-file configs/ablations/naive_boot.py --machine CCV --name NoSpliceInject --cfg.ABSTRACTION.CS_SPLICER.MAX_NEW_PERCENTAGE 0.0
```

* NO Macros

```bash
python scripts/trainer.py --config-file configs/ablations/naive_boot.py --machine CCV --name NoSpliceInject --cfg.ABSTRACTION.CS_SPLICER.MAX_NEW_PERCENTAGE 0.0 --cfg.ABSTRACTION.CS_SPLICER.MACRO_PER_ERA 1
```
