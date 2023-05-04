# To try

* normal

```bash
python scripts/trainer.py --config-file configs/ablations/naive_boot.py --machine CCV --name Base --cfg.NOTIFICATION.ENABLE True
```

* Low Length Tax

```bash
python scripts/trainer.py --config-file configs/ablations/naive_boot.py --machine CCV --name LowLengthTax --cfg.OBJECTIVE.LENGTH_TAX -0.001 --cfg.NOTIFICATION.ENABLE True
```

* Fast Turn around - PLAD MAX OUTER = 0 Longer epoch

```bash
python scripts/trainer.py --config-file configs/ablations/naive_boot.py --machine CCV --name FastTurn --cfg.PLAD.MAX_OUTER_ITER 0 --cfg.TRAIN.DATASET.EPOCH_SIZE 64000 --cfg.NOTIFICATION.ENABLE True
```

* MACRO PER ERA 1

```bash
python scripts/trainer.py --config-file configs/ablations/naive_boot.py --machine CCV --name SingleMacro --cfg.ABSTRACTION.CS_SPLICER.MACRO_PER_ERA 1 --cfg.ABSTRACTION.CS_SPLICER.MAX_NEW_PERCENTAGE 1.0 --cfg.NOTIFICATION.ENABLE True
```

* MAX_NEW_PERCENT = 1.0
  
```bash
python scripts/trainer.py --config-file configs/ablations/naive_boot.py --machine CCV --name MaxIncrease --cfg.ABSTRACTION.CS_SPLICER.MAX_NEW_PERCENTAGE 1.0 --cfg.NOTIFICATION.ENABLE True
```

* Transformer Reload  Longer epoch
  
```bash
python scripts/trainer.py --config-file configs/ablations/naive_boot.py --machine CCV --name SingleMacroTFReload --cfg.ABSTRACTION.CS_SPLICER.MAX_NEW_PERCENTAGE 1.0 --cfg.MODEL.RELOAD_ON_DSL_UPDATE True --cfg.NOTIFICATION.ENABLE True
```

* NO Splice Injection

```bash
python scripts/trainer.py --config-file configs/ablations/naive_boot.py --machine CCV --name NoSpliceInject --cfg.ABSTRACTION.CS_SPLICER.MAX_NEW_PERCENTAGE 0.0 --cfg.NOTIFICATION.ENABLE True
```

* NO Macros

```bash
python scripts/trainer.py --config-file configs/ablations/naive_boot.py --machine CCV --name NoSpliceInject --cfg.ABSTRACTION.CS_SPLICER.MAX_NEW_PERCENTAGE 0.0 --cfg.ABSTRACTION.CS_SPLICER.MACRO_PER_ERA 1 --cfg.NOTIFICATION.ENABLE True
```
