  
# Main

```bash
python scripts/trainer.py --config-file configs/ablations/branching_boot.py --machine CCV --name BaseBranchingBAD --cfg.NOTIFICATION.ENABLE True

python scripts/trainer.py --config-file configs/ablations/naive_boot.py --machine CCV --name BaseNaiveBAD --cfg.NOTIFICATION.ENABLE True

python scripts/trainer.py --config-file configs/ablations/plad.py --machine CCV --name BasePLAD --cfg.NOTIFICATION.ENABLE True
```

# Ablations

Branching factor and New expr

```bash
python scripts/trainer.py --config-file configs/ablations/branching_boot.py --machine CCV --name BranchingBADBFactor5 --cfg.N_BRANCHES 5 --cfg.NOTIFICATION.ENABLE True
python scripts/trainer.py --config-file configs/ablations/naive_boot.py --machine CCV --name NaiveBADNewExpr1 --cfg.ABSTRACTION.CS_SPLICER.MACRO_PER_ERA 1 --cfg.NOTIFICATION.ENABLE True
python scripts/trainer.py --config-file configs/ablations/naive_boot.py --machine CCV --name NaiveBADNewExpr5 --cfg.ABSTRACTION.CS_SPLICER.MACRO_PER_ERA 5 --cfg.NOTIFICATION.ENABLE True
```

# Dream phase

```bash
python scripts/trainer.py --config-file configs/ablations/naive_boot.py --machine CCV --name NaiveBADSearch0 --cfg.ABSTRACTION.CS_SPLICER.MAX_NEW_PERCENTAGE 0.0 --cfg.ABSTRACTION.DELETE_THRESHOLD 0.001 --cfg.NOTIFICATION.ENABLE True
python scripts/trainer.py --config-file configs/ablations/naive_boot.py --machine CCV --name NaiveBADSearch1 --cfg.ABSTRACTION.CS_SPLICER.MAX_NEW_PERCENTAGE 1.0 --cfg.NOTIFICATION.ENABLE True
```

# Weight Update

```bash
python scripts/trainer.py --config-file configs/ablations/naive_boot.py --machine CCV --name NaiveBADwtMode1 --cfg.MODEL.RELOAD_ON_DSL_UPDATE 1 --cfg.NOTIFICATION.ENABLE True
python scripts/trainer.py --config-file configs/ablations/naive_boot.py --machine CCV --name NaiveBADwtMode2 --cfg.MODEL.RELOAD_ON_DSL_UPDATE 2 --cfg.NOTIFICATION.ENABLE True
```

## Other ablations

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
