  
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
