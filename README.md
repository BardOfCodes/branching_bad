# Bootstrapped Abstraction Discovery

Basically library learning where the right programs also have to be discovered from scratch. Similar to macro-action discovery in RL. 

## Targets

1) Naive Bootstrapped LL
2) Many-world Bootstrapped LL

## Environment

1) Simple 2D CSG programs.
2) 3D CSG programs.

## Approach

1. Start from simple actions to create a MDP
2. How to solve the MDP

Training for MDP solving:
    1) NN + PLAD
    2) Rewrite with <DiffOpt, CG, CP>
    3) limit to n-updates where n is small.

The outcome of MDP solving is: 
    1) Set of "best programs" for the training data.
    2) Network which can do good visual inference on validation sets. 

1. Once solved how to discover abstractions.

Discovery will be done with the CG method - <>
Find highest D1 -> D2, Pset1 -> Pset2

[R delta (positive) + D delta (negative)] - Should the solver consider expression quality? [match with target / overall program performance]

1. How to reintegrate the abstractions.

Update bootstrap dataset.
Update the token space with new tokens.
Do 2

## TODO

1. Implement Beam Search with new transformer.
2. Pretrain Model
3. Evaluate pretrained model.
4. Naive BAD
5. Branching BAD
