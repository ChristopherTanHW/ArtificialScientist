# Artificial Scientist

This repository contains the code for our implementation of the [Artificial Scientist](https://drive.google.com/file/d/1xKBUPdbc2X18hyAlSAgAS2tp2gV8NP7d/view?usp=sharing)


- Our work builds on the [RTFM environment and txt2pi model](https://arxiv.org/abs/1910.08210) proposed by Victor Zhong et al.

## Meta Bandit Writer (UCB)

The code for the Bandit Writer is found in metaBandit.py. To train the Meta UCB Bandit Writer:

```
python run_exp.py --mode train_writer --difficulty character_level --num_eps 100
```

The default task is set to a character-level lie correction task based on the RTFM Rock Paper Scissors Game, where only a single character in the wiki corresponding to an entity is false, resulting in a false description of entity relationships. To increase the difficulty of the task such that the writer has to correct an entire statement without any constraints on the ordering of tokens within that statement, run:

```
python run_exp.py --mode train_writer --difficulty statement_level --num_eps 10000000
```

The more difficult statement level task requires substantially more training episodes

## Observant Scientist (supervised training)

The 'Observant Scientist' is an improvement from the Meta Bandit approach, by encoding entire rollouts so as to use features from them to inform what can be deduced and written about the envroniment dynamics rather than random guessing as the meta bandit does. We train the observant scientist in a supervised manner, where the dataset is curated by collecting winning rollouts from the RTFM 'Groups Simple Stationary' game, which is more complex than Rock Paper Scissors but less so than Full RTFM. The dataset features are (rollout, NL task description) and the target is the two statements that can always be deduced about the environment dynamics from a winning episode. One of the statements is about the relationship between weapon type and monster type (which beats which), example: "Gleaming beats Fire.", and the other statement is about which monster belongs to which team, example: "wolf are order of the forest." We have an example dataset of 100k entries, which has been pickled and can be downloaded [here](https://drive.google.com/file/d/1EfaVA8q2FKZOxMn0NHcoD_LCrnagXpMi/view?usp=sharing).

Once downloaded, the code implementation of the observant scientist which uses scheduled teacher forcing and hidden state forcing can be found in the supervisedScientist/observant.ipynb notebook.