import pywood
import torch
import train
import numpy as np

GAME = np.random.randint(len(train.figures), size=300)
policy = train.TorchPolicy('1700.torch')
field = pywood.Field()
field.add_random([train.figures[f] for f in GAME[:3]])
triples = GAME[3:].reshape(-1, 3)
total_score = 0
for i in range(0, triples.shape[0]):
    figs = [train.figures[f] for f in triples[i, :]]
    if not any([field.can_place(f) for f in figs]):
        continue
    next_field, choice, score, _ = policy.play(field, triples[i, :], True)
    total_score += score
    if next_field is None:
        for fig in figs:
            print(fig)
        print(i, total_score)
        break
    print(field.print_choice(figs, choice))
    print(i, total_score)
    field = next_field
