import pywood
import numpy as np

figure = pywood.Figure('xxx\nx\nx')
print(figure)

with open('figures.txt', "rt") as file:
    figs = file.read()
figures = pywood.Figure.read(figs)
print(len(figures))

field = pywood.Field()
print(field)

cnt = field.count_placements(figure)
print(cnt)

field.add(figure, 1, 1)
print(field)

field.add_random(figures[3:8], 1)
print(field)

figs = (figures[15], figures[18], figures[20])
for fig in figs:
    print(fig)
next_fields, choices, score = field.get_all_next(figs, True)
print(score)
print(len(next_fields))
if next_fields:
    print(next_fields[15], choices[15], next_fields[15].weight)
print(field.print_choice(figs, choices[15]))

np_fields = next_fields.to_numpy()
print(np_fields.shape)
print(np_fields[15, :, :])
