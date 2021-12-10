import pywood
import train

field = pywood.Field()
policy = train.TorchPolicy('1700.torch')
figures = list([str(f) for f in train.figures])
figs = None
def next(a, b, c):
    a, b, c = [x.replace(',', '\n') for x in [a, b, c]]
    A = pywood.Figure(a)
    if A not in train.figures:
        print(f"{A} not found")
        return;
    B = pywood.Figure(b)
    if B not in train.figures:
        print(f"{B} not found")
        return;
    C = pywood.Figure(c)
    if C not in train.figures:
        print(f"{C} not found")
        return;
    global figs
    figs = (figures.index(str(A)), figures.index(str(B)), figures.index(str(C)))
    for fig in figs:
        print(train.figures[fig])

def n(a):
    next(*a.split(',,'))

def go():
    global field, prev_field
    next_field, choice, score, _ = policy.play(field, figs, True)
    if choice is None:
        print('Sorry :(')
        return
    print(field.print_choice([train.figures[f] for f in figs], choice))
    prev_field = field
    field = next_field

def new():
    global field
    field = pywood.Field()

def undo():
    global field, prev_field
    field, prev_field = prev_field, field
