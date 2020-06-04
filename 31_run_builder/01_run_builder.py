from collections import OrderedDict
from collections import namedtuple
from itertools import product

class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

params = OrderedDict(
    lr = [.01, .001]
    ,batch_size = [1000, 10000]
)

runs = RunBuilder.get_runs(params)
runs

run = runs[0]
run

print(run.lr, run.batch_size)

for run in runs:
    print(run, run.lr, run.batch_size)

params = OrderedDict(
    lr = [.01, .001]
    ,batch_size = [1000, 10000]
    ,device = ["cuda", "cpu"]
)

runs = RunBuilder.get_runs(params)
runs

params = OrderedDict(
    lr = [.01, .001]
    ,batch_size = [1000, 10000]
)

params.keys()

params.values()

Run = namedtuple('Run', params.keys())

runs = []
for v in product(*params.values()):
    runs.append(Run(*v))
runs

for run in RunBuilder.get_runs(params):
    comment = f'-{run}'
    print(f'comment:{comment} lr={run.lr}, batch_size={run.batch_size}')
