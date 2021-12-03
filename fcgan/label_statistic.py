import numpy as np


class LabelStatistic:

    def __init__(self, ds):
        """
        :param ds: {mocap.datasets.dataset.Dataset}
        """
        assert ds.n_data_entries == 2

        lookup = {}
        for sid in range(len(ds)):
            seq, labels = ds[sid]
            print('labels', labels.shape)
            exit(1)
