class RepeatDataset:
    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times

        self._origin_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._origin_len]

    def __len__(self):
        return self.times * self._origin_len
