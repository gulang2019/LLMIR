from datasets import DatasetDict, Dataset
from dataclasses import dataclass, asdict
from typing import List

from LLMIR.structs import DataPoint, LSF, LANGUAGE_TO_NAME


@dataclass
class IRDataSet:
    datapoints: List[DataPoint]
    
    def __post_init__(self):
        self.indices = []
        for i, dp in enumerate(self.datapoints):
            self.indices.extend([(i, j) for j in range(len(dp.lsfs))])
    
    @staticmethod
    def from_file(file_path: str) -> 'IRDataSet':
        import json 
        datapoints = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                data = json.loads(line)
                datapoints.append(DataPoint.from_dict(data))
        return IRDataSet(datapoints)

    @property
    def len(self):
        return sum([len(dp.lsfs) for dp in self.datapoints])

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        data_id, lsf_id = self.indices[idx]
        datapoint: DataPoint = self.datapoints[data_id]
        lsf: LSF = datapoint.lsfs[lsf_id]
        language_name = LANGUAGE_TO_NAME[lsf.lang]
        return {
            'problem': datapoint.problem,
            'lang': lsf.lang,
            'header': f"Given the intermediate language (IR) above, translate it into {language_name}."
                        "Please only generate the function without main or tests.\n"
                        f"```{lsf.lang}",
            'tests': lsf.tests,
            'stop_tokens': lsf.stop_tokens
        }

def load_and_convert_to_hf(
    dataset_path = 'assets/data.jsonl', 
    train_test_split = 0.8) -> DatasetDict:
    # Load and shuffle datapoints
    dataset = IRDataSet.from_file(dataset_path)
    datapoints = dataset.datapoints
    import random
    random.shuffle(datapoints)

    # Split at DataPoint level
    split_idx = int(train_test_split * len(datapoints))
    train_dps = datapoints[:split_idx]
    test_dps = datapoints[split_idx:]

    # Wrap back into IRDataSet
    train_dataset = IRDataSet(train_dps)
    test_dataset = IRDataSet(test_dps)

    # Convert to HF Datasets
    hf_dataset = DatasetDict({
        "train": Dataset.from_list([train_dataset[i] for i in range(len(train_dataset))]),
        "test": Dataset.from_list([test_dataset[i] for i in range(len(test_dataset))]),
    })
    
    return hf_dataset

if __name__ == '__main__':
    dataset = IRDataSet.from_file('assets/data.jsonl')
    print(dataset.len)
    print(dataset[0])
    
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        print(batch)
        '''problem, lang, header, tests'''
        print(batch.keys())
        break