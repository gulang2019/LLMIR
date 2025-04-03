from datasets import DatasetDict, Dataset
from dataclasses import dataclass, asdict
from typing import List
from torch.utils.data import DataLoader

from LLMIR.structs import DataPoint, LSF, LANGUAGE_TO_NAME


@dataclass
class IRDataSet:
    datapoints: List[DataPoint]
    
    def __post_init__(self):
        self.indices = []
        for i, dp in enumerate(self.datapoints):
            self.indices.extend([(i, j) for j in range(len(dp.lsfs))])
    
    @staticmethod
    def from_file(file_path: str, languages: dict | None = None) -> 'IRDataSet':
        import json 
        datapoints = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                data = json.loads(line)
                datapoints.append(DataPoint.from_dict(data, languages))
        return IRDataSet(datapoints)
    
    @staticmethod
    def get_dataloader(filename: str = 'assets/data.jsonl', languages: dict | None = None, batch_size: int = 2) -> DataLoader:
        def _collate_fn(batch):
            return {
                'name': [item['name'] for item in batch],
                'problem': [item['problem'] for item in batch],
                'lang': [item['lang'] for item in batch],
                'header': [item['header'] for item in batch],
                'header_lang': [item['header_lang'] for item in batch],
                'tests': [item['tests'] for item in batch],  # likely a list of dicts or strings
                'stop_tokens': [item['stop_tokens'] for item in batch],  # list of lists of strings
            }
        dataset = IRDataSet.from_file(filename, languages)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn)
        return dataloader
    
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
            'name': datapoint.name,
            'problem': datapoint.problem,
            'lang': lsf.lang,
            'header': f"Given the intermediate language (IR) above, translate it into {language_name}."
                        "Please only generate the function without main or tests.\n"
                        f"```{lsf.lang}",
            'header_lang': lsf.header_no_desc,
            'tests': lsf.tests,
            'stop_tokens': list(lsf.stop_tokens)
        }

def load_and_convert_to_hf(
    dataset_path = 'assets/data.jsonl',
    languages: list[str] | None = None,
    train_test_split = 0.8) -> DatasetDict:
    # Load and shuffle datapoints
    dataset = IRDataSet.from_file(dataset_path, languages)
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
    
    dataloader = IRDataSet.get_dataloader('assets/data.jsonl', batch_size=2)
    for batch in dataloader:
        print(batch)
        '''problem, lang, header, tests'''
        print(batch.keys())
        break