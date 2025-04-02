from torch.utils.data.dataset import Dataset
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List
import tqdm
import os 
import json

class Language(Enum):
    PYTHON = "py"
    JAVASCRIPT = "js"
    TYPESCRIPT = "ts"
    JAVA = "java"
    D = "d"
    CPP = "cpp"
    R = "r"
    RUST = "rs"
    JULIA = "jl"
    BASH = "sh"
    CSHARP = "cs"
    GO = "go"
    LUA = "lua"
    PERL = "pl"
    PHP = "php"
    RUBY = "rb"
    SCALA = "scala"
    SWIFT = "swift"
    RACKET = "rkt"
    OCAML = "ml"
    HASKELL = "hs"
    ELIXIR = "elixir"
    CLOJURE = "clj"
    ADA = "ada"

LANGUAGE_TO_NAME = {
    Language.PYTHON.value: "Python",
    Language.JAVASCRIPT.value: "JavaScript",
    Language.TYPESCRIPT.value: "TypeScript",
    Language.JAVA.value: "Java",
    Language.D.value: "D",
    Language.CPP.value: "C++",
    Language.R.value: "R",
    Language.RUST.value: "Rust",
    Language.JULIA.value: "Julia",
    Language.BASH.value: "Bash",
    Language.CSHARP.value: "C#",
    Language.GO.value: "Go",
    Language.LUA.value: "Lua",
    Language.PERL.value: "Perl",
    Language.PHP.value: "PHP",
    Language.RUBY.value: "Ruby",
    Language.SCALA.value: "Scala",
    Language.SWIFT.value: "Swift",
    Language.RACKET.value: "Racket",
    Language.OCAML.value: "OCaml",
    Language.HASKELL.value: "Haskell",
    Language.ELIXIR.value: "Elixir",
    Language.CLOJURE.value: "Clojure",
    Language.ADA.value: "Ada",
}

# language specific fields
@dataclass
class LSF:
    lang: Language
    header: str
    header_no_desc: str 
    tests: List[str]
    stop_tokens: List[str]
    gold_code: str = ""

@dataclass
class DataPoint:
    name: str
    problem: str 
    lsfs: List[LSF]
    
    @staticmethod
    def from_dict(d) -> 'DataPoint':
        return DataPoint(
            name=d['name'],
            problem=d['problem'],
            lsfs=[LSF(**lsf) for lsf in d['lsfs']]
        )
    
@dataclass
class RawDatapoint:
    # {"name": "HumanEval_0_has_close_elements", "language": "elixir", "prompt": "# Check if in given list of numbers, are any two numbers closer to each other than\n# given threshold.\n# >>> HumanEval.has_close_elements([1.0, 2.0, 3.0], 0.5)\n# false\n# >>> HumanEval.has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n# true\n\ndefmodule HumanEval do\n  def candidate(numbers, threshold), do: has_close_elements(numbers, threshold)\n  def has_close_elements(numbers, threshold) do\n    ", "doctests": "transform", "original": "/ssd1/siyuanch/workspace/LLMIR/3rdparty/MultiPL-E/datasets/../datasets/originals-with-cleaned-doctests/HumanEval_0_has_close_elements.py", "prompt_terminology": "reworded", "tests": "ExUnit.start()\ndefmodule HumanEvalTest do\n  use ExUnit.Case, async: true\n  test 'has_close_elements' do\n    assert true == HumanEval.candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3)\n    assert false == HumanEval.candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05)\n    assert true == HumanEval.candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95)\n    assert false == HumanEval.candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8)\n    assert true == HumanEval.candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1)\n    assert true == HumanEval.candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0)\n    assert false == HumanEval.candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5)\n  end\nend\n", "stop_tokens": ["\ndefmodule", "\ndefp", "\ndef ", "\n#", "\n\n"]}
    name: str
    language: str
    prompt: str 
    doctests: str
    original: str
    tests: str
    prompt_terminology: str
    stop_tokens: List[str]


def load_raw_data_from_json(filepath) -> List[RawDatapoint]:
    with open(filepath, 'r') as f:
        raw_data = f.readlines()
    raw_datapoints = []
    for raw_datapoint in raw_data:
        raw_datapoint = json.loads(raw_datapoint)
        raw_datapoint = RawDatapoint(**raw_datapoint)
        raw_datapoints.append(raw_datapoint)
    return raw_datapoints
    
def _process_raw_data(langs: List[Language], datasets: List[str]):
    raw_dir = 'assets/raw'
    all_datapoints = []
    for dataset in datasets:
        nl_dataset_path = f'{raw_dir}/{dataset}-nl-transform.jsonl'
        raw_datapoints = load_raw_data_from_json(nl_dataset_path)
        datapoints: List[DataPoint] = []
        for raw_dp in raw_datapoints:
            datapoints.append(DataPoint(
                name=raw_dp.name,
                problem=raw_dp.prompt,
                lsfs = []
            ))
        for lang in tqdm.tqdm(langs, desc = f'Processing {dataset}'):
            lang_dataset_path_reworded = f'{raw_dir}/{dataset}-{lang.value}-reworded.jsonl'
            lang_dataset_path_remove = f'{raw_dir}/{dataset}-{lang.value}-remove.jsonl'
            if not (os.path.exists(lang_dataset_path_reworded) and\
                os.path.exists(lang_dataset_path_remove)):
                print(f"Skipping {lang_dataset_path_reworded} and {lang_dataset_path_remove}")
                continue
            lang_datapoints_reworded = load_raw_data_from_json(lang_dataset_path_reworded)
            name2id_reworded = {dp.name: i for i, dp in enumerate(lang_datapoints_reworded)} 
            lang_datapoints_remove = load_raw_data_from_json(lang_dataset_path_remove)
            name2id_remove = {dp.name: i for i, dp in enumerate(lang_datapoints_remove)}
            
            for dp in datapoints:
                if dp.name not in name2id_reworded or dp.name not in name2id_remove:
                    continue
                lang_dp_remove = lang_datapoints_remove[name2id_remove[dp.name]]
                lang_dp_reworded = lang_datapoints_reworded[name2id_reworded[dp.name]]
                dp.lsfs.append(LSF(
                    lang = lang.value, 
                    header = lang_dp_reworded.prompt,
                    header_no_desc=lang_dp_remove.prompt,
                    tests = lang_dp_reworded.tests,
                    stop_tokens = lang_dp_reworded.stop_tokens
                ))
        all_datapoints.extend(datapoints)
    return all_datapoints

def process_raw_data():

    datapoints = _process_raw_data([
        Language.PYTHON,
        Language.JAVASCRIPT,
        Language.TYPESCRIPT,
        Language.JAVA,
        Language.D,
        Language.CPP,
        Language.R,
        Language.RUST,
        Language.JULIA,
        Language.BASH,
        Language.CSHARP,
        Language.GO,
        Language.LUA,
        Language.PERL,
        Language.PHP,
        Language.RUBY,
        Language.SCALA,
        Language.SWIFT,
        Language.RACKET,
        Language.OCAML,
        Language.HASKELL,
        Language.ELIXIR,
        Language.CLOJURE,
        Language.ADA
    ], ['humaneval', 'mbpp', 'leetcode'])

    # store the datapoint into json
    import json 
    with open('assets/data.jsonl', "w", encoding="utf-8") as f:
        for dp in datapoints:
            f.write(json.dumps(asdict(dp)) + "\n")
            
def data_investigation():
    data_path = 'assets/data.jsonl'
    with open(data_path, 'r') as f:
        data = f.readlines()
    print(len(data))
    dps: List[DataPoint] = []
    for line in data:
        line = json.loads(line)
        # load the data by DataPoint
        dp = DataPoint.from_dict(line)
        dps.append(dp)
    example_dp = dps[0]
    print(example_dp.name, example_dp.problem, type(example_dp.lsfs))
    for lsf in example_dp.lsfs:
        print('Language:', lsf.lang)
        print('*'* 100)
        print('Header:', lsf.header)
        print('*' * 100)
        print('Header no desc:', lsf.header_no_desc)
        print('*' * 100)
        print('Tests:', lsf.tests)
        print('*' * 100)
    import numpy as np
    n_lanaguages = [len(dp.lsfs) for dp in dps]
    print('#Data:', len(data))
    names = [dp.name for dp in dps]
    print(names)
    print('Languages in total', len(Language.__members__))
    print('Number of languages:', np.mean(n_lanaguages), '+-', np.std(n_lanaguages))
if __name__ == '__main__':
    process_raw_data()
    data_investigation()