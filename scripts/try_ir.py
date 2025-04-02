from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # corrected model hub path

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).cuda()  # or .to("cpu") if no GPU

# Define your prompt
prompt = 'You are a helpful coding assistant. You are given a problem description that you need to write a function in a language agonistic intermediate language (IR).'\
        'The format is ```IR\n...\n```\n'\
        'The function \'count_upper\' takes arguments s: str and returns int.'\
        'Description: Given a string s, count the number of uppercase vowels in even indices.\n'\
        'For example:\n>>> a function call to a variable named count_upper with arguments: a string "aBCdEf"\na number 1\n'\
        '>>> a function call to a variable named count_upper with arguments: a string "abcdefg"\na number 0\n'\
        ';>>> a function call to a variable named count_upper with arguments: a string "dBBE"\na number 0\n'\
        'Now, Implement the intermediate language. Stop directly when the IR is generated. Generate only a single function, no main, no test.\n```IR\n'


prompt_ir_to_py = '''```IR
function count_upper(s: str) -> int {
    let upper_count = 0;
    for (let i = 0; i < s.length(); i += 2) {
        if (s[i] === 'A' || s[i] === 'E' || s[i] === 'I' || s[i] === 'O' || s[i] === 'U') {
            upper_count++;
        }
    }
    return upper_count;
}
```
Given the intermediate language (IR) above, translate it into Rust. Please only generate the function without main.\n```rs'''


# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate output
output_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
    do_sample=True,
    top_p=0.95,
    temperature=0.8
)

# Decode and print the generated text
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_text)

