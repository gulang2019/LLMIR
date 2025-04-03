import re

def guided_decoding_regex(endings):
    escaped = [re.escape(e) for e in endings]
    pattern = f"({'|'.join(escaped)})$"
    return pattern

if __name__ == '__main__':
    # Example usage
    endings = ["foo", "bar", "baz"]
    regex = guided_decoding_regex(endings)
    print(regex)  # Output: (foo|bar|baz)$