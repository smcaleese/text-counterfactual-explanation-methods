import pandas as pd
import re

def sst2_formatter(text):
    # 1. remove the space between an apostrophe and s (e.g. "film ' s" -> "film 's")
    text = re.sub(r"\s'\s+s", " 's", text)

    # 2. add a space before full stops or commands (e.g. "film." -> "film ."):
    text = re.sub(r"(\w)([.,])", r"\1 \2", text)

    # 3. replace spaces around hyphens:
    text = re.sub(r"\s-\s", "-", text)
    
    return text

def format_sentence(text):
    pattern = r"\s([.!,?:;')])"
    
    def replace(match):
        return match.group(1)
    
    return re.sub(pattern, replace, text)

# Test the function
test_cases = [
    "hello .",
    "hello , world",
    "What ? Really ?",
    "Oh ; I see",
    "Wow : amazing",
    "This is a test !"
]

for case in test_cases:
    print(f"Original: '{case}'")
    print(f"Formatted: '{format_sentence(case)}'")
    print()
