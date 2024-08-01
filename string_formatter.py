import re

def format_sentence(sentence):
    sentence = sentence.lower()

    # remove two spaces around a comma:
    sentence = re.sub(r"\s(')\s(ve|re|s|t|ll|d)", r"\1\2", sentence)

    # remove spaces around hyphens:
    sentence = re.sub(r"-\s-", "--", sentence)
    sentence = re.sub(r"(\w)\s-\s(\w)", r"\1-\2", sentence)

    def replace(match):
        return match.group(1)

    # remove spaces before punctuation and n't:
    sentence = re.sub(r"\s([.!,?:;')]|n't)", replace, sentence)

    # remove spaces after opening parenthesis:
    sentence = re.sub(r"([(])\s", replace, sentence)
    
    return sentence
