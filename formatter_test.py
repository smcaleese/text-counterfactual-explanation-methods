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

def main():
    # df = pd.read_csv("output/hotflip-output.csv")
    # df["counterfactual_text"] = df["counterfactual_text"].apply(sst2_formatter)
    # df.to_csv("output/hotflip-output-2.csv", index=False)

    # s = "i saw knockaround guys yesterday , and already the details have faded like photographs from the spanish-american war ... it 's so unmemorable that it turned my ballpoint notes to invisible ink ."
    original_text = "the film ' s center will, not hold."
    counterfactual_text = sst2_formatter(original_text)

    label_width = 20
    print(f"\n{'original_text:'.ljust(label_width)} {original_text}")
    print(f"{'counterfactual_text:'.ljust(label_width)} {counterfactual_text}")

    s1 = "despite what anyone believes about the goal of its makers , the show ... represents a spectacular piece of theater , and there 's no denying the talent of the creative forces behind it ."
    s2 = "despite what anyone believes about the goal of its makers , the show ... represents a dummy piece of theater , and there 's no denying the talent of the creative forces behind it ."

if __name__ == "__main__":
    main()
