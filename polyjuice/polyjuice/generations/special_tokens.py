PERETURB_TOK = "<|perturb|>"
BLANK_TOK = "[BLANK]"
SEP_TOK = "[SEP]"
EMPTY_TOK = "[EMPTY]"
ANSWER_TOK = "[ANSWER]"

# control codes
NEGATION = "negation"
QUANTIFIER = "quantifier"
SHUFFLE = "shuffle"
LEXCICAL = "lexical"
RESEMANTIC = "resemantic"
INSERT = "insert"
DELETE = "delete"
RESTRUCTURE = "restructure"

RANDOM_CTRL_CODES = [
    LEXCICAL, RESEMANTIC, NEGATION, INSERT, DELETE
]
ALL_CTRL_CODES = set([
    LEXCICAL, RESEMANTIC, NEGATION, INSERT, 
    DELETE, QUANTIFIER, RESTRUCTURE, SHUFFLE
])