import unittest
from string_formatter import format_sentence

class TestStringFormatter(unittest.TestCase):
    def test_format_sentence(self):
        test_cases = [
            ("hi there !", "hi there!"),
            ("oh ; i see", "oh; i see"),
            ("wow : amazing", "wow: amazing"),
            ("the film 's center will not hold .", "the film's center will not hold."),
            ("ellis '", "ellis'"),
            ("it 's", "it's"),
            ("pretty darn good , despite", "pretty darn good, despite"),
            ("( it 's ) a prison soccer movie", "(it's) a prison soccer movie"),
            ("this opera is n't a favorite", "this opera isn't a favorite"),
            ("you ' ve", "you've"),
            ("one of the most entertaining thrillers i ' ve seen in quite a long time.", "one of the most entertaining thrillers i've seen in quite a long time."),
            ("we ' re", "we're"),
            ("they ' re", "they're"),
            ("you ' ll", "you'll"),
            ("it ' s a movie - - and an album - - you won ' t want to miss .", "it's a movie -- and an album -- you won't want to miss."),
            ("you can ' t wait", "you can't wait"),
            ("it ' s a period", "it's a period"),
            ("self - aware", "self-aware"),
            ("doesn ' t", "doesn't"),
            ("he ' d create a movie better than this.", "he'd create a movie better than this."),
            ("leave these flowers unpicked - - they ' re life on the vine.", "leave these flowers unpicked -- they're life on the vine."),
            ("here ' s", "here's"),
            ("here 's", "here's")
        ]

        for input_string, expected_output in test_cases:
            with self.subTest(input=input_string):
                result = format_sentence(input_string)
                self.assertEqual(result, expected_output)

if __name__ == "__main__":
    unittest.main()
