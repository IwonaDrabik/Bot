import unittest
from nlp import tokenize, stem, bag_of_words
from pythonProject.app import get_response


class ActivityTests(unittest.TestCase):
    def setUp(self):
        self.tokenized_sentence = (["hi"])
        self.words = ["hi"]

    def test_tokenize(self):
        self.assertEqual(tokenize('I like broccoli.'),['I', 'like', 'broccoli', '.'])

    def test_stem(self):
        self.assertEqual(stem("drawing"),"draw")

    def test_bag_of_words(self):
        self.assertEqual(bag_of_words(self.tokenized_sentence, self.words),([1.]))

    def test_get_response(self):
        self.assertEqual(get_response('I would like to buy an angle grinder.'), "Sorry! Can't help you with this one. Try this from my Friend Google https://www.google.com.")


if __name__ == "__main__":
	unittest.main()
