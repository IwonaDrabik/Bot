import unittest
from app import get_response

class ActivityTests(unittest.TestCase):

    def test_get_response1(self):
        self.assertNotEqual(get_response('Could you define the notion of quantum physics?'), 'Yes, of course! It is the study of matter and energy at its most fundamental level.')

    def test_get_response2(self):
        self.assertEqual(get_response('I would like to buy some protein.'), 'Yes, you can buy WHEY protein, protein shales and protein bars. Water is sold in the multiple vending machines at the club.')

if __name__ == "__main__":
	unittest.main()
