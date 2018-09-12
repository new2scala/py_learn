
import unittest
from unittest_data_provider import data_provider


def show_items(a, b):
    print('{}/{}'.format(a, b))

class MixTests(unittest.TestCase):

    join_strs = lambda: (
        ('', ['a', 'b'], 'ab'),
        ('', ['a', 'b', 'x'], 'abx'),
        (' ', ['a', 'b'], 'a b'),
        (' ', ['a', 'b', 'x'], 'a b x')
    )

    @data_provider(join_strs)
    def test_join_strs(self, join_str, strs, exp_res):
        res = join_str.join(strs)
        self.assertEqual(res, exp_res)


    # test_lists = lambda: (
    #     ('a', 'b', 2)
    # )
    #
    #
    # @data_provider(test_lists)
    # def test_expand_list(self, lst, count):
    #     show_items(*lst)

lst1 = ['a', 'b']
show_items(*lst1)
lst2 = ['a', 5]
show_items(*lst2)
# lst3 = ['a', 5, False]
# show_items(*lst3)

t = zip ([1, 2, 3], ['a', 'b'])

for x in t:
    print(x)


if __name__ == '__main__':

    unittest.main()
