
root_path = 'Y:\\vmshare\\fp2Affs-full\\us-au-gb-ca\\'
encoding = 'utf-8'

def read_words(file):
    res_set = set()
    with open(file, 'r', encoding=encoding) as f:
        for line in f.readlines():
            words = line.split()
            res_set = res_set.union(words)

    return res_set

def save_words(out_file, lines):
    with open(out_file, 'w', encoding=encoding) as f:
        for line in lines:
            f.write(line + '\n')


def words_in_files():
    file_indices = range(5)

    for file_index in file_indices:
        file_name = root_path + 'train\\%d.txt' % file_index
        s1 = read_words(file_name)
        print(len(s1))
        out_file_name = root_path + 'words_%d.txt' % file_index
        save_words(out_file_name, s1)


def merge_words():
    file_indices = range(5)

    res = set()
    for file_index in file_indices:
        file_name = root_path + 'words_%d.txt' % file_index
        s1 = read_words(file_name)
        # print(len(s1))
        res = res.union(s1)
        print(len(res))

    out_file_name = root_path + 'all_words.txt'
    save_words(out_file_name, res)


# merge_words()
def load_dict(file):
    # dict = { }
    # rev_dict = { }
    # idx = 0
    with open(file, 'r', encoding=encoding) as f:
        lines = f.read().splitlines()
        dict = { w : i for i, w in enumerate(lines) }
        rev_dict = { dict[w] : w for w in lines}
        print(len(dict))
        print(len(rev_dict))
        return dict, rev_dict


dict, rev_dict = load_dict(root_path + 'all_words.txt')
#s1 = read_words(root_path + 'train\\0.txt')
