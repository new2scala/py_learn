
import torch.nn as nn
import torch
from tqdm import tqdm
from rnn_seq import RnnSeqNet
from torch.utils.data import Dataset, DataLoader


ENCODING = 'utf-8'
ROOT_PATH = 'Y:\\vmshare\\fp2Affs-full\\us-au-gb-ca\\'

from os import listdir
from os.path import isfile, join


def list_files(path):
    res = [f for f in listdir(path) if isfile(join(path, f))]
    return res


class RawDataSet(Dataset):
    def __init__(self, data_path, vocab_dict):
        self._data_files = list_files(data_path)
        self.vocab_dict = vocab_dict
        self._data = []

        for df in self._data_files:
            full_path = join(data_path, df)
            cat = int(df[0])
            with open(full_path, 'r', encoding=ENCODING) as f:
                lines = f.readlines()
                for line in lines:
                    words = line.split()
                    enc = [vocab_dict[w] for w in words]
                    self._data.append((enc, cat))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

    end_token = 0

    @staticmethod
    def normalize_batch_seq(batch):
        max_len = max([len(pair[0]) for pair in batch])+1
        res = torch.zeros(max_len, len(batch)).long()
        tgt = torch.zeros(1, len(batch)).long()
        for i, seq in enumerate(batch):
            sent, cat = seq[0], seq[1]
            res[:len(sent), i] = torch.tensor(sent)
            res[len(sent), i] = RawDataSet.end_token
            tgt[0, i] = cat
        return res, tgt


def load_dict(file):
    # dict = { }
    # rev_dict = { }
    # idx = 0
    with open(file, 'r', encoding=ENCODING) as f:
        lines = f.read().splitlines()
        dict = {w: i+1 for i, w in enumerate(lines)}
        rev_dict = {dict[w]: w for w in lines}
        print(len(dict))
        print(len(rev_dict))
        return dict, rev_dict


test_sents = [
    ("pflege und universitätsspital zürich", 4),
    ("observatory cape town", 4),
    ("and dentistry of new jersey new brunswick new jersey [[d5]]", 3),
    ("professor with the university of british columbia vancouver british columbia", 1),
    ("university of turku", 4),
    ("independent contractor williamsville ny", 3),
    ("spain and novo nordisk [[dk-d4]] bagsvaerd", 4),
    ("westmead hospital westmead nsw", 0),
    ("dentistry and pharmaceutical sciences okayama university", 4),
    ("rosedale mansions boulevard hull [[aad]] [[daa]]", 2),
    ("montana cancer consortium billings mt", 3),
    ("charing cross hospital london", 2),
    ("institut pasteur de la guyane cayenne cedex guyane", 4),
    ("virginia commonwealth university school of nursing richmond virginia", 3),
    ("102nd hospital of chinese pla", 4),
    ("gene experiment center institute of applied biochemistry university of tsukuba tsukuba-city", 4),
    ("education centre freeman hospital newcastle upon tyne", 2),
    ("best practice advocacy centre new zealand dunedin", 4),
    ("gastroenterology and hepatology medical university of vienna vienna", 4),
    ("department of psychology princeton university", 3),
]


def run_test_samples(rnn, vocab_dict):
    correct_count = 0
    for line, cat in test_sents:
        words = line.split()
        enc = [vocab_dict[w] for w in words]
        enc.append(RawDataSet.end_token)
        in_data = (
            torch.tensor([enc]).transpose(0, 1),
            torch.tensor([cat])
        )
        out, _ = rnn(in_data, RawDataSet.end_token)
        _, out_cat = out.data.topk(1)
        if out_cat == cat:
            correct_count += 1
        else:
            print(
                '\tError: A/E(%d/%d): %s' %
                (out_cat, cat, line)
            )
    print(
        '=============== summary %d/%d = %.2f%%' %
        (correct_count, len(test_sents), correct_count*100.0/len(test_sents))
    )

def train_pass1(batch_size, voc_file, data_file, state_file, sample_size):

    # raw_data = MolDataset(data_file)
    # encoder = OneHotEncoder(voc_file)
    # raw_data.generate_data_set_for_rnn(encoder)

    dict, rev_dict = load_dict(ROOT_PATH + 'all_words.txt')

    rnn = RnnSeqNet(          # cell_type='GRU',
        dictionary=dict,
        rev_dictionary=rev_dict,
        input_size=128,
        hidden_size=256,
        layer_num=3,
        out_sz=5,
        learning_rate=1e-3,
        criterion=nn.NLLLoss()
    )

    raw_data = RawDataSet(ROOT_PATH + 'train', dict)

    training_data = DataLoader(
        raw_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=RawDataSet.normalize_batch_seq
    )

#    params = rnn.parameters()
#    print(params)

    rates = []

    for epoch in range(6):

        data_len = len(training_data)
        rates_ep = []
        rates.append(rates_ep)
        for step, batch in tqdm(enumerate(training_data), total=data_len):
            # batch_input_lens = batch_lens(encoder.get_end_tag_index(), batch)

            def clos():
                rnn.get_optimizer().zero_grad()
                _, loss = rnn(
                    batch=batch
                )

                # out = out.permute(0,2,1)
                # loss = criterion(out, batch_target)
                # loss = loss.masked_select(batch_mask)
                # # todo: mask out padding
                loss = loss.mean()
                loss.backward()
                if step % 50 == 0:
                    print('Epoch {} step {} loss: {}'.format(epoch, step, loss.item()))

                if step > 0 and step % 500 == 0:
                    run_test_samples(rnn, dict)
                #     rate_avg, sample_loss = generate_check_samples(
                #         rnn=rnn,
                #         sample_count=100,
                #         encoder=encoder
                #     )
                #     rates_ep.append(rate_avg)
                #     print_rates(rates)
                #     rnn.save_state(state_file)
                #     dec_learning_rate2(step, rnn.get_optimizer(), rate_avg, 1e-6)

            rnn.get_optimizer().step(clos)


def verify_model(model_path, hidden_size, voc_file, sample_count, iterations):
    encoder = OneHotEncoder(voc_file)
    criterion = nn.CrossEntropyLoss(reduction='none')
    rnn = RnnSeqNet(          # cell_type='GRU',
        encoder=encoder,
        input_size=64,
        hidden_size=hidden_size,
        criterion=criterion
    )
    rnn.load_state(model_path)

    rates = []
    rates_ep = []
    rates.append(rates_ep)
    for _ in range(iterations):
        rate_avg, _ = generate_check_samples(
            rnn=rnn,
            sample_count=sample_count,
            encoder=encoder
        )
        rates_ep.append(rate_avg)
    print_rates(rates)


if __name__ == '__main__':
    # verify_model(
    #     model_path='data/rnn_seq_state_1024.ckpt',
    #     hidden_size=1024,
    #     voc_file='data/Voc',
    #     sample_count=200,
    #     iterations=10
    # )

    train_pass1(
        batch_size=256,
        voc_file='data/Voc',
        data_file='data/mols_filtered.smi',
        state_file='data/rnn_seq_state.ckpt',
        sample_size=200
    )
