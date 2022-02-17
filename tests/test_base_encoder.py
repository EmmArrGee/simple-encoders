from unittest import TestCase
from src.base_encoder import BaseEncoder
import pickle


class TestBaseEncoder(TestCase):
    train = [0, 1, '', 'a', 'abc', 'ab ab', 'ab 1']
    other = [2, 10, ' ', 'b', 'cba', '  a  ', '12 a', None]

    encodings = ['label', 'binary']

    def test_fit(self):
        train = self.train * 2

        e = BaseEncoder()
        e.fit(train)

        self.assertListEqual(
            self.train,
            list(e.get_config().get('encoder').keys()))
        self.assertListEqual(
            [],
            [v for v in range(1, len(self.train) + 1) if v not in e.get_config().get('encoder').values()])
        self.assertEqual(
            len(train) / 2,
            len(e.get_config().get('encoder')))

    def test_encode_decode_default(self):
        e = BaseEncoder()
        e.fit(self.train)

        for v in self.train:
            enc = e.encode(v)
            dec = e.decode(enc)
            self.assertEqual(dec, v)

        for v in self.other:
            enc = e.encode(v)
            dec = e.decode(enc)
            self.assertIsNone(dec)

    def test_encode_decode(self):
        e = BaseEncoder()
        e.fit(self.train)

        for encoding in self.encodings:

            for v in self.train:
                enc = e.encode(v, encoding=encoding)
                dec = e.decode(enc, encoding=encoding)
                self.assertEqual(dec, v)

            for v in self.other:
                enc = e.encode(v)
                dec = e.decode(enc)
                self.assertIsNone(dec)

    def test_encode_binary_unknown(self):
        e = BaseEncoder()
        e.fit(self.train)
        for v in self.other:
            enc = e.encode(v, encoding='binary')
            self.assertListEqual([], [d for d in enc if d != 0])

    def test_decode_binary_unknown(self):
        bin_len = 3

        e = BaseEncoder()
        e.fit(list(range(2 ** (bin_len - 1))))

        unknown_val = [1] * bin_len
        dec = e.decode(unknown_val, encoding='binary')
        self.assertIsNone(dec)

    def test_config_save_and_restore(self):
        e = BaseEncoder()
        e.fit(self.train)

        cfg_s = pickle.dumps(e.get_config())
        cfg_r = pickle.loads(cfg_s)

        r = BaseEncoder(cfg_r)

        for v in self.train:
            d = r.decode(e.encode(v))
            self.assertEqual(v, d)

            d = e.decode(r.encode(v))
            self.assertEqual(v, d)
