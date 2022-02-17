from unittest import TestCase
from src.df_encoder import DFEncoder
import pandas as pd
import pickle


class TestDFEncoder(TestCase):
    train = pd.DataFrame(
        [['a', 'a', 'a'],
         ['b', 'b', 'b'],
         ['c', 'c', 'c'],
         ['', '', '']],
        columns=['col1', 'col2', 'col3']
    )

    check_df = pd.DataFrame(
        [['a', 1, 0, 0, 1],
         ['b', 2, 0, 1, 0],
         ['c', 3, 0, 1, 1],
         ['', 4, 1, 0, 0]],
        columns=['col1', 'col2', 'col3_2', 'col3_1', 'col3_0']
    )

    other_to_encode = pd.DataFrame(
        [['d', 'd', 'd', 'd'],
         [' ', ' ', ' ', ' '],
         [1, 1, 1, 1],
         [None, None, None, None]],
        columns=['col1', 'col2', 'col3', 'col4']
    )

    other_to_decode = pd.DataFrame(
        [[5, 5, 1, 0, 1],
         [0, 0, 0, 0, 0]],
        columns=['col1', 'col2', 'col3_2', 'col3_1', 'col3_0']
    )

    def test_encode(self):
        e = DFEncoder()
        e.fit(self.train)
        e.set_encoding({
            'col2': 'label',
            'col3': 'binary'
        })

        enc = e.encode(self.train)

        self.assertListEqual(list(enc.values.flatten()), list(self.check_df.values.flatten()))
        self.assertListEqual(list(enc.columns), list(self.check_df.columns))

    def test_decode(self):
        e = DFEncoder()
        e.fit(self.train)
        e.set_encoding({
            'col2': 'label',
            'col3': 'binary'
        })

        dec = e.decode(self.check_df)

        self.assertListEqual(list(dec.values.flatten()), list(self.train.values.flatten()))
        self.assertListEqual(list(dec.columns), list(self.train.columns))

    def test_unknown_data(self):
        e = DFEncoder()
        e.fit(self.train)
        e.set_encoding({
            'col2': 'label',
            'col3': 'binary'
        })

        enc = e.encode(self.other_to_encode)

        bin_cols = [col for col in enc.columns if col.startswith('col3')]

        self.assertFalse(enc['col2'].values.any())
        self.assertFalse(enc[bin_cols].values.any())

        dec = e.decode(self.other_to_decode)

        self.assertListEqual(list(dec['col1'].values.flatten()), list(self.other_to_decode['col1'].values.flatten()))
        for v in dec[['col2', 'col3']].values.flatten():
            self.assertIsNone(v)

    def test_decode_partial_data(self):
        e = DFEncoder()
        e.fit(self.train)
        e.set_encoding({
            'col2': 'binary',
            'col3': 'binary'
        })

        enc = e.encode(self.train[['col2', 'col3']])

        bin_cols = [col for col in enc.columns if col.startswith('col2')]

        dec = e.decode(enc[bin_cols].iloc[:1])
        orig_slice = self.train['col2'].iloc[:1]

        self.assertListEqual(list(dec['col2'].values), list(orig_slice.values))

        for v in dec['col3'].values:
            self.assertIsNone(v)

    def test_config_save_and_restore(self):
        e = DFEncoder()
        e.fit(self.train)
        e.set_encoding({
            'col2': 'binary',
            'col3': 'binary'
        })

        cfg_s = pickle.dumps(e.get_config())
        cfg_r = pickle.loads(cfg_s)

        r = DFEncoder(cfg_r)

        self.assertListEqual(
            list(r.decode(e.encode(self.train)).values.flatten()),
            list(self.train.values.flatten()))
        self.assertListEqual(
            list(r.decode(e.encode(self.train)).columns),
            list(self.train.columns)
        )
        self.assertListEqual(
            list(e.decode(r.encode(self.train)).values.flatten()),
            list(self.train.values.flatten()))
        self.assertListEqual(
            list(e.decode(r.encode(self.train)).columns),
            list(self.train.columns)
        )
