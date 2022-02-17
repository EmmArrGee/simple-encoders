import pandas as pd

from .base_encoder import BaseEncoder


class DFEncoder(BaseEncoder):

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}
        self.__encoding = {}
        if config:
            self.set_config(config)
        return
        
    def fit(self, df):
        super().fit(df.melt()['value'])
        
    def set_encoding(self, encoding):
        self.__encoding = encoding
        
    def encode(self, df, **kwargs):
        df = df.copy()
        
        if not self.__encoding:
            raise RuntimeError('Encoding not specified. Call DFEncoder.set_encoding() to specify.')
            
        for col, encoding in self.__encoding.items():        
            if col in df.columns:            
                df[col] = df[col].apply(lambda val: super(DFEncoder, self).encode(val, encoding=encoding))                
                if encoding == 'binary':                
                    bin_df = df[col].apply(pd.Series)                    
                    col_names = {}
                    for c in bin_df.columns:
                        col_names[c] = '{}_{}'.format(col, len(bin_df.columns) - int(c) - 1)
                    bin_df.rename(columns=col_names, inplace=True)                    
                    df = pd.concat([df, bin_df], axis = 1)                    
                    df.drop(col, axis=1, inplace=True)
                    
        return df

    def decode(self, df, **kwargs):
        df = df.copy()
        
        for col, encoding in self.__encoding.items():
            if encoding == 'binary':
                cols_to_merge = [c for c in df.columns if c.startswith(col) and c != col]
                if cols_to_merge:
                    df[col] = df[cols_to_merge].astype(str).agg(''.join, axis=1)
                else:
                    df[col] = '0'
                df.drop(cols_to_merge, axis=1, inplace=True)
            if col in df.columns:
                df[col] = df[col].apply(lambda val: super(DFEncoder, self).decode(val, encoding=encoding))
                
        return df
        
    def get_config(self):
        config = super().get_config()
        config['encoding'] = self.__encoding
        return config
    
    def set_config(self, config):
        super().set_config(config)
        if 'encoding' in config.keys():
            self.__encoding = config.get('encoding')
            
    def get_base(self):
        return super()
