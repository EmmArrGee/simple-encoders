import pandas as pd

from .base_encoder import BaseEncoder

class DFEncoder(BaseEncoder):    
    def __init__(self, config={}):
        super().__init__()
        self.__cols = []
        self.__encoding = {}
        if config:
            self.import_config(config)
        return
        
    def fit(self, df):
        super().fit(df.melt()['value'])
        self.__cols = list(df.columns)
        
    def set_encoding(self, encoding):
        missing_cols = [k for k in encoding.keys() if k not in self.__cols]
        if missing_cols:
            raise KeyError("Fitting not done for columns: {}".format(missing_cols))
            
        self.__encoding = encoding

    def get_encoding(self):
        return self.__encoding
        
    def encode(self, input):
        df = input.copy()
        
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
        
    def decode(self, input):
        df = input.copy()
        
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
        config['columns'] = (self.__cols)
        config['encoding'] = (self.__encoding)
        return config
    
    def set_config(self, config):
        super().set_config(config)
        if 'columns' in config.keys():
            self.__cols = config.get('columns')
        if 'encoding' in config.keys():
            self.__encoding = config.get('encoding')
            
    def get_base(self):
        return super()
