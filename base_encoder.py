class BaseEncoder:
    def __init__(self, config={}): 
        self.__encoder = {}
        self.__decoder = {}
        self.__len_bin = 0
        if config:
            self.import_config(config)
        return
    
    def fit(self, data):
        for value in data:
            if value not in self.__encoder:
                n = len(self.__encoder) + 1
                self.__encoder[value] = n
                self.__decoder[n] = value
        
        max_value = max(self.__encoder.values())
        max_value_b = str(bin(max_value))[2:]
        self.__len_bin = len(max_value_b)
            
    def encode(self, value, encoding='label'):
        if encoding == 'label':
            return self.__encode_to_label(value)
        elif encoding == 'binary':
            return self.__encode_to_binary(value)
        else:
            return value
            
    def decode(self, value, encoding='label'):
        if encoding == 'label':
            return self.__decode_from_label(value)
        elif encoding == 'binary':
            return self.__decode_from_binary(value)
        else:
            return value
    
    def __encode_to_label(self, value):
        if value in self.__encoder:
            return self.__encoder[value]
        else:
            return 0
            
    def __encode_to_binary(self, value):
        enc_value = self.__encode_to_label(value)
        b = bin(enc_value)
        s = str(b)[2:]
        s = s.zfill(self.__len_bin)
        d = list(s)
        return d
        
    def __decode_from_label(self, value):
        if value in self.__decoder:
            return self.__decoder[value]
        else:
            return None
            
    def __decode_from_binary(self, value):
        s = ''.join(value)
        s = '0b{}'.format(s)
        dec_value = int(s, 2)
        return self.__decode_from_label(dec_value)
        
    def get_config(self):
        config = {
            'encoder': self.__encoder,
            'decoder': self.__decoder
        }
        return config
    
    def set_config(self, config):
        self.__encoder = config.get('encoder')
        self.__decoder = config.get('decoder')
