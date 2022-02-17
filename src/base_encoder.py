class BaseEncoder:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.__encoder = {}
        self.__decoder = {}
        self.__len_bin = 0
        if config:
            self.set_config(config)
        return
    
    def fit(self, data):
        for value in data:
            if value not in self.__encoder and value is not None:
                n = len(self.__encoder) + 1
                self.__encoder[value] = n
                self.__decoder[n] = value
        self.__init_binary_encoder()
            
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

    def __encode_basic(self, value):
        if value in self.__encoder:
            return self.__encoder[value]
        else:
            return 0
    
    def __encode_to_label(self, value):
        return self.__encode_basic(value)
            
    def __encode_to_binary(self, value):
        enc_value = self.__encode_basic(value)
        b = bin(enc_value)
        s = str(b)[2:]
        s = s.zfill(self.__len_bin)
        d = [int(v) for v in s]
        return d
        
    def __decode_basic(self, value):
        if value in self.__decoder:
            return self.__decoder[value]
        else:
            return None

    def __decode_from_label(self, value):
        return self.__decode_basic(value)
            
    def __decode_from_binary(self, value):
        s = ''.join([str(v) for v in value])
        s = '0b{}'.format(s)
        dec_value = int(s, 2)
        return self.__decode_basic(dec_value)
        
    def get_config(self):
        config = {
            'encoder': self.__encoder
        }
        return config
    
    def set_config(self, config):
        self.__encoder = config.get('encoder')
        self.__decoder = {}
        for key, value in self.__encoder.items():
            self.__decoder[value] = key
        self.__init_binary_encoder()
            
    def __init_binary_encoder(self):
        max_value = max(self.__encoder.values())
        max_value_b = str(bin(max_value))[2:]
        self.__len_bin = len(max_value_b)
