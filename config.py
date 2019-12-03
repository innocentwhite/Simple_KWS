class Config:
    ''' Pytorch training config'''
    def __init__(self):
        self.Base_dir = '/home/liucl/Proj/Simple_KWS/'

        self.workers = 4 # number of data loading workers
        
        # Model info
        self.arch = 'gru'
        self.model_size_info = [1,16]

        # Training config
        self.resume = ''
        self.evaluate = False
        self.epochs = 60 # training loops
        self.batch_size = 100
        self.lr = 0.005 # learning rate
        self.momentum = 0.9 
        self.weight_decay = 1e-4
        self.optimizer_type = 'adam' # sgd or adam
        self.lr_scheduler = 'default'

        # Data config
        self.data_url = None
        self.data_dir = '/home/liucl/tmp'
        self.wanted_words = 'up,no,yes' # yes,no,up,down,left,right,on,off,stop,go
        self.background_volume = 0.0 # How loud the background noise should be, between 0 and 1
        self.background_frequency = 0.0 # How many of the training samples have background noise mixed in
        self.silence_percentage = 10.0
        self.unknown_percentage = 10.0
        self.testing_percentage = 10
        self.validation_percentage = 10
        self.sample_rate = 16000
        self.time_shift_ms = 100.0 # Range to randomly shift the training audio by in time
        self.clip_duration_ms = 1000 # Expected duration in milliseconds of the wavs
        self.window_size_ms = 40.0
        self.window_stride_ms = 40.0
        self.dct_coefficient_count = 10

        # ADMM config
        self.admm_epochs = 5
        self.admm_quant = False
        self.quant_type = 'fixed' # binary ternary fixed
        self.reg_lambda = 1e-4 # initial rho for all layers
        self.verbose = False # whether to report admm convergence condition
        self.num_bits = 8
        self.quant_val = False # whether to use quantize model

        # Activation quantization config
        self.act_bits = 0 # 0 means no activation quantization
        self.act_max = None # activation value's integer part's max value
        self.save_act_value = False # whether to save activation value to file
        self.save_act_dir = self.Base_dir + 'act_value/'
        self.coverage = 1.0 # the percentage of -128~128 covers the whole range of data

        # Other config
        self.logger = False
        self.no_cuda = False
        self.print_freq = 10
        self.save_dir = self.Base_dir + 'checkpoints/'
        