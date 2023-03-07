import os,sys, socket
from logging import (DEBUG, INFO, WARNING, FileHandler, Formatter,
                     StreamHandler, getLogger, basicConfig)


class LogHandler:
    """
    A module to handle the logging of the projeckt
    """

    def __init__(self, name,basepath, verbosity='info'):
        '''
            Initialisation. 

            Args: 
                name: the overall run's name (will create a folder on force_work with this name)
                verbosity: the message level of logging that will lead to print outs on your console - can be one of the following: ['info', 'debug']
        '''
        self.name = name
        self.basepath  = basepath

        self.logger = None
        self.logger = getLogger(self.name)
        self.logger.setLevel(DEBUG)
        self.logger.propagate = False

        if not self.logger.handlers:
            ch_level = {'info': INFO, 'debug': DEBUG}[str.lower(verbosity)]
            fh = FileHandler(os.path.join(self.basepath, 'log.log'))
            fh.setLevel(ch_level)
            ch = StreamHandler(sys.stdout)
            ch.setLevel(ch_level)
            
            che = StreamHandler(sys.stderr)
            che.setLevel(WARNING)
            pc = '{}{}'.format(socket.gethostname().replace('.lfb.rwth-aachen.de', ''), ' {}'.format(os.environ['CUDA_VISIBLE_DEVICES']) if 'CUDA_VISIBLE_DEVICES' in os.environ else '')
            formatter = Formatter(
                '[%(asctime)s {} {}] %(message)s'.format(pc, self.name))
            formatter.datefmt = '%y/%m/%d %H:%M:%S'
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            che.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
            self.logger.addHandler(che)

        self.log('info', 'Started LogHandler in Folder %s ' % (self.basepath))

    def log(self, logtype, msg):
        '''
            Logs messages. 
            
            Args: 
                logtype: the type of logging message - can be one of the following: ['debug', 'info', 'warning', 'error']
        '''
        if self.logger is None:
            return

        if logtype == 'debug':
            self.logger.debug(msg)
        elif logtype == 'info':
            self.logger.info(msg)
        elif logtype == 'warning':
            self.logger.warning(msg)
        elif logtype == 'error':
            self.logger.error(msg)

