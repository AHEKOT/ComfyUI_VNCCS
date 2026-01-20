"""
Stub for RankedLogger - only needed for training, not inference
"""
import logging


class RankedLogger:
    """Minimal stub for RankedLogger - used in training but not needed for inference"""
    
    def __init__(self, name=__name__, rank_zero_only=True, extra=None):
        self.logger = logging.getLogger(name)
        self.rank_zero_only = rank_zero_only
    
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
