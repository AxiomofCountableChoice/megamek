import logging
import os
import sys

def setup_logger(name, log_file="cluster.log"):
    """
    Sets up a logger that outputs to both the console and a shared file.
    The format includes timestamps and the component name.
    """
    # Ensure logs directory exists
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, log_file)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Avoid adding handlers multiple times if instantiated multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Stream Handler (stdout)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO) # Console gets INFO and above
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        
        # File Handler
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG) # File gets everything including DEBUG
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
