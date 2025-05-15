import logging

class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        msg = super().format(record)
        return f"{color}{msg}{self.RESET}"

def get_color_logger(name: str = "ColorLogger", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = ColorFormatter("%(asctime)s [%(levelname)s]\n%(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
