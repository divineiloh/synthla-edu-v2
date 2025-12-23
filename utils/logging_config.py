"""
Logging configuration for SYNTHLA-EDU V2.

Provides structured logging setup that can replace print statements
throughout the codebase. Supports console and file output with
configurable verbosity levels.

Usage:
    from utils.logging_config import setup_logger
    
    logger = setup_logger("synthla_edu", log_file="runs/experiment.log")
    
    logger.info("Starting experiment...")
    logger.debug("Detailed debug information")
    logger.warning("Warning message")
    logger.error("Error occurred")

    # Instead of:
    # print(f"Training {synth_name}...")
    
    # Use:
    # logger.info(f"Training {synth_name}...")
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color-coded log levels for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logger(name: str = "synthla_edu",
                log_file: Optional[str] = None,
                level: str = "INFO",
                console: bool = True,
                file_level: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with console and/or file handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (None to disable file logging)
        level: Console logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Enable console logging
        file_level: File logging level (uses 'level' if None)
    
    Returns:
        Configured logger instance
    
    Example:
        # Console only (INFO level)
        logger = setup_logger("experiment1")
        
        # Console + file (DEBUG to file, INFO to console)
        logger = setup_logger("experiment2", 
                             log_file="runs/exp2.log",
                             level="INFO",
                             file_level="DEBUG")
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture everything, filter at handler level
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        # Use colored formatter for console
        console_format = ColoredFormatter(
            fmt='%(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create log directory if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(getattr(logging, (file_level or level).upper()))
        
        # Use detailed format for file (no colors)
        file_format = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ProgressLogger:
    """
    Progress logging utility for long-running operations.
    
    Provides percentage-based progress updates without overwhelming the log.
    
    Example:
        logger = setup_logger("training")
        progress = ProgressLogger(logger, total_steps=300, prefix="Epoch")
        
        for epoch in range(300):
            train_loss = train_epoch()
            progress.update(1, extra_msg=f"Loss: {train_loss:.4f}")
        
        progress.finish()
    """
    
    def __init__(self, logger: logging.Logger, total_steps: int,
                prefix: str = "Progress", log_every_pct: float = 10.0):
        """
        Initialize progress logger.
        
        Args:
            logger: Logger instance
            total_steps: Total number of steps
            prefix: Prefix for log messages
            log_every_pct: Log every N percent (default: 10%)
        """
        self.logger = logger
        self.total_steps = total_steps
        self.prefix = prefix
        self.log_every_pct = log_every_pct
        
        self.current_step = 0
        self.last_logged_pct = -1
        self.start_time = datetime.now()
    
    def update(self, n: int = 1, extra_msg: str = ""):
        """
        Update progress by n steps.
        
        Args:
            n: Number of steps completed
            extra_msg: Additional message to append
        """
        self.current_step += n
        current_pct = (self.current_step / self.total_steps) * 100
        
        # Check if we should log
        pct_bucket = int(current_pct / self.log_every_pct) * self.log_every_pct
        
        if pct_bucket > self.last_logged_pct and pct_bucket <= 100:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            
            msg = f"{self.prefix}: {current_pct:.1f}% ({self.current_step}/{self.total_steps})"
            if extra_msg:
                msg += f" | {extra_msg}"
            msg += f" | Elapsed: {elapsed:.1f}s"
            
            self.logger.info(msg)
            self.last_logged_pct = pct_bucket
    
    def finish(self, final_msg: str = "Complete"):
        """Mark progress as finished."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(f"{self.prefix}: {final_msg} | Total time: {elapsed:.1f}s")


class LogContext:
    """
    Context manager for hierarchical logging sections.
    
    Example:
        logger = setup_logger("experiment")
        
        with LogContext(logger, "Data Loading"):
            # Log messages here are indented
            logger.info("Loading OULAD...")
            logger.info("Loading ASSISTments...")
        
        with LogContext(logger, "Model Training"):
            logger.info("Training CTGAN...")
    """
    
    def __init__(self, logger: logging.Logger, section_name: str):
        """
        Initialize log context.
        
        Args:
            logger: Logger instance
            section_name: Name of the section
        """
        self.logger = logger
        self.section_name = section_name
        self.start_time = None
    
    def __enter__(self):
        """Enter context."""
        self.start_time = datetime.now()
        self.logger.info(f"{'='*60}")
        self.logger.info(f"{self.section_name}")
        self.logger.info(f"{'='*60}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(f"{'-'*60}")
        self.logger.info(f"{self.section_name} complete | Time: {elapsed:.1f}s")
        self.logger.info("")


# Example usage
if __name__ == "__main__":
    # Demonstration of logging utilities
    print("Logging Configuration Demonstration\n")
    
    # Setup logger
    logger = setup_logger(
        name="demo",
        log_file="demo.log",
        level="INFO",
        file_level="DEBUG"
    )
    
    # Basic logging
    logger.debug("This is a debug message (only in file)")
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    # Progress logging
    print("\nProgress Logging Demo:")
    with LogContext(logger, "Training Model"):
        progress = ProgressLogger(logger, total_steps=100, prefix="Epoch")
        
        import time
        for i in range(100):
            time.sleep(0.01)  # Simulate work
            progress.update(1, extra_msg=f"Loss: {0.5 - i*0.004:.4f}")
        
        progress.finish()
    
    print("\nLog file written to: demo.log")
