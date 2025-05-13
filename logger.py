from loguru import logger

logger.add(
    "logs/pipeline_{time:YYYY-MM-DD}.log", 
    rotation="10 MB", 
    retention="7 days", 
    compression="zip", 
    level="INFO",
    enqueue=True,
    backtrace=True,
    diagnose=True
)

__all__ = ["logger"]
