import logging

logging.basicConfig(
    level=logging.INFO,
    style="{",
    format="{asctime}: {levelname[0]} {pathname}:{lineno}] {message}",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
)
