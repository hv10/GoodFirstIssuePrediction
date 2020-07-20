import logging

logging.basicConfig(
    level=logging.DEBUG,
    style="{",
    format="{asctime}: {levelname[0]} {pathname}:{lineno}] {message}",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
)
