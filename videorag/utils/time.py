"""Time-format related utility helpers."""


def fmt_time(sec: float) -> str:
    """Format seconds as MM:SS.ss."""
    m, s = divmod(float(sec), 60.0)
    return f"{int(m):02d}:{s:05.2f}"
