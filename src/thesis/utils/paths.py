from pathlib import Path, PurePosixPath, PureWindowsPath
import json
from urllib.parse import urlencode


def win_to_linux(x: str) -> str:
    return str(PurePosixPath(PureWindowsPath(x)))


def linux_to_win(x: str) -> str:
    return str(PurePosixPath(PureWindowsPath(x)))


def to_image_element(x: Path, parent: Path) -> str:
    """Convert path to HTML img element

    Example:

        win_path = "TowardsUniversal\\progan_train\\person\\1_fake\\03610.png"
        img      = convert_to_html(win_path)
        assert   x == "TowardsUniversal/progan_train/person/1_fake/03610.png"   # Hypothetical
        assert img == '{"img_src": <x>, "alt": <x>, "href": <x>, "width": 200}' # x = as above

    """
    url = urlencode(str(x.relative_to(parent)))
    data = {"img_src": url, "alt": url, "href": url, "width": 200}
    return json.dumps(data)
