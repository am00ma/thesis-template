from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")


def convert_to_linux(x: str) -> str:
    """Convert windows path to linux, and get path relative to "data"

    Example:

        win_path = "c:\\Users\\stijn\\msc\\deepfake_detection\\data\\TowardsUniversal\\progan_train\\person\\1_fake\\03610.png"
        lnx_path = convert_to_linux(win_path)
        assert lnx_path == "TowardsUniversal/progan_train/person/1_fake/03610.png"

    """
    out = x.replace("\\", "/")
    out = out.split("data")[1][1:]
    return out


def convert_to_windows(x: str) -> str:
    """Get windows path relative to "data"

    Example:

        win_path = "c:\\Users\\stijn\\msc\\deepfake_detection\\data\\TowardsUniversal\\progan_train\\person\\1_fake\\03610.png"
        rel_path = convert_to_windows(win_path)
        assert rel_path == "TowardsUniversal\\progan_train\\person\\1_fake\\03610.png"

    """
    out = x.split("data")[1][1:]
    return out


def convert_to_html(x: str) -> str:
    """Convert path to HTML img element

    Example:

        win_path = "TowardsUniversal\\progan_train\\person\\1_fake\\03610.png"
        img      = convert_to_html(win_path)
        assert   x == "TowardsUniversal/progan_train/person/1_fake/03610.png"   # Hypothetical
        assert img == '{"img_src": <x>, "alt": <x>, "href": <x>, "width": 200}' # x = as above

    """
    x = x.replace("\\", "/")
    x = "/" + x
    data = {"img_src": x, "alt": x, "href": x, "width": 200}
    return json.dumps(data)
