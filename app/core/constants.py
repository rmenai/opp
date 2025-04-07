from pydantic import BaseModel


class Keys(BaseModel):
    """Keyboard keys."""

    A: str = 'a'
    B: str = 'b'
    C: str = 'c'
    D: str = 'd'
    E: str = 'e'
    F: str = 'f'
    G: str = 'g'
    H: str = 'h'
    I: str = 'i'
    J: str = 'j'
    K: str = 'k'
    L: str = 'l'
    M: str = 'm'
    N: str = 'n'
    O: str = 'o'
    P: str = 'p'
    Q: str = 'q'
    R: str = 'r'
    S: str = 's'
    T: str = 't'
    U: str = 'u'
    V: str = 'v'
    W: str = 'w'
    X: str = 'x'
    Y: str = 'y'
    Z: str = 'z'

    ZERO: str = '0'
    ONE: str = '1'
    TWO: str = '2'
    THREE: str = '3'
    FOUR: str = '4'
    FIVE: str = '5'
    SIX: str = '6'
    SEVEN: str = '7'
    EIGHT: str = '8'
    NINE: str = '9'

    TAB: str = 'TAB'
    CAPS: str = 'CAPS'
    LSHIFT: str = 'LSHIFT'
    RSHIFT: str = 'RSHIFT'
    LCTRL: str = 'LCTRL'
    RCTRL: str = 'RCTRL'
    ENTER: str = 'ENTER'
    BACKSPACE: str = 'BACKSPACE'


class Constants(BaseModel):
    """The app constants."""

    SAMPLE_RATE = 44100
    CHANNELS = 2

    keys: Keys = Keys()


constants = Constants()
