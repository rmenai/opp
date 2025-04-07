# [Operation Pele Password](./README.md) &middot; [![GitHub license]](./LICENSE)

Have you ever wanted to know what someone is typing just by listening to their
keyboard? Well, now you can! You can now eavesdrop on somebody's typing just
from the recording of their keys.

<!-- Table of Contents -->

- [Usage](#usage)
- [Features](#features)
- [License](#license)
- [Contact](#contact)

## Usage

### Machine Learning Approach

In this approach, we train a neural network by giving it many audio samples of
the same key. What you need to do is to run the ... file and follow the
instructions. The model will adapt to your keyboard model and hopefully give you
acceptable results

### Naive Approach

We first make an algorithm that will split a keyboard sample into well-defined
chunks, normalising the sound and aligning it at the center of the audio sample.
This approach should work for either provided files or live recorded audio.

## Features

* Live audio recording
* Automatic audio splitting
* Key classification


## License

Distributed under the MIT License. See [LICENSE](./LICENSE) for more information.


<!-- Packages Links -->

[.env]: https://pypi.org/project/python-dotenv/
[colorlog]: https://pypi.org/project/colorlog/
[docker]: https://www.docker.com/
[dotenv]: https://pypi.org/project/python-dotenv/
[flake8]: https://flake8.pycqa.org/en/latest/
[poetry]: https://python-poetry.org/
[pre-commit]: https://pre-commit.com/
[pydantic]: https://pydantic-docs.helpmanual.io/
[pytest]: https://docs.pytest.org/en/6.2.x/

<!-- Repository links -->

[community standards]: https://github.com/boilercodes/python/community

<!-- Shields.io links -->

[gitHub license]: https://img.shields.io/badge/license-MIT-blue.svg
