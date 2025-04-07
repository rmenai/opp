# [Operation Pele Password](./README.md) &middot; [![GitHub license]](./LICENSE) ![Test Action]

Have you ever wanted to know what someone is typing just by listening to their
keyboard? Well, now you can! You can now eavesdrop on somebody's typing just
from the recording of their keys.

<!-- Table of Contents -->

- [Usage](#usage)
- [Features](#features)
- [License](#license)

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



<!-- Shields.io links -->

[gitHub license]: https://img.shields.io/badge/license-MIT-blue.svg
[test action]: https://github.com/rmenai/operation-pele-password/actions/workflows/test.yaml/badge.svg
