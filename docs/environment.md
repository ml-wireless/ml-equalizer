# Environment

## Notes

Since `pytorch` does not support `python` 2.7, so `python` 3 is used.

For now, only cpu version of `pytorch` is used.

`python` version 3.7.4 and `pip` version 19.3.1 is used for now, but it seems like any `python` version >= 3.5 can work.

(by Tiny) I'm using `pip` to install all dependencies and `virtualenv` for virtual environment for now. `conda` can be used alternatively (but not included in this doc).

## All Dependencies Now

- pytorch=1.3.0+cpu
- numpy=1.17.3
- matplotlib=3.1.1

## Set up a python Environment

There are multiple ways to install `python` on either Windows or other systems.

For Windows:

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Anaconda](https://www.anaconda.com/)
- [Officially](https://www.python.org/downloads/windows/)

If you have both `python` 2 and 3 installed, you will have to use commands like `python3` and `pip3` to use version 3. Set up a virtual environment using either `conda` (if you install `python` with `conda`) or `virtualenv` (if you install only `python`) can avoid it.

Docs:

- [virtualenv](https://virtualenv.pypa.io/en/latest/)
- [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)

If you are using `virtualenv`, please put the virtual environment folder in `venv/` as it is ignored by `git`.

Finally make sure your `python` and `pip` version:

    $ python --version
    $ pip --version

## pip Dependencies

Install `pytorch` first:

    pip install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

Then install the rest:

    pip install -r requirements.txt
