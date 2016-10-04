try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from setuptools import setup, find_packages
from setuptools.command.install import install as _install
from subprocess import call

class Install(_install):
    def run(self):
        _install.run(self)
        import nltk
        nltk.download("punkt")

config = {
    'name': 'digRandomIndexingExtractor',
    'description': 'digRandomIndexingExtractor',
    'author': 'Jason Slepicka',
    'url': 'https://github.com/usc-isi-i2/dig-random-indexing-extractor',
    'download_url': 'https://github.com/usc-isi-i2/dig-random-indexing-extractor',
    'author_email': 'jasonslepicka@gmail.com',
    'version': '0.1',
    # these are the subdirs of the current directory that we care about
    'packages': ['digRandomIndexingExtractor'],
    'scripts': [],
    'install_requires':['scipy>=0.9', 'numpy', 'digExtractor>=0.1.8', 'scikit-learn==0.17','nltk'],
    'cmdclass':{'install': Install}
}

setup(**config)
