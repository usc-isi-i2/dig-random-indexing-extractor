try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

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
    'install_requires':['digExtractor>=0.1.7']
}

setup(**config)
