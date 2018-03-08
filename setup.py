from distutils.core import setup

setup(
    name = 'behavior2text',
    packages=['behavior2text'],
    package_dir={'behavior2text':'behavior2text'},
    package_data={'behavior2text':['inputData/*', 'labelData/*', 'management/commands/*', 'utils/*']},
    version = '0.8',
    description = 'A django App for behavior2text',
    author = ['davidtnfsh'],
    author_email = 'davidtnfsh@gmail.com',
    url = 'https://github.com/udicatnchu/behavior2text',
    download_url = 'https://github.com/udicatnchu/behavior2text/archive/v0.8.tar.gz',
    keywords = ['behavior2text', 'context Text', 'context log', 'user intension', 'keyword extraction'],
    classifiers = [],
    license='GPL3.0',
    install_requires=[
        'numpy',
        'scipy',
        'requests',
        'pyprind',
        'udicOpenData',
        'matplotlib'
    ],
    zip_safe=True,
)
