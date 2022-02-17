from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.MD'), encoding='utf-8') as file:
    long_description = file.read()

setup(
    name='simple_encoders',
    version='0.1.5',
    description='Simple encoders to pre-process categoric variables for machine learning systems',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Fabian Margreiter',
    author_email='fabian.margreiter@gmail.com',
    packages=['simple_encoders'],
    package_dir={'simple_encoders': 'src'},
    url='https://github.com/EmmArrGee/simple-encoders',
    keywords='python data science pandas machine learning category encoding',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent'
    ],
    extras_require={
        'dataframes': 'pandas>=1.4.1'
    },
    test_suite="tests"
)
