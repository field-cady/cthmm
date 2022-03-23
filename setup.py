'''
This library was developed at Zeitworks Inc, written by Field Cady.
'''

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='cthmm',
    version='0.0.1',
    author='Field Cady',
    author_email='field@zeitworks.com',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/field-cady/cthmm',
    #project_urls = {
    #    "Bug Tracker": "https://github.com/mike-huls/toolbox/issues"
    #},
    license='MIT',
    packages=['cthmm'],
    install_requires=['numpy', 'scipy'],
)