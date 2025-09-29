from setuptools import setup, find_packages

setup(
    name='test_earthquake_resilience_1',
    version='0.0.7',
    author='Bence Popovics',
    author_email='popbence@gmail.com',
    description='Earthquake resilience calculation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/popbence/test_earthquake_resilience/',
    packages=find_packages(),
    py_modules=['test_earthquake_resilience'],
    install_requires=[
        'numpy',
        'scikit-learn',
        'scipy',
        'pandas',
        'matplotlib',
        'reverse_geocoder',
        'folium'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
    ],
)