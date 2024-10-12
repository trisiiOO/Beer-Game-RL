from setuptools import setup, find_packages

setup(
    name='supply_chain_simulation',
    version='0.1.0',
    author='Tristan Kruse',
    description='A Q-learning-based supply chain simulation tool for decision-making in inventory management',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/trisiiOO/Beer-Game-RL',
    packages=find_packages(),
    py_modules=['main'],
    install_requires=[
        'pandas>=1.0.0',
        'numpy>=1.19.0',
        'pytest>=6.0.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'supply_chain_sim=main:main',
        ],
    },
)
