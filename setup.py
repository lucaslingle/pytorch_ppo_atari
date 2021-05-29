from setuptools import setup


setup(
    name="pytorch_ppo_atari",
    py_modules=["ppo"],
    version="1.0",
    description="A Pytorch implementation of Proximal Policy Optimization for Atari 2600 games.",
    author="Lucas D. Lingle",
    install_requires=[
        'atari-py==0.2.6',
        'gym==0.18.0',
        'matplotlib==3.4.2',
        'moviepy==1.0.3',
        'mpi4py==3.0.3',
        'opencv-python==4.5.2.52',
        'torch==1.8.1'
    ]
)
