"""Package setup script."""
import setuptools

setuptools.setup(
    name="mbbl",
    version="0.1",
    description="Model-based RL Baselines",
    author="Tingwu Wang",
    author_email="tingwuwang@cs.toronto.edu",
    packages=setuptools.find_packages(),
    package_data={'': ['./env/dm_env/assets/*.xml',
                       './env/dm_env/assets/common/*.xml',
                       './env/gym_env/fix_swimmer/assets/*.xml',
                       './env/gym_env/pets_env/assets/*.xml']},
    include_package_data=True,
    install_requires=[
    ],
)

"""
        "pyquaternion",
        "beautifulsoup4",
        "Box2D>=2.3.2",
        "num2words",
        "six",
        "tensorboard_logger",
        "tensorflow==1.12.0",
        "termcolor",
        "gym[mujoco]==0.7.4",
        "mujoco-py==0.5.7",
"""
