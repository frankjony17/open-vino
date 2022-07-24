import io
import re

from setuptools import find_packages, setup

with io.open('./src/__init__.py', encoding='utf8') as v:
    ver = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", v.read(), re.M)
    if ver:
        version = ver.group(1)
    else:
        raise RuntimeError("Unable to find version string.")

with io.open('README.md', encoding='utf8') as readme:
    description = readme.read()

dev_requirements = ['pycodestyle', 'isort', 'bandit', 'flake8']
unit_test_requirements = ['pytest']
prod_requirements = [
    'opencv-python',
    'fastapi',
    'uvicorn',
    'gunicorn',
    'jinja2',
    'starlette',
    'aiofiles',
    'numpy',
    'openvino'
]


setup(
    name='openvino',
    version=version,
    author='Frank Ricardo. R',
    packages=find_packages(exclude='tests'),
    python_requires='>=3.6',
    long_description=description,
    install_requires=prod_requirements,
    extras_require={
         'dev': dev_requirements,
         'unit': unit_test_requirements
    },
    entry_points={
        'console_scripts': [
            'openvino = src.__main__:start'
        ],
    },
    description='TODO.',
    include_package_data=True,
    url='',
    license='COPYRIGHT',
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Information Technology',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6'
    ],
    keywords=('API', 'Company', 'OpenVino')
)
