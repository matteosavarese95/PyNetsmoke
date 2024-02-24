from setuptools import setup
setup(name='PyNetsmoke',
version='0.1',
description='Python-Netsmoke interface',
url='#',
author='matteosavarese',
author_email='matteo.savarese@ulb.be',
license='MIT',
packages=['PyNetsmoke'],
package_dir={'PyNetsmoke': 'src/PyNetsmoke'},
install_requires=[
        # List your dependencies here
        'numpy',
        'matplotlib',
        'cantera',
        'pandas',
        'os',
        'time'
    ],
zip_safe=False)