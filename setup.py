from setuptools import setup

setup(name='predictivecontrol',
      version='0.1',
      description='Python package which implements Predictive Control techniques (e.g., MPC, E-MPC)',
      url='https://github.com/rgmaidana/predictiveControl',
      author='Renan Maidana',
      author_email='renan.maidana@acad.pucrs.br',
      license='MIT',
      packages=['predictivecontrol'],
      install_requires=[
          'numpy',
          'scipy'
      ],
      zip_safe=False
    )