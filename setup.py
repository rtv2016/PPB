from setuptools import setup

setup(name='chem',
      version='0.0',
      description='Data driven QSAR modeling',
      author='Brandon Veber',
      author_email='veber001@umn.edu',
      license='mit?',
      packages=['chem'],
      install_requires=[
          'numpy',
          'pandas',
          'scikit-learn',
      ]
      )
