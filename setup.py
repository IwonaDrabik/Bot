from setuptools import setup

# reading long description from file
with open('Description.txt') as file:
    long_description = file.read()

with open('requirements.txt') as f:
    requirements = f.readlines()
#
#
# # specify requirements of your package here
# REQUIREMENTS = ['requests']

# some more details
CLASSIFIERS = [
     'Programming Language :: Python :: 3.8',
    ]

# calling the setup function
setup(name='pythonProject',
      version='1.0',
      description='A first-contact virtual assistant answering club/fitness center related questions',
      url='https://github.com/IwonaDrabik/Bot',
      author='Iwona Drabik',
      author_email='drabik.iwona@gmail.com',
      license='',
      packages=['charlie'],
      classifiers=CLASSIFIERS,
      install_requires=requirements,
      keywords='NLP, torch, neural network, tkinter'
      )
