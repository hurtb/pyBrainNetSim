from distutils.core import setup

setup(
    name='pyBrainNetSim',
    version='0.1',
    packages=['examples', 'pyBrainNetSim', 'pyBrainNetSim.test', 'pyBrainNetSim.models',
              'pyBrainNetSim.drawing', 'pyBrainNetSim.solvers', 'pyBrainNetSim.generators',
              'pyBrainNetSim.generators.settings', 'pyBrainNetSim.simulation'],
    url='https://github.com/hurtb/pyBrainNetSim',
    license='',
    author='brian hurt',
    author_email='BrianJ.Hurt@gmail.com',
    description='Simulate biologically-inspired neural networks under evolutionary forces.'
)
