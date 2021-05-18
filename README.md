# BiNN-SDE

Implementation of the paper Learning stochastic dynamical systems with neural networks mimicking the Euler-Maruyama scheme.

Associated paper:

License: CECILL-C license Copyright IMT Atlantique/OceaniX,
contributor(s) : Noura Dridi, Lucas Drumetz, Ronan Fablet 18/05/2021

Contact person: nourradridi@gmail.com

This software is a computer program aims to learn the parameters of a Stochastic Differential Equation (SDE) https://en.wikipedia.org/wiki/Stochastic_differential_equation used to model stochastic dynamical sytem. The parameters of the SDE are represented by a neural network with a built-in SDE integration scheme using on Euler Maruyama method https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method .
The software is availble for one dimension example Geometric Brownian Motion https://en.wikipedia.org/wiki/Geometric_Brownian_motion , as well as three dimensional one called Stochastic Lorenz. The latter is a stochastic version of the Lorenz-63 system well known to represent ocean-atmosphere interactions derived from the Navier-Stokes equations https://hal.inria.fr/hal-01629898. 


Citation :
@article{Dridi2021,
  title={Learning stochastic dynamical systems with neural networks mimicking the Euler-Maruyama scheme},
  author={Noura, Dridi and Lucas, Drumetz and Ronan Fablet},
  journal={European Signal Processing Conference, EUSIPCO},
  year={2021}
}


