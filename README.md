## Occupation matrix tools
Code for manipulating the density matricies (also known as occupation matricies) in Quantum ESPRESSO and Quanty. With the aim of allowing the starting density matrix in QE to be set based on a Quanty output.
For this to work, it is recommended that gist <https://gist.github.com/ETrewick/0b4b484a2e680e94b11d3fe4ce74d27d> be applied to QE7.5, allowing a density matrix is to be read from outdir/prefix.save/occup.txt after the first step of an scf cycle, if any of the `starting_ns_eigenvalue` input flags is set to 67.
This functionality overides and replaces the normal `starting_ns_eigenvalue` behaviour
An additional patch, <https://gist.github.com/ETrewick/bd995760a8e44b2617e8639f092a3a43> allows QE to print the full complex density matrix in the non-collinear case, allowing the progress of convergence to be more easily monitored by plotting the density matrix as the calculation progresses, ensuring that the desired GS is reached.
In it's current state, the code currently only operates on the f shell, but the functions could be extended to the the d shell without much effort.

Please see the main function of this code for an example of it's use
