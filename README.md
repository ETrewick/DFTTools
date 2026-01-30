# DFTTools
DFT Tools is a collection of code for reading and writing inputs and outputs for Quantum ESPRESSO. Hundreds before me have written code to do this, and surely hundreds more will in the future. I publish this code here in the hope that _some of_ this code will be useful to in the future.



## Occupation matrix tools
Code for manipulating the density matrices (also known as occupation matrices) in Quantum ESPRESSO and Quanty. With the aim of allowing the starting density matrix in QE to be set based on a Quanty output.
The an example calculation for the starting density matrix found by Quanty is found at <https://gist.github.com/ETrewick/2553eff5ca63b17a55373e760d4560df> 
To input this matrix into QE, it is recommended that gist <https://gist.github.com/ETrewick/0b4b484a2e680e94b11d3fe4ce74d27d> be applied to QE7.5, allowing a density matrix is to be read from `outdir/prefix.save/occup.txt` after the first step of an scf cycle, if any of the `starting_ns_eigenvalue` input flags is set to the "special number" 67. This functionality overrides and replaces the normal `starting_ns_eigenvalue` behaviour when this "special number" is set in the input.
An additional patch, <https://gist.github.com/ETrewick/bd995760a8e44b2617e8639f092a3a43> allows QE to print the full complex density matrix in the non-collinear case, allowing the progress of convergence to be more easily monitored by plotting the density matrix as the calculation progresses, ensuring that the desired GS is reached.

This code was written the 4f shell in mind, but should operate correctly on any single shell manifold (S,P,D,F,G?!).

Please see the main function of this code for examples of it's use.



## plotBands
Functions for reading and plotting (mostly 2spin) bandstructures and density of states from QE's standard tools


## plotUnfold
Functions for reading and plotting the ***projected and unfolded*** bandstructures from supercell calculations made with `unfold.x`. This is somewhat difficult as the data must be plotted on a 2d image due to the requirement that bands which overlap should have weights which add, but add independently for the different colour weight projections, which usually has far more axes that the three commonly devoted to colour.


## pw2xsf
This script supposedly converts QE input or output files into a format that can be read by XCrySDen. At this date (Jan 2026) I have no recollection of writing it.


## multicoloredline
The implementation of the multicolored lines given in the maplotlib docs
