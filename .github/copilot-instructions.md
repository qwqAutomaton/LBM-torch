# Copilot Instructions

## 1. General Principles

This project is a scientific computing framework for lattice Boltzmann methods (LBM), initially implemented in Python with PyTorch and optional GPU acceleration.

Priorities, in order:

1. Mathematical and physical correctness.
2. Reproducibility and independently verifiable results.
3. Clear, maintainable implementation.
4. Efficient execution and memory usage.
5. Extensibility for explicitly planned models and features.
6. Convenience and abstraction.

Do not sacrifice correctness or clarity for premature optimization or speculative extensibility.

The developer understands C/C++, algorithms, and data structures. Communicate technical decisions precisely. Do not explain elementary programming concepts unless requested.

## 2. Development Workflow

### Before implementation

For non-trivial tasks:

1. Inspect relevant source files and documentation.
2. Identify the requirements, existing conventions, and affected interfaces.
3. Check for contradictions or unresolved design decisions.
4. Propose a concise implementation plan before editing files.
5. State important assumptions and how correctness will be verified.

Do not immediately generate a complete implementation from a broad requirement.

If a significant mathematical, physical, or architectural decision remains unresolved, ask the developer instead of silently choosing a solution.

For small, unambiguous tasks, proceed directly without unnecessary planning overhead.

### During implementation

- Implement one coherent, testable increment at a time.
- Follow the approved design and existing project conventions.
- Avoid unrelated refactoring or speculative features.
- Do not silently change public interfaces, mathematical conventions, data layouts, or file formats.
- If implementation reveals a material flaw in the design, stop and explain it before proceeding.
- Prefer simple implementations whose behavior can be reasoned about directly.

### After implementation

Report concisely:

- What changed and why.
- Important implementation decisions.
- Tests executed and their results.
- Known limitations and unverified assumptions.
- Any deviation from the agreed design.

Never claim a test passed unless it was actually executed and passed.

A successful import, compilation, or test run does not by itself establish numerical or physical correctness.

After that, modify the documentation to reflect the new behavior, including any changes to public interfaces, mathematical conventions, or file formats.

## 3. Architecture and Extensibility

### Prefer explicit specialization over excessive abstraction

The initial target is D2Q9. D3Q19 and D3Q27 may be supported in the future.

- Implement D2Q9 as the first complete, tested model.
- Keep D3Q19 and D3Q27 recognizable by the model factory, but do not implement their numerical solvers until explicitly requested.
- Unsupported models must fail clearly, for example with `NotImplementedError`.
- Do not create generic frameworks merely because additional models might eventually exist.
- A small amount of duplicated code is acceptable when it makes specialized implementations simpler and easier to verify.

Use a factory when it simplifies model selection and construction. Keep model-specific behavior explicit.

### Keep extension points narrow and intentional

Potential future extensions include:

- Slip boundary conditions.
- Variable viscosity.
- High-Knudsen-number models and related physical corrections.
- Additional lattice models.
- Additional output formats.
- GPU compilation and performance optimizations.

Preserve reasonable extension points without implementing these features prematurely.

Prefer stable interfaces and explicit contracts over elaborate inheritance hierarchies, plugin systems, or highly generic configuration frameworks.

Do not introduce an abstraction unless it solves a current problem or protects a clearly identified future interface.

## 4. Numerical and Physical Correctness

Treat the numerical method as a mathematical specification, not merely an implementation detail.

- Preserve explicitly documented lattice conventions, direction numbering, weights, sound speed, and coordinate ordering.
- Use lattice units unless the project documentation explicitly specifies otherwise.
- For the standard D2Q9 lattice, use the documented value `cs^2 = 1/3`.
- Keep collision, streaming, boundary handling, macroscopic reconstruction, forcing, and diagnostics logically distinguishable.
- Ensure that the implementation of forcing and the reconstruction of macroscopic velocity follow the same discrete formulation.
- Check parameter validity and document the relevant mathematical constraints.
- Do not silently change equations, sign conventions, relaxation parameters, or boundary semantics.

For BGK, TRT, and MRT, document the applicable relaxation parameters and their relationship to viscosity according to the chosen formulation.

When mathematical conventions are ambiguous, identify the ambiguity and ask for clarification.

## 5. Data Layout and State Management

Follow the project's documented tensor layout. Do not change tensor axis ordering without explicit approval.

The initial design uses double-buffered distribution functions with a current-buffer index rather than copying one complete PDF tensor into another after every step.

- Preserve the double-buffering contract.
- Update the current-buffer index consistently.
- Avoid unnecessary full-tensor copies.
- Make buffer ownership and read/write semantics explicit.
- Ensure that diagnostics and output access the current physical state, not a stale buffer.

For solid masks, use a boolean tensor with the documented spatial dimensions and semantics.

Handle fluid and solid nodes consistently across collision, streaming, diagnostics, and output.

Do not introduce implicit layout conversions or hidden device transfers into performance-critical paths.

## 6. Collision, Streaming, and Boundaries

### Collision

The initial complete implementation is BGK.

- Accept physical or lattice parameters at the solver construction boundary as specified by the project API.
- Derive relaxation parameters in one clearly defined place.
- Validate parameter ranges according to the selected numerical formulation.
- Keep the collision interface suitable for future TRT/MRT implementations without prematurely implementing them.

### Streaming

Use pull streaming as specified by the project design.

- Handle each lattice direction's velocity shift correctly.
- Respect periodic boundaries and solid-node treatment.
- Avoid `torch.roll` when it introduces unnecessary full-tensor operations or unwanted wraparound behavior.
- Prefer explicit indexing or slicing when it makes the operation and its boundary semantics clearer.
- Do not replace a correct implementation with a more complicated optimization without evidence of benefit.

### Boundary conditions

Implement only the boundary conditions explicitly required by the current version.

The initial version uses periodic boundaries and half-way bounce-back for solid walls as specified in the numerical design.

Future Zou/He inlet/outlet conditions and slip boundary conditions must not be implemented implicitly or mixed into the initial solver.

Keep boundary behavior explicit and testable.

## 7. Forcing and Macroscopic Quantities

Guo forcing may be supported, but forcing must be disabled by default unless explicitly configured.

Use a consistent discrete forcing formulation.

For the documented convention, reconstruct velocity as:

`u = (sum_i(f_i * c_i) + F/2) / rho`

This expression assumes the project's force-density and lattice-time convention. If the formulation changes, update the mathematical documentation, collision implementation, and tests together.

Do not independently modify velocity reconstruction or forcing terms without checking their consistency.

Compute macroscopic quantities only on fluid nodes where required by the project contract.

Handle invalid density, NaN, and other non-finite values explicitly. Do not silently replace invalid numerical states with plausible-looking results.

## 8. Precision and Device Support

Separate computation precision from output precision.

- The initial computation default is `float32`.
- The initial output default may be `float16` only as explicitly specified by the output contract.
- Make precision conversion explicit.
- Never silently change the internal computation dtype to match the output dtype.
- Document that reduced-precision output is lossy.

Keep device selection explicit and compatible with the intended PyTorch backends, including Apple MPS and NVIDIA CUDA.

Do not assume CUDA, Triton, or `triton-windows` is available on every machine.

### Compilation

Compilation is optional.

- Eager execution must remain the baseline path.
- Compile the single-step kernel or function, not a multi-step loop whose iteration count may vary.
- If compilation is requested and fails, propagate a clear error according to the project contract.
- Do not silently fall back to eager execution when the configured behavior requires compilation to fail explicitly.
- Keep optional compilation dependencies out of the mandatory path where practical.

Do not add backend-specific workarounds without understanding their correctness and portability implications.

## 9. Time Stepping, Diagnostics, and Output

The solver owns its simulation time step counter.

The public stepping interface may advance a caller-specified number of steps. Do not assume that every invocation advances the same number of steps.

Keep simulation stepping independent from sampling and output scheduling.

Diagnostics should initially provide:

- Total mass over fluid nodes.
- Maximum fluid velocity magnitude.
- Minimum and maximum fluid density.
- Explicit NaN/non-finite detection according to the documented contract.

Do not introduce an extensive monitoring framework unless requested.

### Output

The initial supported format is NPZ.

- Save the current distribution functions (PDFs), not only macroscopic quantities.
- Include the simulation step in output naming according to the project contract.
- Write run metadata separately in JSON.
- Save the solid mask as a separate NPZ artifact.
- Keep future HDF5 and VTK support as explicit extension points or stubs until requested.
- Avoid unnecessary data copies and transfers when saving state.
- Clearly distinguish stored precision from internal precision.

Output must be sufficient to identify the simulation configuration and interpret the saved state.

## 10. Testing and Validation

Tests must establish numerical properties, not merely exercise code paths.

Use independent validation whenever practical:

- Analytical solutions.
- Conservation laws.
- Equilibrium and limiting cases.
- Known symmetries.
- Reference implementations.
- Grid-convergence studies.
- Differential tests across supported devices or execution modes.

For Poiseuille flow:

- Use the documented geometry, forcing, and solid-wall configuration.
- Ensure the maximum velocity remains in the specified low-Mach-number regime, `Ma = u_max / cs < 0.1`.
- Compare the computed velocity profile against the analytical solution using the documented coordinate convention.
- Check transverse velocity, mass conservation, and steady-state behavior.
- Account for the wall location and effective channel width when evaluating the analytical profile.

Do not accept a test merely because the observed profile looks qualitatively correct.

Do not adjust a test to accommodate a questionable implementation without first investigating the numerical discrepancy.

When a test fails, identify whether the cause is an implementation bug, an incorrect test assumption, an inconsistent mathematical convention, or an unsupported backend behavior.

## 11. Performance and Code Style

Prefer clear, idiomatic Python and PyTorch.

- Use descriptive names for physical quantities and numerical parameters.
- Avoid unnecessary wrappers, deep inheritance, and over-generalized APIs.
- Keep numerical kernels easy to inspect.
- Avoid hidden allocations, repeated full-tensor copies, and unnecessary synchronization in performance-critical paths.
- Optimize only after correctness has been established.
- Use profiling or representative benchmarks to justify non-trivial performance changes.

Do not introduce dependencies without a clear benefit.

Follow existing formatting, typing, naming, and testing conventions. Do not add tooling or repository-wide style changes unrelated to the task.

## 12. Documentation and Communication

Documentation should be concise, technically accurate, and directly useful for implementation, verification, or future research.

Prefer plain Markdown with clear explanations over elaborate templates or excessive formatting.

Keep mathematical assumptions and implementation contracts in the appropriate documentation files rather than duplicating lengthy explanations across source files.

When discussing a design:

- Be direct and technically precise.
- Distinguish facts, assumptions, recommendations, and unresolved questions.
- Explain important trade-offs rather than listing every imaginable alternative.
- Ask only questions that materially affect correctness, architecture, or scope.

Do not create additional documentation files merely to make the project appear more complete.

## 13. Scope Discipline

The current objective is a reliable, verifiable LBM prototype that can serve as a foundation for future scientific research.

Do not silently expand the scope to include:

- Fully implemented D3Q19/D3Q27 solvers.
- TRT/MRT.
- Slip or high-Knudsen-number boundary models.
- Variable-viscosity physics.
- Additional output backends.
- Distributed computing.
- Complex configuration systems.
- Premature GPU-specific optimization.

Keep future requirements visible, but implement them only when explicitly requested and when their mathematical and engineering requirements are sufficiently defined.