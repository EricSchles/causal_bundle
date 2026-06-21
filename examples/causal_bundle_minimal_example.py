from causal_bundle.system.geometric_system import CausalGeometricSystem

system = CausalGeometricSystem(
    environments=envs,        # from 4.2
    encoder=encoder,          # from 4.3
    pysr_model=pysr_model     # symbolic regression
)

results = system.run_full_pipeline(X, Y)

print(results["invariance"])
print(results["fiber"].model)
