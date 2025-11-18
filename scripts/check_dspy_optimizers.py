"""Check available DSPy optimizers in current installation."""
import dspy.teleprompt as tp
import inspect

optimizers = [
    name for name, obj in inspect.getmembers(tp)
    if inspect.isclass(obj) and not name.startswith('_')
]

print("Available DSPy teleprompt optimizers:")
print("=" * 50)
for opt in sorted(optimizers):
    print(f"  ✓ {opt}")

print("\n" + "=" * 50)
print("\nChecking for GEPA specifically:")
has_gepa = 'GEPA' in optimizers
print(f"  GEPA available: {has_gepa}")

if not has_gepa:
    print("\n⚠️  GEPA not available - will use MIPROv2 fallback")
    print("  Expected target: 90-93% F1 (vs 95-98% with GEPA)")
