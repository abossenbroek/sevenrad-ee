"""Check available DSPy teleprompt modules and optimizers."""
import pkgutil
from dspy import teleprompt

print("Available DSPy teleprompt modules:")
print("=" * 50)

modules = []
for importer, modname, ispkg in pkgutil.iter_modules(teleprompt.__path__):
    modules.append(modname)
    print(f"  ✓ {modname}")

print("\n" + "=" * 50)
print("\nChecking for specific optimizers:")
print(f"  BootstrapFewShot: {'bootstrap' in modules or 'bootstrap_few_shot' in modules}")
print(f"  MIPRO: {'mipro' in modules}")
print(f"  MIPROv2: {'mipro_v2' in modules or 'miprov2' in modules}")
print(f"  GEPA: {'gepa' in modules}")

print("\n" + "=" * 50)
print("\nTrying to import known optimizers:")

try:
    from dspy.teleprompt import BootstrapFewShot
    print("  ✓ BootstrapFewShot imported successfully")
except ImportError as e:
    print(f"  ✗ BootstrapFewShot: {e}")

try:
    from dspy.teleprompt import MIPRO
    print("  ✓ MIPRO imported successfully")
except ImportError as e:
    print(f"  ✗ MIPRO import failed")

try:
    from dspy.teleprompt import MIPROv2
    print("  ✓ MIPROv2 imported successfully")
except ImportError as e:
    print(f"  ✗ MIPROv2 import failed")

try:
    from dspy.teleprompt import GEPA
    print("  ✓ GEPA imported successfully")
except ImportError as e:
    print(f"  ✗ GEPA import failed")

print("\n" + "=" * 50)
print("\nRecommendation:")
if 'mipro' in modules or 'miprov2' in modules:
    print("  → Use MIPRO/MIPROv2 as primary optimizer")
    print("  → Expected F1: 90-93% (conservative), 95%+ (optimistic)")
else:
    print("  → Use BootstrapFewShot with enhanced features")
    print("  → Expected F1: 85-90% (realistic)")
