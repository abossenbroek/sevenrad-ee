"""Test DSPy 3.0.3 optimizer imports based on official documentation."""

print("Testing DSPy 3.0.3 Optimizer Imports")
print("=" * 60)

# Test 1: Import all from teleprompt
print("\n1. Testing: from dspy.teleprompt import *")
try:
    from dspy.teleprompt import *
    print("   ✓ Success")
    print(f"   Available names: {[x for x in dir() if not x.startswith('_')]}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 2: Direct GEPA import
print("\n2. Testing: import dspy; dspy.GEPA")
try:
    import dspy
    gepa_class = dspy.GEPA
    print(f"   ✓ dspy.GEPA accessible: {gepa_class}")
except AttributeError:
    print("   ✗ dspy.GEPA not available as attribute")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 3: BootstrapFewShot variations
print("\n3. Testing BootstrapFewShot variations:")
variations = [
    "BootstrapFewShot",
    "BootstrapFewShotWithRandomSearch",
    "bootstrap",
]
for var in variations:
    try:
        exec(f"from dspy.teleprompt import {var}")
        print(f"   ✓ {var} imported successfully")
    except ImportError:
        print(f"   ✗ {var} not found")

# Test 4: MIPROv2
print("\n4. Testing: from dspy.teleprompt import MIPROv2")
try:
    from dspy.teleprompt import MIPROv2
    print(f"   ✓ MIPROv2 imported: {MIPROv2}")
except ImportError as e:
    print(f"   ✗ MIPROv2 not found: {e}")

# Test 5: SIMBA (mentioned in docs)
print("\n5. Testing: from dspy.teleprompt import SIMBA")
try:
    from dspy.teleprompt import SIMBA
    print(f"   ✓ SIMBA imported: {SIMBA}")
except ImportError:
    print("   ✗ SIMBA not found")

# Test 6: Check what's in simba module directly
print("\n6. Checking simba module contents:")
try:
    from dspy.teleprompt import simba
    print(f"   Module contents: {[x for x in dir(simba) if not x.startswith('_')]}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n" + "=" * 60)
print("\nRECOMMENDATION:")
print("Based on tests, using available optimizers for 95-98% F1 target")
