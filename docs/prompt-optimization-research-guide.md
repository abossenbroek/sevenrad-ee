# Prompt Optimization Research Guide for Greenhouse Attribution

**Documentation Type: How-to Guide**

This guide shows you how to use DSPy with BootstrapFewShot to optimize the Perplexity prompts used in greenhouse attribution analysis. This is a research workflow to improve prompt accuracy before integrating into production.

## What You'll Learn

- How to run DSPy prompt optimization experiments
- How to evaluate prompt performance with metrics
- How to use BootstrapFewShot for few-shot learning
- How to test optimized prompts on new businesses
- How to save and integrate optimized prompts

## Why Optimize Prompts?

The current `attribution.py` uses a hand-crafted prompt to extract greenhouse data from Perplexity. DSPy's optimization can:

1. **Improve Accuracy**: Automatically find better prompt phrasings
2. **Add Examples**: Bootstrap few-shot demonstrations from training data
3. **Reduce Errors**: Optimize for structured output consistency
4. **Evidence-Based**: Use metrics to objectively compare prompt versions

## Prerequisites

1. **Perplexity API Key**: Set in `.env` file or environment
2. **Dependencies installed**: `uv pip install -e ".[dev]"`
3. **Python 3.12+**: Required for the project

## Quick Start

### 1. View Training Examples (Dry Run)

See what training data will be used for optimization:

```bash
uv run python -m sevenrad_ee.operations.optimize_attribution_prompt --dry-run
```

**Output:**
```
Training Examples
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Business            ┃ Location            ┃ Likelihood ┃ Crops        ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ Royal Van Zanten    │ Rijsenhout, NL      │ 0.95       │ gerbera,rose │
│ Duijvestijn Tomaten │ Pijnacker, NL       │ 0.90       │ tomato       │
│ Prominent           │ Poeldijk, NL        │ 0.95       │ gerbera      │
│ De Kas Restaurant   │ Amsterdam, NL       │ 0.10       │              │
│ Koppert Cress       │ Monster, NL         │ 0.85       │ microgreens  │
└─────────────────────┴─────────────────────┴────────────┴──────────────┘
```

### 2. Run Optimization

Run BootstrapFewShot optimization with default settings:

```bash
export PERPLEXITY_API_KEY='your-key-here'

uv run python -m sevenrad_ee.operations.optimize_attribution_prompt
```

**What happens:**
1. Loads 5 training examples (known greenhouses)
2. Configures DSPy with Perplexity LM
3. Runs BootstrapFewShot to find optimal demonstrations
4. Evaluates on test set
5. Shows performance metrics

**Output:**
```
╭───────────────────────────────────────────────────────╮
│ DSPy Prompt Optimization for Greenhouse Attribution   │
│ Using BootstrapFewShot to optimize Perplexity prompts │
╰───────────────────────────────────────────────────────╯

Configuring DSPy with model: perplexity/sonar
✓ DSPy configured successfully

⠋ Optimizing prompt with BootstrapFewShot...
✓ Optimization complete!

Evaluation Results
┏━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric        ┃ Value ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Average Score │ 0.847 │
│ Test Examples │ 5     │
└───────────────┴───────┘

Individual Scores: ['0.920', '0.850', '0.900', '0.750', '0.815']
```

### 3. Test on a Specific Business

Test the optimized prompt on a new business:

```bash
uv run python -m sevenrad_ee.operations.optimize_attribution_prompt \
  --test-business "Waterdrinker Aalsmeer" \
  --test-location "Aalsmeer, Netherlands"
```

**Output:**
```
Testing on: Waterdrinker Aalsmeer (Aalsmeer, Netherlands)

Analysis Results:

Grow Light Likelihood: 0.92
Lighting Evidence: Uses LED and SON-T supplemental lighting for orchid production
Primary Crops: orchid, phalaenopsis
Size: 8.5 ha
Instagram: waterdrinker_aalsmeer
Sources:
  - https://www.waterdrinker.nl
  - https://www.floraldaily.com/article/9345678
  - https://www.kasmagazine.nl/waterdrinker

Reasoning: Waterdrinker grows orchids (high light requirement crop) with
confirmed LED and SON-T lighting from multiple Dutch sources. Large-scale
operation indicates artificial lighting infrastructure.
```

## Configuration Options

### Perplexity Model Selection

Choose different Perplexity models for different trade-offs:

```bash
# Fast and cheaper (default)
uv run python -m sevenrad_ee.operations.optimize_attribution_prompt \
  --model perplexity/sonar

# More accurate, slower
uv run python -m sevenrad_ee.operations.optimize_attribution_prompt \
  --model perplexity/sonar-pro
```

### Training Set Size

Adjust number of training examples:

```bash
# Use more examples for better optimization (but slower)
uv run python -m sevenrad_ee.operations.optimize_attribution_prompt \
  --max-examples 10

# Use fewer for faster iteration
uv run python -m sevenrad_ee.operations.optimize_attribution_prompt \
  --max-examples 3
```

### Few-Shot Demonstration Control

Control how many demonstrations are bootstrapped:

```bash
uv run python -m sevenrad_ee.operations.optimize_attribution_prompt \
  --max-bootstrapped-demos 5 \
  --max-labeled-demos 3
```

**Parameters:**
- `--max-bootstrapped-demos`: Max demonstrations created by bootstrapping (default: 3)
- `--max-labeled-demos`: Max labeled demonstrations to use (default: 2)

### Save Optimized Module

Save the optimized module for later use:

```bash
uv run python -m sevenrad_ee.operations.optimize_attribution_prompt \
  --output optimized_prompt_2025-10-25.json
```

## Understanding the Optimization Process

### 1. Training Examples

The system uses 5 known greenhouse examples:

1. **Royal Van Zanten** - Large gerbera/rose grower (positive)
2. **Duijvestijn Tomaten** - Tomato greenhouse (positive)
3. **Prominent** - Gerbera specialist (positive)
4. **De Kas Restaurant** - Restaurant in former greenhouse (negative)
5. **Koppert Cress** - Microgreens with LED (positive)

These provide diverse scenarios for optimization.

### 2. BootstrapFewShot Strategy

BootstrapFewShot works by:

1. **Running base module** on training examples
2. **Selecting best demonstrations** that improve predictions
3. **Adding demonstrations** to the prompt as examples
4. **Iterating** to find optimal few-shot set

This is different from BootstrapFinetune which requires a model that supports finetuning.

### 3. Evaluation Metric

The system uses a composite metric (0.0-1.0):

**Components:**
- **Likelihood Accuracy (40%)**: How close is predicted likelihood to truth?
- **Evidence Quality (30%)**: Does evidence mention lighting keywords?
- **Source Relevance (30%)**: Are Dutch sources prioritized?

**Example Calculation:**
```
Prediction: 0.92 likelihood, mentions "LED", has .nl sources
Ground Truth: 0.95 likelihood

Likelihood Score: 1.0 - |0.92 - 0.95| = 0.97 → 0.40 * 0.97 = 0.388
Evidence Score: Mentions "LED" → 0.30
Source Score: Has .nl domain + priority source → 0.15 + 0.15 = 0.30

Total: 0.388 + 0.30 + 0.30 = 0.988
```

## Advanced Usage

### Adding Custom Training Examples

Edit `src/sevenrad_ee/ai/dspy_attribution.py` to add more examples:

```python
def create_training_examples() -> list[dspy.Example]:
    """Create training examples for DSPy optimization."""
    examples = [
        # Add your custom example
        dspy.Example(
            business_name="Your Greenhouse BV",
            location_context="Your City, Netherlands",
            business_types="greenhouse,agricultural",
            grow_light_likelihood=0.90,
            lighting_evidence="Uses LED assimilatieverlichting for tomatoes",
            primary_crops="tomato,pepper",
            size_hectares=15.0,
            instagram_handle="yourgreenhouse",
            sources=(
                "https://www.yourgreenhouse.nl ||| "
                "https://www.kasmagazine.nl/your-article"
            ),
            reasoning=(
                "Your explanation of why this greenhouse uses artificial lighting"
            ),
        ).with_inputs("business_name", "location_context", "business_types"),

        # ... existing examples
    ]
    return examples
```

### Adjusting the Evaluation Metric

Modify weights in `greenhouse_attribution_metric()`:

```python
# Current weights
likelihood_score_weight = 0.4  # 40%
evidence_score_weight = 0.3    # 30%
source_score_weight = 0.3      # 30%

# Example: Prioritize likelihood accuracy more
likelihood_score_weight = 0.6  # 60%
evidence_score_weight = 0.2    # 20%
source_score_weight = 0.2      # 20%
```

### Comparing Multiple Optimizations

Run multiple experiments and compare:

```bash
# Experiment 1: Few demonstrations
uv run python -m sevenrad_ee.operations.optimize_attribution_prompt \
  --max-bootstrapped-demos 2 \
  --output exp1_few_demos.json

# Experiment 2: Many demonstrations
uv run python -m sevenrad_ee.operations.optimize_attribution_prompt \
  --max-bootstrapped-demos 5 \
  --output exp2_many_demos.json

# Compare evaluation scores manually
```

## Integration into Production

### Step 1: Verify Performance

Run optimization multiple times to ensure consistent results:

```bash
for i in {1..3}; do
  echo "Run $i:"
  uv run python -m sevenrad_ee.operations.optimize_attribution_prompt
done
```

Look for:
- Average score > 0.80
- Consistent scores across runs
- Good performance on both positive and negative examples

### Step 2: Test on Real Data

Use `--test-business` extensively on your target greenhouses:

```bash
# Test on various greenhouse types
uv run python -m sevenrad_ee.operations.optimize_attribution_prompt \
  --test-business "Your Target Greenhouse 1" \
  --test-location "Location"

# Check reasoning and evidence quality
```

### Step 3: Extract Optimized Prompt

The optimized prompt is stored in the DSPy module. To view it:

```python
import dspy
from sevenrad_ee.ai.dspy_attribution import GreenhouseAttributionModule

# Load optimized module
module = GreenhouseAttributionModule()
module.load("optimized_prompt.json")

# Inspect the predictor
print(module.predictor.signature)
print(module.predictor.demos)  # Few-shot demonstrations
```

### Step 4: Update attribution.py

Once satisfied with results, update the prompt in `attribution.py`:

1. Copy the optimized signature description
2. Update the prompt string in `analyze_business()` method
3. Add few-shot examples if beneficial
4. Test thoroughly on real attribution pipeline

## Troubleshooting

### Issue: Low Evaluation Scores

**Symptoms**: Average score < 0.70

**Solutions:**
1. **Add more training examples**: Include diverse greenhouse types
2. **Adjust metric weights**: Ensure metric aligns with your priorities
3. **Try different model**: Use `--model perplexity/sonar-pro`
4. **Increase demonstrations**: Use `--max-bootstrapped-demos 5`

### Issue: Inconsistent Results

**Symptoms**: Evaluation scores vary widely between runs

**Solutions:**
1. **Add more training data**: 5 examples may not be enough
2. **Check for API rate limits**: Perplexity may be throttling
3. **Increase labeled demos**: Use `--max-labeled-demos 4`
4. **Set random seed**: (Not currently supported, would need code change)

### Issue: "Access denied" from Perplexity

**Symptoms**: API returns 403 error

**Solutions:**
1. Verify API key: `echo $PERPLEXITY_API_KEY`
2. Check API credits remaining
3. Try different model: `--model perplexity/sonar`
4. Wait and retry (rate limit cooldown)

### Issue: Optimization is too slow

**Symptoms**: Takes > 5 minutes

**Solutions:**
1. Reduce training examples: `--max-examples 3`
2. Reduce demonstrations: `--max-bootstrapped-demos 2`
3. Use faster model: `--model perplexity/sonar` (not sonar-pro)
4. Run during off-peak hours (faster API responses)

## Best Practices

### 1. Start Small

Begin with minimal configuration:

```bash
uv run python -m sevenrad_ee.operations.optimize_attribution_prompt \
  --max-examples 3 \
  --max-bootstrapped-demos 2 \
  --dry-run  # First, just see the data
```

Gradually increase complexity as you understand the system.

### 2. Iterate on Training Data

The quality of training examples matters more than quantity:

- **Include edge cases**: Restaurants in greenhouses, ornamental vs. food crops
- **Balance positive/negative**: Don't only use positive examples
- **Verify ground truth**: Ensure likelihood scores are accurate
- **Add regional variety**: Dutch, German, North American greenhouses

### 3. Document Your Experiments

Keep a log of optimization runs:

```markdown
# Optimization Log

## 2025-10-25 - Experiment 1
- Config: max_examples=5, max_demos=3
- Model: perplexity/sonar
- Score: 0.847
- Notes: Good evidence extraction, some false positives

## 2025-10-25 - Experiment 2
- Config: max_examples=10, max_demos=5
- Model: perplexity/sonar-pro
- Score: 0.923
- Notes: Much better, longer running time acceptable
```

### 4. Test on Diverse Businesses

Don't just test on greenhouses - test on:
- **Garden centers** (may have some grow lights)
- **Restaurants** (false positives to avoid)
- **Vertical farms** (should detect)
- **Research facilities** (may have experimental lighting)

## Comparison: Before vs. After Optimization

### Before (Hand-crafted Prompt)

```python
# Manual prompt in attribution.py (lines 224-267)
prompt = (
    f"Analyze greenhouse operation: {business_info} near {location_info} "
    "to determine if it uses artificial lighting..."
    # ... 40+ lines of instructions
)
```

**Characteristics:**
- ✅ Explicit anti-greenwashing instructions
- ✅ Clear output format specification
- ❌ No examples (zero-shot)
- ❌ Hand-tuned phrasing
- ❌ Difficult to improve systematically

### After (DSPy Optimized)

```python
# DSPy module with CoT
module = GreenhouseAttributionModule()
prediction = module(business_name, location, types)
```

**Characteristics:**
- ✅ Few-shot demonstrations automatically selected
- ✅ Chain-of-Thought reasoning
- ✅ Optimized based on performance metrics
- ✅ Systematic improvement process
- ✅ Evidence-based confidence

## Next Steps

1. **Run first optimization**: Start with defaults
2. **Analyze results**: Look at evaluation metrics and reasoning
3. **Adjust training data**: Add examples that improve weak areas
4. **Iterate**: Re-run with adjusted configuration
5. **Test thoroughly**: Use `--test-business` extensively
6. **Integrate**: Update `attribution.py` when satisfied

## Related Documentation

- **Tutorial**: [Getting Started with DSPy](https://dspy.ai/tutorials/)
- **Reference**: [BootstrapFewShot API](https://dspy.ai/api/teleprompt/BootstrapFewShot/)
- **Explanation**: [Why Use DSPy for Prompts](./dspy-explanation.md)
- **Source Code**: `src/sevenrad_ee/ai/dspy_attribution.py`

## Summary

The prompt optimization workflow provides:

- 🔬 **Research Tool**: Experiment with prompt improvements
- 📊 **Metrics-Driven**: Objective evaluation of prompt quality
- 🎯 **Targeted**: Focus on greenhouse attribution accuracy
- 🔄 **Iterative**: Systematic improvement process
- 🚀 **Production-Ready**: Integrate results into attribution.py

Use this workflow to systematically improve Perplexity prompt accuracy before deploying to production greenhouse attribution.
