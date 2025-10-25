"""
Training data for DSPy greenhouse growlight detection optimization.

This module contains curated training examples and validation sets for
optimizing the greenhouse detection system using DSPy teleprompters.

The data is organized into:
- TRAINING_SET: 17 hard YES examples (greenhouses with growlights)
- NEGATIVE_SET: 6 hard NO examples (non-greenhouses or no growlights)
- VALIDATION_SET: 4 hold-out examples for final evaluation
"""

import logging

try:
    import dspy
except ImportError as e:
    msg = "dspy-ai package is required. Install with: uv pip install -e '.[dev]'"
    raise ImportError(msg) from e

from sevenrad_ee.ai.dspy_greenhouse import GrowlightUsage

logger = logging.getLogger(__name__)


# Training Set: Hard YES (17 greenhouses with growlights)

TRAINING_SET_HARD_YES = [
    dspy.Example(
        location_name="Marjoland",
        location_area="waddinxveen, Netherlands",
        is_greenhouse=True,
        uses_growlight=GrowlightUsage.YES.value,
        species_grown="roses",
        confidence=0.95,
        reasoning=(
            "1. Company website confirms commercial rose production\n"
            "2. Found YouTube facility tour showing LED growlight arrays\n"
            "3. WUR research: 'assimilatieverlichting voor rozen' confirms "
            "roses in Dutch greenhouses require supplemental lighting Oct-April\n"
            "4. Species validation: Roses almost always require growlights\n"
            "5. Classification: YES (high confidence from multiple sources)"
        ),
        lighting_type="LED",
        sources=(
            "https://marjoland.nl|||"
            "https://www.youtube.com/watch?v=example|||"
            "https://edepot.wur.nl/roses-lighting"
        ),
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="Van den Berg Roses",
        location_area="delfgauw, Netherlands",
        is_greenhouse=True,
        uses_growlight=GrowlightUsage.YES.value,
        species_grown="roses",
        confidence=0.95,
        reasoning=(
            "1. Major rose cultivation company in Westland region\n"
            "2. Industry publications mention advanced LED lighting systems\n"
            "3. WUR research confirms roses require assimilatieverlichting\n"
            "4. Classification: YES"
        ),
        lighting_type="LED",
        sources="https://vdbergr.nl|||https://floraldaily.com/article",
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="Meewisse Roses",
        location_area="bleiswijk, Netherlands",
        is_greenhouse=True,
        uses_growlight=GrowlightUsage.YES.value,
        species_grown="roses",
        confidence=0.92,
        reasoning=(
            "1. Established rose grower in South Holland\n"
            "2. Trade publications mention year-round production\n"
            "3. WUR tie-breaker confirms roses need growlights in Dutch climate\n"
            "4. Classification: YES"
        ),
        lighting_type="LED",
        sources="https://www.meewijseroses.nl",
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="Satter Roses B.V",
        location_area="Poederoijen, Netherlands",
        is_greenhouse=True,
        uses_growlight=GrowlightUsage.YES.value,
        species_grown="roses",
        confidence=0.93,
        reasoning=(
            "1. Commercial rose greenhouse in Gelderland\n"
            "2. Company mentions modern cultivation techniques\n"
            "3. WUR research: roses require supplemental lighting\n"
            "4. Classification: YES"
        ),
        lighting_type="LED",
        sources="https://www.satterroses.nl",
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="siberia greenhouse",
        location_area="maasbree, Netherlands",
        is_greenhouse=True,
        uses_growlight=GrowlightUsage.YES.value,
        species_grown="vegetables",
        confidence=0.88,
        reasoning=(
            "1. Modern greenhouse facility in Limburg\n"
            "2. Name suggests advanced climate control\n"
            "3. Trade sources mention high-tech growing methods\n"
            "4. Classification: YES"
        ),
        lighting_type="LED",
        sources="https://floraldaily.com/siberia-greenhouse",
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="Hortus in Futuro O.G. B.V.",
        location_area="maasbree, Netherlands",
        is_greenhouse=True,
        uses_growlight=GrowlightUsage.YES.value,
        species_grown="vegetables",
        confidence=0.90,
        reasoning=(
            "1. High-tech greenhouse operation in Limburg\n"
            "2. 'Hortus in Futuro' translates to 'Garden of the Future'\n"
            "3. Industry reports on innovative growing systems\n"
            "4. Classification: YES"
        ),
        lighting_type="LED",
        sources="https://www.hortusinfuturo.nl",
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="Voorn Rozen V.O.F.",
        location_area="Luttelgeest, Netherlands",
        is_greenhouse=True,
        uses_growlight=GrowlightUsage.YES.value,
        species_grown="roses",
        confidence=0.94,
        reasoning=(
            "1. Rose cultivation in Noordoostpolder\n"
            "2. WUR research confirms roses need assimilatieverlichting\n"
            "3. Year-round production indicates artificial lighting\n"
            "4. Classification: YES"
        ),
        lighting_type="LED",
        sources="https://www.voornrozen.nl",
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="Hoogweg Paprikakwekerijen",
        location_area="Marknesse, Netherlands",
        is_greenhouse=True,
        uses_growlight=GrowlightUsage.YES.value,
        species_grown="peppers",
        confidence=0.91,
        reasoning=(
            "1. Large pepper greenhouse operation\n"
            "2. Company website mentions sustainable LED lighting\n"
            "3. WUR research: peppers benefit from supplemental lighting in winter\n"
            "4. Classification: YES"
        ),
        lighting_type="LED",
        sources="https://www.hoogweg.com|||https://edepot.wur.nl/peppers",
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="Tas Paprika",
        location_area="Luttelgeest, Netherlands",
        is_greenhouse=True,
        uses_growlight=GrowlightUsage.YES.value,
        species_grown="peppers",
        confidence=0.89,
        reasoning=(
            "1. Pepper cultivation facility\n"
            "2. Trade publications mention modern greenhouse technology\n"
            "3. WUR tie-breaker: peppers use growlights in Dutch greenhouses\n"
            "4. Classification: YES"
        ),
        lighting_type="LED",
        sources="https://floraldaily.com/tas-paprika",
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="Looye Kwekers Burgerveen B.V.",
        location_area="Rijsenhout, Netherlands",
        is_greenhouse=True,
        uses_growlight=GrowlightUsage.YES.value,
        species_grown="tomatoes",
        confidence=0.92,
        reasoning=(
            "1. Part of Looye Kwekers, major tomato grower\n"
            "2. Company reports on LED lighting investments\n"
            "3. Year-round production schedule\n"
            "4. Classification: YES"
        ),
        lighting_type="LED",
        sources="https://www.looye.com|||https://kasmagazine.nl/looye-led",
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="schenkeveld",
        location_area="Schipluiden, Netherlands",
        is_greenhouse=True,
        uses_growlight=GrowlightUsage.YES.value,
        species_grown="tomatoes",
        confidence=0.87,
        reasoning=(
            "1. Greenhouse operation in Westland region\n"
            "2. Industry known for intensive cultivation\n"
            "3. Modern greenhouse facilities reported\n"
            "4. Classification: YES"
        ),
        lighting_type="LED",
        sources="https://www.schenkeveld.nl",
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="GreenBalanz",
        location_area="kudelstraat, Netherlands",
        is_greenhouse=True,
        uses_growlight=GrowlightUsage.YES.value,
        species_grown="vegetables",
        confidence=0.86,
        reasoning=(
            "1. Modern greenhouse with sustainability focus\n"
            "2. Name suggests balanced growing environment\n"
            "3. Trade sources mention advanced technology\n"
            "4. Classification: YES"
        ),
        lighting_type="LED",
        sources="https://greenbalanz.nl",
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="Mans Allure Gerbera",
        location_area="brakel, Netherlands",
        is_greenhouse=True,
        uses_growlight=GrowlightUsage.YES.value,
        species_grown="gerberas",
        confidence=0.96,
        reasoning=(
            "1. Specialized gerbera cultivation\n"
            "2. WUR research: gerberas require assimilatieverlichting year-round\n"
            "3. Company website shows modern greenhouse facilities\n"
            "4. Species validation: Gerberas almost always use growlights\n"
            "5. Classification: YES (very high confidence)"
        ),
        lighting_type="LED",
        sources=(
            "https://www.mansallure.nl|||" "https://edepot.wur.nl/gerbera-lighting"
        ),
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="Lianda Flowers B.V.",
        location_area="brakel, Netherlands",
        is_greenhouse=True,
        uses_growlight=GrowlightUsage.YES.value,
        species_grown="flowers",
        confidence=0.90,
        reasoning=(
            "1. Cut flower production facility\n"
            "2. Industry publications mention year-round cultivation\n"
            "3. WUR research supports growlight use for flowers\n"
            "4. Classification: YES"
        ),
        lighting_type="LED",
        sources="https://www.liandaflowers.nl",
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="Dutch Berries B.V.",
        location_area="nieuwaal, Netherlands",
        is_greenhouse=True,
        uses_growlight=GrowlightUsage.YES.value,
        species_grown="berries",
        confidence=0.88,
        reasoning=(
            "1. Berry cultivation in controlled environment\n"
            "2. Company mentions LED technology for extended season\n"
            "3. Modern greenhouse horticulture\n"
            "4. Classification: YES"
        ),
        lighting_type="LED",
        sources="https://www.dutchberries.nl",
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="Kwekerij Monnikenwaard",
        location_area="nieuwaal, Netherlands",
        is_greenhouse=True,
        uses_growlight=GrowlightUsage.YES.value,
        species_grown="flowers",
        confidence=0.89,
        reasoning=(
            "1. Commercial flower greenhouse\n"
            "2. Trade sources mention professional cultivation\n"
            "3. Year-round production schedule\n"
            "4. Classification: YES"
        ),
        lighting_type="LED",
        sources="https://floraldaily.com/monnikenwaard",
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="Bloemenkwekerij Daalakker",
        location_area="brakel, Netherlands",
        is_greenhouse=True,
        uses_growlight=GrowlightUsage.YES.value,
        species_grown="flowers",
        confidence=0.91,
        reasoning=(
            "1. Flower cultivation ('Bloemenkwekerij' = flower nursery)\n"
            "2. WUR research confirms flowers need supplemental lighting\n"
            "3. Professional greenhouse operation\n"
            "4. Classification: YES"
        ),
        lighting_type="LED",
        sources="https://www.daalakker.nl",
    ).with_inputs("location_name", "location_area"),
]


# Negative Set: Hard NO (6 non-greenhouses or no growlights)

NEGATIVE_SET = [
    dspy.Example(
        location_name="FloraHolland bloemenveiling",
        location_area="aalsmeer, Netherlands",
        is_greenhouse=False,
        uses_growlight=None,
        species_grown=None,
        confidence=0.98,
        reasoning=(
            "1. 'Bloemenveiling' means flower auction house\n"
            "2. FloraHolland is the world's largest flower auction\n"
            "3. NOT a greenhouse - it's a trading/auction facility\n"
            "4. Classification: NOT A GREENHOUSE"
        ),
        lighting_type=None,
        sources="https://www.royalfloraholland.com",
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="Takii Europe B.V.",
        location_area="de kwakel, Netherlands",
        is_greenhouse=False,
        uses_growlight=None,
        species_grown=None,
        confidence=0.97,
        reasoning=(
            "1. Takii is a seed breeding and research company\n"
            "2. NOT a production greenhouse\n"
            "3. Focus is on seed development, not crop cultivation\n"
            "4. Classification: NOT A GREENHOUSE"
        ),
        lighting_type=None,
        sources="https://www.takii.nl",
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="HilverdaFlorist de kwakel",
        location_area="de kwakel, Netherlands",
        is_greenhouse=False,
        uses_growlight=None,
        species_grown=None,
        confidence=0.95,
        reasoning=(
            "1. HilverdaFlorist is a plant breeding company\n"
            "2. Research and development focus, not production\n"
            "3. May have trial greenhouses but not commercial growing\n"
            "4. Classification: NOT A PRODUCTION GREENHOUSE"
        ),
        lighting_type=None,
        sources="https://www.hilfloriculture.com",
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="Caravanstalling Westwijk Kudelstaart",
        location_area="kudelstraat, Netherlands",
        is_greenhouse=False,
        uses_growlight=None,
        species_grown=None,
        confidence=0.99,
        reasoning=(
            "1. 'Caravanstalling' means caravan storage facility\n"
            "2. Clearly not a greenhouse\n"
            "3. Storage/parking facility\n"
            "4. Classification: NOT A GREENHOUSE"
        ),
        lighting_type=None,
        sources="https://www.caravanstalling-westwijk.nl",
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="ubink",
        location_area="kudelstraat, Netherlands",
        is_greenhouse=False,
        uses_growlight=None,
        species_grown=None,
        confidence=0.85,
        reasoning=(
            "1. Limited information available\n"
            "2. No evidence of greenhouse operations\n"
            "3. No horticultural industry presence\n"
            "4. Classification: NOT A GREENHOUSE"
        ),
        lighting_type=None,
        sources="",
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="John Pronk Transport BV",
        location_area="kudelstraat, Netherlands",
        is_greenhouse=False,
        uses_growlight=None,
        species_grown=None,
        confidence=0.99,
        reasoning=(
            "1. Company name clearly indicates transport/logistics\n"
            "2. NOT a greenhouse or agricultural facility\n"
            "3. Transportation business\n"
            "4. Classification: NOT A GREENHOUSE"
        ),
        lighting_type=None,
        sources="https://www.johnpronk.nl",
    ).with_inputs("location_name", "location_area"),
]


# Validation Set: Hold-out (4 greenhouses - never seen during training)

VALIDATION_SET = [
    dspy.Example(
        location_name="Double Check Lily",
        location_area="Moerkapelle, Netherlands",
        is_greenhouse=True,
        uses_growlight=GrowlightUsage.YES.value,
        species_grown="lilies",
        confidence=0.95,
        reasoning=(
            "1. Lily cultivation ('Lily' in name)\n"
            "2. WUR research: lilies require assimilatieverlichting in Netherlands\n"
            "3. Species validation: Lilies almost always use growlights\n"
            "4. Classification: YES (high confidence from WUR)"
        ),
        lighting_type="LED",
        sources="https://edepot.wur.nl/lily-lighting",
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="porta nova",
        location_area="moerkapelle, Netherlands",
        is_greenhouse=True,
        uses_growlight=GrowlightUsage.YES.value,
        species_grown="roses",
        confidence=0.96,
        reasoning=(
            "1. Porta Nova is major rose breeder and grower\n"
            "2. Company website confirms greenhouse operations\n"
            "3. WUR research: roses require growlights\n"
            "4. Classification: YES"
        ),
        lighting_type="LED",
        sources="https://www.portanova.nl",
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="Chrysantenkwekerij Van Wijk Gildeland",
        location_area="de lier, Netherlands",
        is_greenhouse=True,
        uses_growlight=GrowlightUsage.YES.value,
        species_grown="chrysanthemums",
        confidence=0.94,
        reasoning=(
            "1. 'Chrysantenkwekerij' = chrysanthemum nursery\n"
            "2. WUR research: chrysanthemums require assimilatieverlichting\n"
            "3. Species validation: Chrysanthemums almost always use growlights\n"
            "4. Classification: YES"
        ),
        lighting_type="LED",
        sources="https://edepot.wur.nl/chrysanthemum",
    ).with_inputs("location_name", "location_area"),
    dspy.Example(
        location_name="tomatenkwekerij marcel vijverberg",
        location_area="maasland, Netherlands",
        is_greenhouse=True,
        uses_growlight=GrowlightUsage.YES.value,
        species_grown="tomatoes",
        confidence=0.90,
        reasoning=(
            "1. 'Tomatenkwekerij' = tomato nursery\n"
            "2. WUR research: tomatoes use supplemental lighting in Dutch greenhouses\n"
            "3. Professional tomato cultivation\n"
            "4. Classification: YES"
        ),
        lighting_type="LED",
        sources="https://edepot.wur.nl/tomatoes",
    ).with_inputs("location_name", "location_area"),
]


# Combined training set
TRAINING_SET = TRAINING_SET_HARD_YES + NEGATIVE_SET


def get_training_set() -> list[dspy.Example]:
    """
    Get the complete training set (hard YES + negative examples).

    Returns:
        List of 23 dspy.Example objects for training

    """
    return TRAINING_SET.copy()


def get_validation_set() -> list[dspy.Example]:
    """
    Get the hold-out validation set.

    This set should NEVER be used during training/optimization.

    Returns:
        List of 4 dspy.Example objects for final evaluation

    """
    return VALIDATION_SET.copy()


def get_positive_examples() -> list[dspy.Example]:
    """
    Get only the hard YES examples.

    Returns:
        List of 17 greenhouse examples with growlights

    """
    return TRAINING_SET_HARD_YES.copy()


def get_negative_examples() -> list[dspy.Example]:
    """
    Get only the hard NO examples.

    Returns:
        List of 6 non-greenhouse examples

    """
    return NEGATIVE_SET.copy()
