import os
import sys
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..'
        )
    )
)

from markovchain.attribution import Attribution, Markov
from markovchain.query import ADHQueryError, Campaign, ADH, Query
from markovchain.decorators import validate_config
