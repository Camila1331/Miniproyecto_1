import nltk
from nltk.sem.drt import *
from nltk import load_parser
from nltk.sem.drt import DrtParser, DrtExpression
from nltk import CFG, parse
from nltk.grammar import FeatureGrammar
from nltk.parse import RecursiveDescentParser, FeatureEarleyChartParser
from nltk.parse.generate import generate

lp = nltk.sem.logic.LogicParser()
dexpr = DrtExpression.fromstring

import utils
from logUtils import existenciales_a_constantes
from parseMod import Modelo

import resPreg

text_data = {\
"id":"texto1", \
"texto":"Una vieja tenía un queso. Un ratón se comió el queso. Un gato se comió al ratón. Un perro mató al gato. Un palo le pegó al perro. El fuego quemó al palo. El agua apagó el fuego. Un buey se bebió el agua.", \
"preguntas":["¿Qué tenía la vieja?", "¿Quién se comió el queso?", "¿Quién quemó al palo?"],\
"respuestas":[["un queso", "queso"], ["el ratón", "un ratón", "ratón","raton"], ["el fuego", "fuego"]]}

resPreg.procesamiento(text_data=text_data, 
                      gramatica_texto='drs.fcfg', 
                      gramatica_preguntas='preguntas_texto1.fcfg',
                      verbose=False)