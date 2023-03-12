import re
import nltk
import nltk.data
from nltk import Tree
from nltk.grammar import FeatureGrammar
from nltk.parse import FeatureEarleyChartParser
from nltk.sem.drt import DrtParser, DrtExpression
nltk.download('punkt')
sent_detector = nltk.data.load('tokenizers/punkt/spanish.pickle')
lp = nltk.sem.logic.LogicParser()
dexpr = DrtExpression.fromstring


def preprocess_text(text:str) -> list:
    '''
    Pequeño preprocesamiento:
        * Eliminamos el punto final de la oración
        * Todo a minúsculas
        * Pone espacio alrededor de símbolos de interrogación
    Input:
        - text
    Output:
        - List of sentences
    '''
    sentences = sent_detector.tokenize(text.strip())
    sentences = [s for s in sentences if s not in ['.', ',', ';', ':']]
    new_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub(r"([¿?])", r" \1 ", sentence)
        if sentence[-1] in ['.', ' ']:
            sentence = sentence[:-1]
        if sentence[0] == ' ':
            sentence = sentence[1:]
        new_sentences.append(sentence)
    return new_sentences

def arbol_sin_caracteristicas(s:str) -> Tree:
    '''
    Toma un árbol de una cadena, de la cual elimina las características
    que se encuentran entre paréntesis cuadrados, y devuelve un Tree de nltk.
    Input:
        - s, una cadena con un árbol en representación plana
    Output:
        - un árbol de la clase Tree de la librería nltk
    '''
    s = s.replace('[', '{')
    s = s.replace(']', '}')
    s = re.sub('{.*?}', '', s)
    try:
        arbol = Tree.fromstring(s)
    except:
        s = re.sub(',.*?>}', '', s)  
        arbol = Tree.fromstring(s)
    return arbol

def parsear(tokens:list, parser, verbose=False) -> Tree:
    '''
    Toma una lista de tokens y devuelve el árbol de análisis
    usando el parser suministrado.
    Input:
        - toekns, una lista de cadenas con una oración
        - parser, un parser de nltk
        - verbose, booleano para imprimir información
    Output:
        - un árbol de la clase Tree de la librería nltk o None
    '''
    if verbose:
        print(f'Haciendo el parsing de la oración:\n\n\t{" ".join(tokens)}\n')
    trees = parser.parse(tokens)
    arboles = []
    for t in trees:
        arboles.append(str(t))
    if len(arboles) > 0:
        arbol = arboles[0]
        if verbose:
            print(f'El árbol lineal obtenido es:\n\n\t{arbol}\n')
        return arbol_sin_caracteristicas(arbol)
    else:
        if verbose:
            print('¡El parser no produjo ningún árbol!')
        return None

def clausurar(formula:nltk.sem.logic) -> nltk.sem.logic:
    '''
    Toma una fórmula y devuelve su clausura universal.
    Input:
        - formula, expresión lógica de nltk.sem.logic
    Output:
        - formula clausurada
    '''
    variables_libres = formula.free()
    if len(formula.free()) == 0:
        return formula
    else:
        clausura = ''
        for v in variables_libres:
            clausura += f'all {v}.'
        clausurada = f'{clausura}{formula}'
        return lp.parse(clausurada)
    
def obtener_formula(tokens:list, parser, clausura:bool=False) -> nltk.sem.logic:
    '''
    Toma una lista de tokens y devuelve su representación lógica.
    Input:
        - toekns, una lista de cadenas con una oración.
        - parser, un parser de nltk.
        - clausura, un booleano para devolver la fórmula clausurada o no.
    Output:
        - formula clausurada
    '''
    try:
        trees = parser.parse(tokens)
    except Exception as e:
        print(f'Error de parser: {e}')
        return None
    arboles = []
    for t in trees:
        arboles.append(t)
    if len(arboles) > 0:
        arbol = arboles[0]
        formula = arbol.label().get('SEM')
        if clausura:
            return clausurar(formula)
        else:
            return formula
    else:
        return None

def cambio_y_sencillo(x1:str, x2:str, drs:nltk.sem.drt) -> nltk.sem.drt:
    '''
    Toma una DRS y dos variables. Sustituye la una por la otra
    y elimina referentes y condiciones duplicados de la DRS.
    Input:
        - x1, cadena
        - x2, cadena
        - drs, drs en formato nltk 
    Output:
        - drs, en formato nltk
    '''
    x1 = dexpr(fr'{str(x1)}')
    x2 = dexpr(fr'{str(x2)}')
    drs_nueva = drs.replace(x1.variable, x2, True)
    drs_nueva.refs = list(set(drs_nueva.refs))
    drs_nueva.conds = list(set(drs_nueva.conds))
    drs_nueva = drs_nueva.simplify()
    return drs_nueva

def asegura_alfa_variante(drs1:nltk.sem.drt, drs2:nltk.sem.drt) -> nltk.sem.drt:
    '''
    Toma dos DRS y la simplifica, asegurando no acotar variables libres inadvertidamente.
    Input:
        - drs1, drs en formato nltk
        - drs2, drs en formato nltk
    Output:
        - drs en formato nltk
    '''
    free_nueva = set([str(v) for v in drs2.free()])
    refs_drs = set([str(v) for v in drs1.refs + list(drs1.free())])
    interseccion = [str(v) for v in refs_drs.intersection(free_nueva)]
    drs_nueva = drs2
    if len(interseccion) > 0:
#        print(f'¡Oops, unificación incorrecta con {interseccion}!', end='')
        for v in interseccion:
            contador = 1
            v_nueva = f'{v}{contador}'
            while v_nueva in refs_drs:
                contador += 1
                v_nueva = f'{v}{contador}'
#            print(f'\tReemplazando {v} => {v_nueva}')
            x1 = dexpr(fr'{str(v)}')
            x2 = dexpr(fr'{str(v_nueva)}')
            drs_nueva = drs_nueva.replace(x1.variable, x2, True)
    return (drs1 + drs_nueva).simplify()
