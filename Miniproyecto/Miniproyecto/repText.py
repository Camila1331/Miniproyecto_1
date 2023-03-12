import logUtils as LU
from parseMod import Modelo
import nltk
from nltk import load_parser
from nltk.sem.drt import DrtParser
import utils
import resAnafora as A

def test_parser(texto:str, parser:nltk.parse, verbose:bool=False) -> list:
    '''
    Toma un texto y prueba el parser. Devuelve una lista de errores.
    Input:
        - texto, cadena con oraciones
        - parser, un objeto nltk para parsing semántico
        - verbose, booleano para imprimir detalles del proceso
    Output:
        - errores, lista de tuplas de la forma (num_oracion, estatus, error)
    ''' 
    errores = []
    oraciones = utils.preprocess_text(texto)
    for i, oracion in enumerate(oraciones):
        if verbose:
            print(f'Procesando \"{oracion}\"')
        tokens = oracion.split(' ')
        formula = utils.obtener_formula(tokens, parser)
        if formula is None:
            errores.append((i, oracion, 'fallo parser'))
        else:
            if verbose:
                print(f'{oracion} => {formula}')
    if verbose:
        print(f'Porcentaje de aciertos: {(1 - len(errores)/len(oraciones))*100}% ({len(oraciones) - len(errores)}/{len(oraciones)})')
    return errores         


def procesa_texto(texto:str, parser:nltk.parse, verbose:bool=False) -> nltk.sem.logic:
    '''
    Procesa el texto del discurso en su representación lógica
    Input:
        - texto, cadena con oraciones
        - parser, un objeto nltk para parsing semántico
        - verbose, booleano para imprimir detalles del proceso
    Output:
        - formula_texto, una formula fundamentada en formato nltk
    '''
    # ---------------------
    # Procesando oraciones
    # ---------------------
    oraciones = utils.preprocess_text(texto)
    inicial = True
    for i, oracion in enumerate(oraciones):
        if verbose:
            print(f'\nProcesando \"{oracion}\"\t({i+1}/{len(oraciones)})\n')
        # ---------------------------
        # Encontrando representación
        # ---------------------------
        tokens = oracion.split(' ')
        drs_nueva = utils.obtener_formula(tokens, parser)
        if drs_nueva is None:
            print(f'¡Error de parser en oracion {oracion}!')
            return None
        if inicial:
            # ---------------------------------
            # Asignando representación inicial
            # ---------------------------------                
            drs = drs_nueva
            inicial = False
        else:
            # ---------------------------
            # Incorporamos en DRS previa
            # ---------------------------
            drs = utils.asegura_alfa_variante(drs, drs_nueva)
            # --------------------
            # Resolvemos anáforas
            # --------------------
            drs = A.resuelve_anaforas(drs, verbose=verbose)
        if verbose:
            print(f'\nRepresentación en DRS:')
            drs.pretty_print()
    # -------------------------------------
    # Transformamos a lógica de predicados
    # -------------------------------------
    formula_ = drs.fol()
    formula_, lista_eliminaciones = LU.existenciales_a_constantes(formula_)
    # -------------------------
    # Fundamentamos la fórmula
    # -------------------------
    M = Modelo()    # Instaciamos la clase Modelo
    M.poblar_con(formula_)     # Poblamos el modelo
    formula_texto = M.fundamentar(formula_)     # Fundamentamos
    if verbose:
        print(f'\nEsta es la fórmula fundamentada:\n\n\t{formula_texto}\n')      
    return formula_texto
                  
def procesa_preguntas(preguntas_:str, parser:nltk.parse, verbose:bool=False):
    '''
    Procesa una lista de preguntas sobre el texto en su representación lógica
    Input:
        - preguntas_, una cadena con las preguntas
        - parser, un objeto nltk para parsing semántico
        - verbose, booleano para imprimir detalles del proceso
    '''
    preguntas = utils.preprocess_text(preguntas_)
    formulas_preguntas = []
    if verbose:
        print('\nPreguntas:\n')
    for pregunta in preguntas:
        tokens = pregunta.split(' ')
        f1 = utils.obtener_formula(tokens, parser)
        if verbose:
            print(f'{pregunta}')
            print(f'{f1}')
            print(f'Variables a solucionar: {f1.term.free()}\n')
        formulas_preguntas.append(f1)
    return formulas_preguntas