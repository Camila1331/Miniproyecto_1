from parseMod import Modelo
from nltk import load_parser
from nltk.sem.drt import DrtParser
import logUtils as LU
import repText as T
import nltk
import numpy as np

lp = nltk.sem.logic.LogicParser()
                                 
def responde_pregunta(formula_texto:nltk.sem.logic, pregunta:nltk.sem.logic, verbose:bool=False):
    '''
    Toma una pregunta e intenta dar una respuesta
    Input:
        - formula_texto, una fórmula en nltk que se asume fundamentada.
        - pregunta_, una fórmula en nltk que se asume fundamentada y con un operador lambda.
        - verbose, booleano para imprimir detalles del proceso
    '''
    # ------------------------------------------
    # Verificamos representación de la pregunta 
    # ------------------------------------------
    tipo = LU.obtener_type(pregunta)
    assert(tipo == 'LambdaExpression'), f'Error: tipo no adecuado. Se esperaba LambdaExpression y se obtuvo {tipo}. La fórmula dada es {pregunta}'
    pregunta_ = pregunta.term
    variables = [v for v in list(pregunta_.free())]
    assert(len(variables) == 1), f'Error: se esperaba una sola variable libre y se obtuvieron {len(variables)}'
    # ------------------------------------------
    # Encontramos constantes a partir de texto
    # ------------------------------------------
    M = Modelo()
    M.poblar_con(formula_texto)
    # ------------------------------------------
    # Creamos la lista de candidatos
    # ------------------------------------------
    candidatos = M.entidades['individuo']
    formulas_candidatos = []
    for candidato in candidatos:
        # ------------------------------------------------
        # Reemplazamos el candidato por la variable libre
        # ------------------------------------------------
        reemplazo = fr'{pregunta.term}({candidato})'
        formula = LU.sust(pregunta.variable, pregunta.term, candidato)
        # -------------------------
        # Fundamentamos la fórmula
        # -------------------------
        pregunta_fundamentada = M.fundamentar(formula)
        formulas_candidatos.append(pregunta_fundamentada)
        if verbose:
            print(f'Candidato:\n\t{candidato} => {pregunta_fundamentada}\n')
    # -------------------------------------------
    # Verificamos implicación por cada candidato
    # Detenemos al primer acierto
    # -------------------------------------------
    respuesta = None
    for i, preg in enumerate(formulas_candidatos):
        res = M.ASK_dpll(preg, [formula_texto], 'success', verbose=verbose)
        if res:
            respuesta = str(candidatos[i])
            break
    return respuesta
            
    
def resuelve_preguntas(formula_texto:nltk.sem.logic, preguntas:list, verbose:bool=False) -> list:
    '''
    Intenta resolver una lista de preguntas sobre el texto
    Input:
        - formula_texto, una fórmula en nltk que se asume fundamentada.
        - preguntas, una lista de preguntas como fórmulas en nltk.
        - verbose, booleano para imprimir detalles del proceso
    '''
    # ------------------------------------
    # Intentamos responder cada pregunta
    # ------------------------------------
    intentos_respuestas = []
    for i, pregunta in enumerate(preguntas):
        if verbose:
            print(f'\nIntentando resolver la pregunta {i}\n')
        respuesta = responde_pregunta(\
                             formula_texto=formula_texto,\
                             pregunta=pregunta,\
                             verbose=verbose)
        intentos_respuestas.append(respuesta)
        if verbose:
            if respuesta is not None:
                print(f'La respuesta a la pregunta {i} es:\n\t{respuesta}\n')
            else:
                print(f'No se encontró respuesta a la pregunta. {preguntas[i]}')
    return intentos_respuestas


def procesamiento(text_data:dict, gramatica_texto:str, gramatica_preguntas:str, verbose:bool=True) -> tuple:
    '''
    Toma un texto, unas preguntas e intenta resolverlas. 
    Input:
        - text_data, un diccionario con las siguientes claves:
            * texto: una cadena con las oraciones
            * preguntas: una lista de cadenas con preguntas
            * respuestas: una lista de listas de cadenas con las respuestas posibles
        - gramatica_texto, cadena con el nombre de archivo de la gramática para 
                     hacer parsing del texto a DRT
        - gramatica_preguntas, cadena con el nombre de archivo de la gramática para 
                     hacer parsing de las preguntas al cálculo lambda
        - verbose, booleano para imprimir detalles del proceso
    Output:
        - respuestas, una lista con las respuestas a las preguntas
        - errores_parser, una lista con los errores del parser
    ''' 
    # ------------------------------------
    # Verificación de gramática del texto
    # ------------------------------------
    parser = load_parser(gramatica_texto, logic_parser=DrtParser(), trace=0)
    texto = text_data['texto']
    print(f'Verificando {gramatica_texto} como gramática para el texto...')
    errores_parser = T.test_parser(texto, parser, verbose=verbose)
    if len(errores_parser) > 0:
        print('='*20)
        print('Errores parser')
        print('='*20)
        print(errores_parser)
        print('='*20)
        return None
    print('Gramática sin errores.')
    print('Continuamos con el procesamiento del texto...')    
    # ------------------------------------
    # Procesamiento del texto
    # ------------------------------------
    formula_texto = T.procesa_texto(texto, parser, verbose=verbose)
    # ---------------------------------------
    # Verificación de gramática de preguntas
    # ---------------------------------------
    parser = load_parser(gramatica_preguntas, trace=0)
    preguntas = ' '.join(text_data['preguntas'])
    print(f'Verificando {gramatica_preguntas} como gramática para preguntas...')
    errores_parser = T.test_parser(preguntas, parser, verbose=verbose)
    if len(errores_parser) > 0:
        print('='*20)
        print('Errores parser')
        print('='*20)
        print(errores_parser)
        print('='*20)
        return None
    # ---------------------------------------
    # Representamos las preguntas
    # ---------------------------------------
    lista_preguntas = T.procesa_preguntas(preguntas, parser, verbose=verbose)
    # ---------------------------------------
    # Intentamos responder las preguntas
    # ---------------------------------------
    print('Intentamos resolver las preguntas...')
    respuestas = resuelve_preguntas(formula_texto=formula_texto,\
        preguntas=lista_preguntas,\
        verbose=verbose)
    soluciones = text_data['respuestas']
    aciertos = []
    for i, pregunta in enumerate(text_data['preguntas']):
        print(f'La respuesta a la pregunta {pregunta} es {respuestas[i]}')
        if respuestas[i] in soluciones[i]:
            print('\t=> Respuesta correcta')
            aciertos.append(1)
        else:
            print('\t=> Respuesta incorrecta')
            aciertos.append(0)
    print(f'Porcentaje de aciertos: {np.mean(aciertos)*100}')