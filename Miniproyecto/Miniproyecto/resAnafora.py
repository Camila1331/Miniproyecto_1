import re
import nltk
import utils
from parseMod import Modelo
from logUtils import existenciales_a_constantes
from nltk.sem.drt import DrtExpression

dexpr = DrtExpression.fromstring

def elimina_constantes_duplicadas(drs:nltk.sem.drt, verbose=False) -> nltk.sem.drt:
    '''
    Toma una DRS y trata de eliminar constantes duplicadas como
    una forma de resolución de la anáfora.
        Ejemplo: Un hombre camina. El hombre ríe.
    Input:
        - drs, una drs en formato nltk
        - verbose, booleano para imprimir detalles del proceso
    Output:
        - drs, una drs en formato nltk
        - resoluciones, un booleano para decir si hubo o no resoluciones
    '''
    from collections import defaultdict    
    from more_itertools import pairwise

    # Variable de bookkeeping sobre intercambios
    resoluciones = False
    # pasamos la drs a lógica de primer orden
    formula = drs.fol()
    if verbose:
        print(f'Esta es la fórmula correspondiente a la DRS:\n\n\t{formula}\n')
    formula, lista_eliminaciones = existenciales_a_constantes(formula)
    if verbose:
        print(f'Esta es la fórmula eliminando existenciales:\n\n\t{formula}\n')
        print('Obtenemos la lista de eliminaciones:\n')
    constantes = list(set([str(x[1]) for x in lista_eliminaciones]))
    dict_eliminaciones = defaultdict(list)
    igualdades = []
    drs_nueva = drs
    for constante in constantes:
        for var, const in lista_eliminaciones:
            if constante == str(const):
                if verbose:
                    print(f'\t{var} ===> {const}')
                dict_eliminaciones[str(const)].append(str(var))
        variables = list(set(dict_eliminaciones[constante]))
        # No debemos considear eventos
        variables = [v for v in variables if str(v)[0] != 'e']
        if len(variables) > 1:
            if verbose:
                print('\n\t\t' + '/'*50)
                print(f'\t\tDebemos unificar las siguientes variables:\n\t\t\t{variables}')
                print('\t\t' + '/'*50 + '\n')
            var_cambio = variables.pop()
            for variab in variables:
                drs_nueva = utils.cambio_y_sencillo(variab, var_cambio, drs_nueva)
            resoluciones = True
    return drs_nueva, resoluciones


def resuelve_variable_libre(drs:nltk.sem.drt, verbose:bool=False) -> nltk.sem.drt:
    '''
    Toma una DRS y trata de asignar la única variable libre a la única 
    constante como una forma de resolución de la anáfora.
        Ejemplo: Un hombre camina. Él ríe.
    Input:
        - drs, una drs en formato nltk
        - verbose, booleano para imprimir detalles del proceso
    Output:
        - drs, una drs en formato nltk
        - resoluciones, un booleano para decir si hubo o no resoluciones
    '''
    resoluciones = False
    if len(drs.free()) == 0:
        return drs, resoluciones
    # Buscamos las constantes
    formula = drs.fol()
    formula_, lista_eliminaciones = existenciales_a_constantes(formula)
    # Instaciamos la clase Modelo para llevar cuenta de 
    # constantes y predicados y poder fundamentadar las fórmulas
    M = Modelo()
    # Usamos la fórmula para poblar el modelo
    M.poblar_con(formula_)
    # Obtenemos los individuos
    candidatos = M.entidades['individuo']
    # Verificamos caso fácil: una variable y un individuo
    if (len(formula.free()) == 1) and (len(candidatos) == 1):
        variab = list(formula.free())[0]
        constante = candidatos[0]
        var_cambio = [x[0] for x in lista_eliminaciones if str(x[1]) == str(constante)][0]
        ig_drs = dexpr(fr'DRS([{variab}],[{var_cambio}={variab}])')
        drs = (drs + ig_drs)
        if verbose:
            print(f'\nResolvemos {variab} por {var_cambio}')
            drs.pretty_print()
        drs = drs.simplify().eliminate_equality()
        drs.refs = list(set(drs.refs))
        drs.conds = list(set(drs.conds))
        resoluciones = True
    return drs, resoluciones
    

def resuelve_libre_con_genero(drs, verbose=False):
    '''
    Toma una DRS y trata de asignar variables libres con 
    constantes con concordancia de género.
        Ejemplo: María tiene un perrito. Ella lo ama.
    Input:
        - drs, una drs en formato nltk
        - verbose, booleano para imprimir detalles del proceso
    Output:
        - drs, una drs en formato nltk
        - resoluciones, un booleano para decir si hubo o no resoluciones
    '''
    resoluciones = False
    drs_nueva = drs
    formula = str(drs_nueva.fol()) # la DRS como fórmula
    # Buscamos las variables femeninas
    vars_fem = [x for x in list(drs_nueva.free()) if f'FEME({x})' in formula]
    if len(vars_fem) == 1:
        # Buscamos las constantes femeninas
        referentes_fem = [x for x in drs_nueva.refs if f'FEME({x})' in formula]
        referentes_fem += vars_fem
        if verbose:
            refs = '\n\t'.join([str(x) for x in referentes_fem]) 
            print(f'\nEstos son los referentes femeninos:\n\n\t{refs}')
        if len(referentes_fem) == 2:
            variab = referentes_fem[0]
            var_cambio = referentes_fem[1]
            drs_nueva = utils.cambio_y_sencillo(variab, var_cambio, drs_nueva)
            resoluciones = True
    # Buscamos las variables masculinas
    vars_mas = [x for x in list(drs_nueva.free()) if f'MASC({x})' in formula]
    if len(vars_mas) == 1:
        # Buscamos las constantes masculinas
        referentes_mas = [x for x in drs_nueva.refs if f'MASC({x})' in formula]
        referentes_mas += vars_mas
        if verbose:
            refs = '\n\t'.join([str(x) for x in referentes_mas]) 
            print(f'\nEstos son los referentes masculinos:\n\n\t{refs}')
        if len(referentes_mas) == 2:
            variab = referentes_mas[0]
            var_cambio = referentes_mas[1]
            drs_nueva = utils.cambio_y_sencillo(variab, var_cambio, drs_nueva)
            resoluciones = True
    return drs_nueva, resoluciones


def resuelve_libre_con_sujeto(drs, verbose=False):
    '''
    Toma una DRS y trata de asignar variables libres con 
    constantes en posición sujeto.
        Ejemplo: María tiene un perrito. Ella está feliz.
    Input:
        - drs, una drs en formato nltk
        - verbose, booleano para imprimir detalles del proceso
    Output:
        - drs, una drs en formato nltk
        - resoluciones, un booleano para decir si hubo o no resoluciones
    '''
    resoluciones = False
    if not (0 < len(drs.free()) < 2):
        return drs, resoluciones
    # Buscamos sujetos
    referentes = drs.refs
    formula = str(drs.fol())
    sujetos = [x for x in referentes if re.search(f'SUJETO([^)]+?,{x})', formula)]
    if verbose:
        sujetos_ = '\n\t'.join([str(x) for x in sujetos]) 
        print(f'\nEstos son los sujetos (no pronombres):\n\n\t{sujetos_}\n')
    if len(sujetos) == 1:
        variab = sujetos[0]
        var_cambio = list(drs.free())[0]
        if verbose:
            print(f'Resolviendo {variab} => {var_cambio}')
        drs_nueva = utils.cambio_y_sencillo(variab, var_cambio, drs)
        resoluciones = True
        return drs_nueva, resoluciones
    return drs, resoluciones


def resuelve_anaforas(drs:nltk.sem.drt, verbose:bool=False) -> nltk.sem.drt:
    '''
    Toma una drs y trata de resolver las anáforas.
    Input:
        - drs, una drs en formato nltk
        - verbose, booleano para imprimir detalles del proceso
    Output:
        - drs, una drs en formato nltk con anáforas resueltas
    '''
    drs, info = elimina_constantes_duplicadas(drs, verbose=False)
    drs, info = resuelve_libre_con_genero(drs, verbose=False)
    drs, info = resuelve_libre_con_sujeto(drs, verbose=False)
    if len(drs.free()) == 0:
        return drs
    else:
        drs.pretty_print()
        raise Exception(f'\n¡Anáforas no resueltas, imposible continuar! {drs.free()}\n')