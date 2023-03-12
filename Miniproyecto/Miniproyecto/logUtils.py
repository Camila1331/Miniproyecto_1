'''
Funciones auxiliares sobre fórmulas en lógica de primer orden (lpo).
'''
from logClases import *
import nltk

lp = nltk.sem.logic.LogicParser()

def unir_constantes(consts1:list, consts2:list) -> list:
    '''
    Toma dos listas de constantes y devuelve una sola
    sin repeticiones.
    Input:
        -consts1, lista de objetos constante
        -consts2, lista de objetos constante
    Output
        - lista de objetos constante
    '''
    unicos = [c for c in consts1]
    for c in consts2:
        if not c.en_conjunto(consts1):
            unicos.append(c)
    return unicos

def unir_predicados(preds1:list, preds2:list) -> list:
    '''
    Toma dos listas de predicados y devuelve una sola
    sin repeticiones.
    Input:
        -consts1, lista de objetos predicado
        -consts2, lista de objetos predicado
    Output
        - lista de objetos predicado
    '''
    unicos = [p for p in preds1]
    for p in preds2:
        if not p.en_conjunto(preds1):
            unicos.append(p)
    return unicos

def vocabulario(expresion:nltk.sem.logic) -> list:
    '''
    Toma una fórmula en lpo de nltk y devuelve sus
    constantes y predicados como objetos de parseSit.
    Input:
        - expresion, que es una fórmula en lpo de nltk
    Output:
        - constantes, que es un conjunto de 
                      constantes (como objetos de parseSit)
        - predicados, que es un conjunto de
                      predicados (como objetos de parseSit)
    '''
    tipo = obtener_type(expresion)
    if tipo in ['ExistsExpression', 'AllExpression', 'NegatedExpression', 'LambdaExpression']:
        return vocabulario(expresion.term)
    elif tipo in ['AndExpression', 'OrExpression', 'ImpExpression']:
        constantes1, predicados1 = vocabulario(expresion.first)
        constantes2, predicados2 = vocabulario(expresion.second)
        return  unir_constantes(constantes1, constantes2), unir_predicados(predicados1, predicados2)
    elif tipo in ['ApplicationExpression']:
        # Creamos el predicado
        ## Encontramos el nombre
        predicados_ = expresion.predicates()
        assert(len(predicados_) == 1)
        nomb = str(list(predicados_)[0])
        ## Encontramos los argumentos
        argumentos = expresion.args
        tipos_argumentos = [obtener_type(x) for x in argumentos]
        predicado = Predicado(nombre=nomb, tipos_argumentos=tipos_argumentos)
        predicados = [predicado] 
        # Creamos las constantes
        constantes = []
        for x in argumentos:
            tipo_x = obtener_type(x)
            if 'Constant' in tipo_x:
                if str(x)[0:3] == 'Ev_':
                    tipo_constante = 'evento'
                else:
                    tipo_constante = 'individuo'
                c = Constante(tipo=tipo_constante, nombre=str(x))
                if not c.en_conjunto(constantes):
                    constantes.append(c)
        return  constantes, predicados
    elif tipo in ['EqualityExpression']:
        constantes = []
        for x in expresion.constants():
            if str(x)[0:3] == 'Ev_':
                tipo_constante = 'evento'
            else:
                tipo_constante = 'individuo'
            c = Constante(tipo=tipo_constante, nombre=str(x))
            if not c.en_conjunto(constantes):
                constantes.append(c)    
        predicado = Predicado(nombre='IGUALDAD', tipos_argumentos=['any', 'any'])
        predicados = [predicado] 
        return constantes, predicados
    else:
        raise Exception(f'¡Tipo de expresión desconocido! {tipo}')
        
def obtener_type(objeto):
    '''Toma un objeto y retorna su tipo de manera legible
    Input:
        - objeto
    Output:
        - tipo, cadena con el tipo del objeto
    '''
    c = str(type(objeto))
    c = c.split('.')[-1][:-2]
    return c.split(r"'")[-1]

def Ytoria(lista_forms:list) -> nltk.sem.logic:
    '''
    Toma una lista de formulas y las une mediante &.
    Input:
        - lista_forms, que es una lista de fórmulas como objetos de nltk
    Output:
        - formula, que es un objeto de nltk
    '''
    if len(lista_forms) == 0:
        return None
    elif len(lista_forms) == 1:
        return lista_forms[0]
    else:
        form = lista_forms[0]
        for f in lista_forms[1:]:
            form = nltk.sem.logic.AndExpression(form, f)
        return form

def Otoria(lista_forms:list) -> nltk.sem.logic:
    '''
    Toma una lista de formulas y las une mediante |.
    Input:
        - lista_forms, que es una lista de fórmulas como objetos de nltk
    Output:
        - formula, que es un objeto de nltk
    '''
    if len(lista_forms) == 0:
        return None
    elif len(lista_forms) == 1:
        return lista_forms[0]
    else:
        form = lista_forms[0]
        for f in lista_forms[1:]:
            form = nltk.sem.logic.OrExpression(form, f)
        return form

def sust(var:nltk.sem.logic, exp1:nltk.sem.logic, exp2:nltk.sem.logic) -> nltk.sem.logic:
    '''
    Sustituye una expresión en otra.
    Input:
        - var, que es una variable en lpo de nltk
        - exp1, que es una fórmula en lpo de nltk
        - exp2, que es una fórmula en lpo de nltk
    Output:
        - expresion, en la cual exp2 se ha sustituido por las
                     ocurrencias libres de var en exp1
    '''
    aux = lp.parse(rf'\{var}.({exp1})({exp2})')
    return aux.simplify()

def encuentra_nombre(variable:str, predicados:list, formula) -> str:
    '''
    Toma una variable y una lista de predicados y devuelve el nombre de la entidad.
    Por ejemplo, devuelve "perro" si variable es "x" y "PERRO(x)" está en la fórmula.
    Input:
        - variable, cadena con el nombre de la variable
        - predicados, lista de objetos Predicado
        - formula, cadena con la fórmula
    Output:
        - nombre, cadena
    '''
    nombre = None
    for p in predicados:
        inicial = True
        if p.aridad == 1:
            if p.nombre not in ['MASC', 'FEME']:
                prueba = f'{p.nombre}({variable})'
                if prueba in formula:
                    if inicial:
                        nombre = p.nombre.lower()
                        inicial = False
                    else:
                        nombre = f'{nombre}-{p.nombre.lower()}'
    return nombre
    
def remover_existencial(expresion:nltk.sem.logic, constante:nltk.sem.logic) -> nltk.sem.logic:
    '''
    Toma una fórmula de tipo existe x phi(x), en la
    cual se sustituyen todas las ocurrencias libres de x 
    por una constante.
    Input:
        - expresion, que es una fórmula en lpo de nltk
        - constante, que es una constante en lpo de nltk
    Output:
        - formula, que es un objeto lpo de nltk donde la variable
                   del existencial fue reemplazada por la constate.
    '''
    tipo = obtener_type(expresion)
    assert(tipo == 'ExistsExpression'), f'¡La fórmula debe ser de tipo existencial!\nSe obtuvo {tipo}'
    var = expresion.variable
    funcion = expresion.term
    return sust(var=var, exp1=funcion, exp2=constante)

def existenciales_a_constantes(expresion:nltk.sem.logic) -> nltk.sem.logic:
    '''
    Toma una fórmula que tiene existenciales y los cambia por 
    una constante del tipo respectivo (individuo o evento).
    '''
    constantes, predicados = vocabulario(expresion)
    tipo = obtener_type(expresion)
    if tipo in ['ExistsExpression']:
        var = expresion.variable
        nombre_ = encuentra_nombre(variable=expresion.variable, predicados=predicados, formula=str(expresion.term))
        if nombre_ is not None:
            if var.name[0] == 'e':
                nombre = f'Ev_{nombre_}_{var.name[1:]}' 
            else:
                nombre = nombre_
            constante = lp.parse(nombre)
            formula = remover_existencial(expresion=expresion, constante=constante)
            f_nueva, lista = existenciales_a_constantes(formula)
            return f_nueva, lista + [(var, constante)]
    elif tipo in ['AllExpression']:
        var = expresion.variable
        formula, lista = existenciales_a_constantes(expresion.term)
        return lp.parse(rf'all {var}.{formula}'), lista
    elif tipo in ['AndExpression']:
        first, l1 = existenciales_a_constantes(expresion.first)
        second, l2 = existenciales_a_constantes(expresion.second)
        return nltk.sem.logic.AndExpression(first, second), l1 + l2
    elif tipo in ['OrExpression']:
        first, l1 = existenciales_a_constantes(expresion.first)
        second, l2 = existenciales_a_constantes(expresion.second)
        return nltk.sem.logic.OrExpression(first, second), l1 + l2
    elif tipo in ['ImpExpression']:
        first, l1 = existenciales_a_constantes(expresion.first)
        second, l2 = existenciales_a_constantes(expresion.second)
        return nltk.sem.logic.ImpExpression(first, second), l1 + l2
    elif tipo in ['ApplicationExpression', 'EqualityExpression']:
        return expresion, []
    else:
        raise Exception(f'¡Tipo de expresión desconocido! {tipo}')

def maxima_aridad(predicados:list) -> int:
    '''
    Toma una lista de predicados y devuelve la aridad máxima
    '''
    aridades = [p.aridad for p in predicados]
    return max(aridades)

