'''
Define la clase modelo, que guarda las constantes y predicados que se usarán 
para fundamentar y codificar las fórmulas para el uso de SAT-solvers.
'''
import nltk
import numpy as np
import logUtils as U
from logClases import *
from logProp import *
import re

class Modelo:
    '''
    Contendor del modelo de discurso.
    '''
    def __init__(self, formula=None):
        self.entidades = {}
        self.predicados = []
        self.vocabulario = []
        self.descriptor = None
        self.nltk_log_parser = nltk.sem.logic.LogicParser()
        if formula is not None:
            s = self.nltk_log_parser.parse(formula)
            self.poblar_con(s)

    def nueva_entidad(self, tipo:str, nombre:str):
        '''
        Crea una nueva entidad en la situación.
        '''
        try:
            n = len(self.entidades[tipo])
            nombres_previos = [c.nombre for c in self.entidades[tipo]]
            if nombre in nombres_previos:
                #print(f'¡Entidad ya existente! No se creó una nueva entidad. ({nombre})')
                pass
            else:
                self.entidades[tipo].append(Constante(tipo, nombre))
        except:
            n = 0
            self.entidades[tipo] = [Constante(tipo, nombre)]
        self.actualizar()

    def actualizar(self):
        '''
        Actualiza el vocabulario de la situación.
        '''
        tipos = list(self.entidades.keys())
        lista_aux = [self.entidades[l] for l in tipos]
        lista_aux = [item for sublist in lista_aux for item in sublist]
        self.vocabulario = [str(x) for x in lista_aux]
        self.vocabulario += [p.nombre for p in self.predicados] + ['<BLANK>']
        m = U.maxima_aridad(self.predicados) + 1
        lens = [len(self.vocabulario)]*m
        self.descriptor = Descriptor(lens)

    def poblar_con(self, expresion:nltk.sem.logic):
        '''
        Toma una fórmula y extrae los individuos allí representados. 
        '''
        constantes, predicados = U.vocabulario(expresion)
        for p in predicados:
            if not p.en_conjunto(self.predicados):
                self.predicados.append(p)
       # Creamos las constantes
        for c in constantes:
            self.nueva_entidad(tipo=c.tipo, nombre=c.nombre)

    def fundamentar(self, expresion:nltk.sem.logic) -> nltk.sem.logic:
        '''
        Toma una fórmula en lpo de nltk y cambia los cuantificadores
        existenciales por Otorias y los cuantificadores universales
        por Ytorias. En ambos casos se utilizan las entidades y 
        eventos de la situación.
        Input:
            - expresión, que es un objeto fórmula en lpo de nltk
        Output:
            - fórmula fundamentada, que es un objeto fórmula en lpo de nltk
        '''
        tipo = U.obtener_type(expresion)
#        print('\nFórmula es:\n\t', expresion)
#        print('\nSu tipo es', tipo)
        if tipo in ['ExistsExpression']:
            # La expresión es un cuantificador existencial 
            # de una fórmula phi.
            # Determinamos si la variable del cuantificador es
            # o bien una entidad o bien un evento. 
            phi = expresion.term
            var = expresion.variable.name
            tipo_var = 'evento' if var[0] == 'e' else 'entidad'
            if tipo_var == 'evento':
                consts = [str(c) for c in self.entidades['evento']]
#                print('\nOtoria sobre los eventos', consts)
                otoria = [self.nltk_log_parser.parse(rf'\{var}.({phi})({c})').simplify() for c in consts]
            else:
                consts = [str(c) for c in self.entidades['individuo']]
#                print('\nOtoria sobre las entidades', consts)
                otoria = [self.nltk_log_parser.parse(rf'\{var}.({phi})({c})').simplify() for c in consts]
            return U.Otoria([self.fundamentar(f) for f in otoria])
        elif tipo in ['AllExpression']:
            # La expresión es un cuantificador universal
            # de una fórmula phi.
            # Determinamos si la variable del cuantificador es
            # o bien una entidad o bien un evento. 
            phi = expresion.term
            var = expresion.variable.name
            tipo_var = 'evento' if var[0] == 'e' else 'entidad'
            if tipo_var == 'evento':
                consts = [str(c) for c in self.entidades['evento']]
#                print('\nYtoria sobre los eventos', consts)
                ytoria = [self.nltk_log_parser.parse(rf'\{var}.({phi})({c})').simplify() for c in consts]
            else:
                consts = [str(c) for c in self.entidades['individuo']]
#                print('\nYtoria sobre las entidades', consts)
                ytoria = [self.nltk_log_parser.parse(rf'\{var}.({phi})({c})').simplify() for c in consts]
            return U.Ytoria([self.fundamentar(f) for f in ytoria])
        elif tipo in ['NegatedExpression']:
            phi = expresion.term
            return nltk.sem.logic.NegatedExpression(phi)
        elif tipo in ['AndExpression']:
            first = self.fundamentar(expresion.first)
            second = self.fundamentar(expresion.second)
            return nltk.sem.logic.AndExpression(first, second)
        elif tipo in ['OrExpression']:
            first = self.fundamentar(expresion.first)
            second = self.fundamentar(expresion.second)
            return nltk.sem.logic.OrExpression(first, second)
        elif tipo in ['ImpExpression']:
            first = self.fundamentar(expresion.first)
            second = self.fundamentar(expresion.second)
            return nltk.sem.logic.ImpExpression(first, second)
        elif tipo in ['ApplicationExpression']:
            argumentos = expresion.args
            for x in argumentos:
                tipo_argumento = U.obtener_type(x)
                assert('Constant' in tipo_argumento), f'¡Error: Átomo no fundamentado! {tipo_argumento} en {expresion}'
            return expresion
        elif tipo in ['EqualityExpression']:
            assert(len(expresion.variables()) == 0), f'¡Error: Átomo no fundamentado! {expresion.variables()} en {expresion}'
            return expresion
        else:
            raise Exception(f'¡Tipo de expresión desconocido! {tipo}')

    def codificar_(self, pred:nltk.sem.logic.ApplicationExpression) -> str:
        '''
        Toma un predicado y devuelve su codificación
        Input:
            - pred, que es un ApplicationExpression de nltk
        Output:
            - codigo, que es un string
        '''
        tipo = U.obtener_type(pred)
        assert(tipo in ['ApplicationExpression', 'EqualityExpression'])
        if tipo == 'EqualityExpression':
            predicado = [self.vocabulario.index('IGUALDAD')]
            argumentos = [self.vocabulario.index(str(x)) for x in pred.constants()]
        else:
            predicado = [self.vocabulario.index(str(pred.pred))]
            argumentos = [self.vocabulario.index(str(x)) for x in pred.args]
        lista_valores = predicado + argumentos
        n = len(lista_valores)
        m = len(self.descriptor.args_lista)
        if n < m:
            lista_valores += [self.vocabulario.index('<BLANK>')] * (m - n)
        return self.descriptor.codifica(lista_valores=lista_valores)

    def decodificar_(self, x:int) -> str:
        '''
        Toma un entero y lo decodifica en una fórmula atómica fundamentada.
        Input:
            - x, que es un entero dentro del rango del descriptor.
        Output:
            - Una cadena con un predicado y sus argumentos.
        '''
        lista_valores = self.descriptor.decodifica(literal(abs(x)))
        predicado = self.vocabulario[lista_valores[0]]
        argumentos = [self.vocabulario[i] for i in lista_valores[1:]]     
        formula = f'{predicado}({",".join(argumentos)})'
        formula = re.sub(r"<BLANK>", r" ", formula)
        formula = re.sub(r" ,", r"", formula)
        formula = re.sub(r", ", r"", formula)
        if x<0:
            formula = '-' + formula
        return formula            
    
    def codificar_lp(self, expresion:nltk.sem.logic) -> str:
        '''
        Toma una fórmula y devuelve su versión codificada 
        en lógica proposicional.
        Input:
            - expresión, que es un objeto fórmula en lpo de nltk
        Output:
            - codigo, que es un string en codificación lp
        '''
        tipo = U.obtener_type(expresion)
        if tipo in ['ExistsExpression', 'AllExpression']:
            raise Exception(f'¡Expresión no está fundamentada!')
        elif tipo in ['AndExpression']:
            first = self.codificar_lp(expresion.first)
            second = self.codificar_lp(expresion.second)
            return f'({first}∧{second})'
        elif tipo in ['OrExpression']:
            first = self.codificar_lp(expresion.first)
            second = self.codificar_lp(expresion.second)
            return f'({first}∨{second})'
        elif tipo in ['ImpExpression']:
            first = self.codificar_lp(expresion.first)
            second = self.codificar_lp(expresion.second)
            return f'({first}>{second})'
        elif tipo in ['ApplicationExpression', 'EqualityExpression']:
            return self.codificar_(expresion)
        else:
            raise Exception(f'¡Tipo de expresión desconocido! {tipo}')

    def __str__(self):
        cadena = '\n' + '='*20 + 'COMPONENTES DEL MODELO' + '='*20
        cadena += '\n\nEntidades:\n'
        for tipo in self.entidades:
            cadena += f'\n\tTipo: {tipo}\n'
            for o in self.entidades[tipo]:
                cadena += f'\t\tNombre={o}\n'
        cadena += '\nPredicados:\n\n'
        for p in self.predicados:
            cadena += f'\tNombre={p.nombre}\n'
        cadena += '\n'
        cadena += '='*24 + 'FIN DEL MODELO' + '='*24 + '\n'
        return cadena

    def ASK_dpll(self, objetivo_:str, premisas_:list, valor:str, verbose=False) -> bool:
        '''
        Determina si el objetivo se sigue lógicamente de las premisas.
        Input:
            - objetivo_, fórmula fundamentada
            - premisas_, lista de fórmulas fundamentadas
            - valor, debe ser "success" o "failure"
            - verbose, booleano para imprimir info
        Output:
            - True o False dependiendo de si el objetivo se sigue lógicamente de las premisas y valor='success' 
        '''
        import pycosat
        # Codificamos en lógica proposicional las fórmulas fundamentadas
        objetivo = self.codificar_lp(objetivo_)
        premisas = [self.codificar_lp(prem) for prem in premisas_]
        # Creamos la fórmula F para verificar satisfacibilidad
        if len(premisas) == 0:
            F = objetivo
        elif len(premisas) == 1:
            F = f'({premisas[0]}∧-{objetivo})'
        elif len(premisas) > 1:
            premisas_ = U.Ytoria(premisas)
            F = f'({premisas_}∧-{objetivo})'
        # Hacemos la transformación de Tseitin sobre F
        m, letrasp = max_letras(F)
        A = tseitin(F)
        # Buscamos el modelo mediante pycosat
        res = pycosat.solve(A)
        if verbose:
            print('\n' + '-'*50)
            print('Buscando implicación lógica...')
            print('\nLas premisas son:\n')
            for p in premisas_:
                print('\t', p, end='\n\n')
            print('El objetivo es:\n\n\t', objetivo_)
            if res == 'UNSAT':
                print('\n¡El objetivo se sigue lógicamente de las premisas!')
            else:
                print('\n¡El objetivo NO se sigue lógicamente de las premisas')
#                print(f'\nUn modelo es:\n\n\t{[literal(x) for x in res]}')
                print(f'\nUn modelo es:\n\n\t{[self.decodificar_(x) for x in res if abs(x) in letrasp]}')
        return ((res == 'UNSAT') and valor == 'success')
        
class Descriptor:
    '''
    Codifica una lista de N argumentos mediante un solo caracter
    '''

    def __init__ (self, args_lista, chrInit=256) :
        '''
        Input:  
            - args_lista, lista con el total de opciones para cada
                        argumento del descriptor
            - chrInit, entero que determina el comienzo de la codificación chr()
        Output: str de longitud 1
        '''
        self.args_lista = args_lista
        assert(len(args_lista) > 0), "Debe haber por lo menos un argumento"
        self.chrInit = chrInit
        self.rango = [chrInit, chrInit + np.prod(self.args_lista)]

    def check_lista_valores(self,lista_valores) :
        assert(len(lista_valores) == len(self.args_lista)), "Lista de valores in completa o en exceso. Se recibieron {len(lista_valores)}, pero deben recibirse {\len(self.args_lista)}."
        for i, v in enumerate(lista_valores) :
            assert(v >= 0), "Valores deben ser no negativos"
            assert(v < self.args_lista[i]), f"Valor debe ser menor o igual a {self.args_lista[i]}"

    def lista_a_numero(self,lista_valores) :
        self.check_lista_valores(lista_valores)
        cod = lista_valores[0]
        n_columnas = 1
        for i in range(0, len(lista_valores) - 1) :
            n_columnas = n_columnas * self.args_lista[i]
            cod = n_columnas * lista_valores[i+1] + cod
        return cod

    def numero_a_lista(self,n) :
        decods = []
        if len(self.args_lista) > 1:
            for i in range(0, len(self.args_lista) - 1) :
                n_columnas = np.prod(self.args_lista[:-(i+1)])
                decods.insert(0, int(n / n_columnas))
                n = n % n_columnas
        decods.insert(0, n % self.args_lista[0])
        return decods

    def codifica(self,lista_valores) :
        codigo = self.lista_a_numero(lista_valores)
        return chr(self.chrInit+codigo)

    def decodifica(self,codigo) :
        n = ord(codigo)-self.chrInit
        return self.numero_a_lista(n)