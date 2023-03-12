def numero(x):
    '''
    Toma un literal y devuelve su número
    entero correspondiente.
    '''
    try:
        if '-' in x:
            return - (ord(x[-1])-256)
        else:
            return ord(x)-256
    except Exception as e:
        if 'expected a character' in str(e):
            msg_error = f'¡Error! Algo quedó mal codificado con {x}'
            raise Exception(msg_error)

def literal(x):
    '''
    Toma un número y devuelve su
    literal  correspondiente.
    '''
    if x < 0:
        return '-' + chr(abs(x) + 256)
    else:
        return chr(x + 256)

def max_letras(a):
    '''
    Retorna el número máximo que corresponde
    a los átomos codificados en la fórmula a.
    Input:
        - a, fórmula como cadena en formato lp
    Output:
        - máximo más 1 de los caracteres
        - lista de números correspondientes a los caracteres
    '''
    a = a.replace('∧', ',')
    a = a.replace('∨', ',')
    a = a.replace('>', ',')
    a = a.replace('(', '')
    a = a.replace('-', '')
    a = a.replace(')', '')
    a = a.split(',')
    a = [numero(x) for x in a]
    a = list(set(a))
    a.sort()
    return max(a) + 1, a

def a_clausal(A):
    # Subrutina de Tseitin para encontrar la FNC de
    # la formula en la pila
    # Input: A (cadena) de la forma
    #                   p=-q
    #                   p=(q∧r)
    #                   p=(q∨r)
    #                   p=(q>r)
    # Output: B (cadena), equivalente en FNC
    assert(len(A)==4 or len(A)==7), f"Fórmula incorrecta! {A}"
    B = ''
    p = A[0]
    # print('p', p)
    if "-" in A:
        q = A[-1]
        # print('q', q)
        B = "-"+p+"∨-"+q+"∧"+p+"∨"+q
    elif "∧" in A:
        q = A[3]
        # print('q', q)
        r = A[5]
        # print('r', r)
        B = q+"∨-"+p+"∧"+r+"∨-"+p+"∧-"+q+"∨-"+r+"∨"+p
    elif "∨" in A:
        q = A[3]
        # print('q', q)
        r = A[5]
        # print('r', r)
        B = "-"+q+"∨"+p+"∧-"+r+"∨"+p+"∧"+q+"∨"+r+"∨-"+p
    elif "⇒" in A:
        q = A[3]
        # print('q', q)
        r = A[5]
        # print('r', r)
        B = q+"∨"+p+"∧-"+r+"∨"+p+"∧-"+q+"∨"+r+"∨-"+p
    elif "=" in A:
        q = A[3]
        # print('q', q)
        r = A[5]
        # print('r', r)
        #q∨-r∨-p∧-q∨r∨-p∧-q∨-r∨p∧q∨r∨p
        B = q+"∨"+"-"+r+"∨"+"-"+p+"∧"+"-"+q+"∨"+r+"∨"+"-"+p+"∧"+"-"+q+"∨"+"-"+r+"∨"+p+"∧"+q+"∨"+r+"∨"+p
    else:
        print(u'Error enENC(): Fórmula incorrecta!')
    B = B.split('∧')
    B = [c.split('∨') for c in B]
    return B

def tseitin(A):
    '''
    Algoritmo de transformacion de Tseitin
    Input: A (cadena) en notacion inorder
    Output: B (cadena), Tseitin
    '''
    # Creamos letras proposicionales nuevas
    m, l = max_letras(A)
    letrasp = [chr(x+256) for x in l]
    letrasp_tseitin = [chr(x) for x in range(m+256, m + 100000)]
    letrasp = letrasp + letrasp_tseitin
    L = [] # Inicializamos lista de conjunciones
    Pila = [] # Inicializamos pila
    i = -1 # Inicializamos contador de variables nuevas
    s = A[0] # Inicializamos símbolo de trabajo
    while len(A) > 0: # Recorremos la cadena
        # print("Pila:", Pila, " L:", L, " s:", s)
        if (s in letrasp) and (len(Pila) > 0) and (Pila[-1]=='-'):
            i += 1
            atomo = letrasp_tseitin[i]
            Pila = Pila[:-1]
            Pila.append(atomo)
            L.append(atomo + "=-" + s)
            A = A[1:]
            if len(A) > 0:
                s = A[0]
        elif s == ')':
            w = Pila[-1]
            O = Pila[-2]
            v = Pila[-3]
            Pila = Pila[:len(Pila)-4]
            i += 1
            atomo = letrasp_tseitin[i]
            L.append(atomo + "=(" + v + O + w + ")")
            s = atomo
        else:
            Pila.append(s)
            A = A[1:]
            if len(A) > 0:
                s = A[0]
    if i < 0:
        atomo = Pila[-1]
    else:
        atomo = letrasp_tseitin[i]
    B = [[[atomo]]] + [a_clausal(x) for x in L]
    B = [val for sublist in B for val in sublist]
    C = [[numero(x) for x in b] for b in B]
    return C