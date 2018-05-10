# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:19:57 2018
# FUNCION TESTS COMPROBACION DE POINTERS Y DISTNACIAS
@author: jobqu
"""

import numpy as np     #For arrays

# Global Variables
#Memory Parameters
N = 10000               # Vectors' lenght
ones = int (N * 0.6931)  #.694 para N = 1,000   ,   .6931 para N = 10,000
Dict = {}
Symb_mat = []
labels = []

class HDvector (object):
    def __init__(self, vec_or_len, label=None, pointer = None): # Se inicializa con contenido (array) o con longitud (aleatorio)
        global Symb_mat
        if type(vec_or_len) is int:
            self.lenght = vec_or_len
            self.vec = SparseBitString(vec_or_len)
        elif type(vec_or_len) is np.ndarray:
            self.lenght = len(vec_or_len)
            self.vec = vec_or_len
        else:
            raise TypeError("Has to be int (length) or numpy array")        
        # Si se proporciona etiqueta se agrega el vector a catalogos globales
        if label in labels:
            raise NameError( "Label '" + str(label) + "' is already in catalog")
        elif label:
            labels.append(label)
            if len(Symb_mat) != 0:
                Symb_mat = np.concatenate((Symb_mat, self.vec.reshape((1,N))))     
            Dict[label] = self # Se agrega al diccionario
        self.label = label
        self.pointer = pointer
    def getVec(self):
        return self.vec
    def getLabel(self):  # Buscar en diccionario, mínima distancia es el bueno
        if self.label:
            return self.label
        else: #Hace busqueda en matriz
            HamVec = np.array([self.dist_vec(x) for x in Symb_mat])  # Potencialmente paralelizable...
            if HamVec.min() < N * 0.4: #tenia un .47
                return labels[HamVec.argmin()]
#            else:
#                #print('No hay etiqueta para este vector.')
#                return None #No hay ningun vector lo suficientemente parecido
    def getPointer(self):
        return self.pointer
    def setContent(self, in_array):
        self.vec = in_array
    def setPointer(self, other):
        self.pointer = other
    def dist(self, other):  #Mide distancia contra otro objeto
        assert self.lenght == other.lenght
        return np.count_nonzero(self.vec != other.vec) 
    def dist_vec(self, vecc): #Mide distancia contra otro vector
        assert self.lenght == len(vecc)
        return np.count_nonzero(self.vec != vecc)
    def p(self, times):
        return HDvector( np.roll(self.vec, times) )
    def ip(self, times):
        return HDvector( np.roll(self.vec, self.lenght - times) )
    def conc(self, other):
        return HDvector( np.concatenate((self.vec, other.vec)) )
    def __str__(self):
        return str(self.vec)
    def __add__(self, other):
        return ADD(self,other)
    def __mul__(self, other):
        return HDvector( np.bitwise_xor(self.vec, other.vec) )
    def __pow__(self, other): #Multiplica por apuntador de self
        if isinstance(self.getPointer(), HDvector):
            return HDvector(np.bitwise_xor(self.getPointer().getVec(), other.vec))
#        else:
#            return None
    def __div__(self, times):
        if isinstance(self.getPointer(), HDvector):
            return self.getPointer().ip(times)
#        else:
#            return None


" FUNCIONES GENERALES VECTORES "
def SparseBitString (n):
    """This function generate a random Binary Vector
    n: size of vector
    BitString: Binary Sparse Vector"""       
    # Generate 'ones' different random numbers in the range of 0 to n-1
    Address = np.random.randint(0, n-1, ones)    
    # Initialize Binary Vector
    BitString = np.zeros(n, dtype = np.int8)
    # Set to 1 generated addresses
    BitString[Address] = 1    
    return BitString

def ADD (*arg):
    """ Adición en general, asume que el argumento es objeto vector"""
    if len(arg) == 1 and type(arg[0]) is list:
        arg = arg[0]
    len_0 = arg[0].lenght
    Sum = np.zeros(len_0)
    n = len(arg)
    for vec in arg:
        assert vec.lenght == len_0 # Todas dimensiones son iguales
        Sum = Sum + vec.vec      # Suma normal
    Sum = Sum / n                # Promedio
    Sum[Sum > 0.5] = 1
    Sum[Sum == 0.5] =  SparseBitString(len_0)[Sum == 0.5]
    return HDvector(Sum.astype(np.int8))

"""*******************************************************************************************************"""
def init():
    global Symb_mat
    null = HDvector(N,'null')
    Symb_mat = np.array([null.getVec()])

def SaveListC (List):
    "Dada una lista de conceptos, los almacena en memoria"
    for c in List:
        HDvector(N,c)

def flat_list (L):
    if L == []:
        return L
    if type(L[0]) is list:
        return flat_list(L[0]) + flat_list(L[1:])
    return L[:1] + flat_list(L[1:])

def SaveConcepts (Dic):
    """A partir de un diccionario de definiciones extrae todas las etiquetas de conceptos, genera
    vectores aleatorios y los almacena en memoria."""
    keys = Dic.keys()
    vals = Dic.values()
    all_concepts = set(flat_list(vals) + keys)
    SaveListC (all_concepts)

        
def Definitions (loc_dict):
    def Def_permutations(List):
        "Recibe una lista con el concepto_ID, y las acepciones. Crea el apuntador correspondiente y lo asigna"
        Pointer = Dict[List[1]].p(1)
        if len(List) > 2:
            for i in range(2,len(List)):
                Pointer += Dict[List[i]].p(i)            
        Dict[List[0]].setPointer(Pointer)
    
    def Def_XOR (List):
        "Recibe una lista con el concepto y parejas (en forma de lista) atributo-valor. Crea y asigna apuntador"
        global Category
        vecs = []
        for par in List[1:]:
            vecs.append(Dict[par[0]] * Dict[par[1]])
            Category.append(par[0])
        Dict[List[0]].setPointer(ADD(vecs))
    
    global Category  #Recopila los atributos de propiedades 'is', 'has', ...
    
    for key, value in loc_dict.iteritems():
        if type(value[0]) is list:
            Def_XOR([key] + value)
        else:
            Def_permutations([key] + value)
    Category = set(Category)
    
# PONER ESTE DIC EN TESTS
Category = []
Dict_defs = {'coca'        : ['coca_1'], #, 'coca_2'],
             'soda'        : ['soda_1', 'soda_2'],
             'can'         : ['can_1', 'can_2'],
             'recipient'   : ['recipient_1'],
             'recipient_1' : [['is', 'object'], ['function','storage']] , 
             'can_1'       : [['is', 'recipient'], ['shape', 'cylinder_P'], ['size', 'medium_P']],
             'can_2'       : [['is', 'recipient'], ['shape', 'cylinder_P'], ['size', 'small_P']],
             'soda_1'      : [['is', 'can'], ['function', 'food']], 
             'soda_2'      : [['is', 'bottle_P'], ['size', 'medium_P']], #
             'coca_1'      : [['is', 'soda'], ['has', 'text_P'], ['color', 'red_P']] }   

#Dict_defs = {'A'    : ['A1','A2'],
#             'A2'   : [['is','p10_P']],
#             'A1'   : [['is','B'],['has','p1_P'],['size','p2_P'],['within','E']],
#             'B'    : ['B1','B2'],
#             'B1'   : [['has','p3_P'],['size', 'p4_P']],
#             'B2'   : [['has','p5_P'], ['is','C']],
#             'C'    : ['C1', 'C2','C3'],
#             'C1'   : [['is','p6_P'], ['size','p7_P']], #, ['has','D']],
#             'C2'   : [['is','F'],['shape','p8_P'],['inside-of','p9_P']],
#             'C3'   : [['has','p15_P']],
#             'E'    : ['E1','E2'],
#             'E1'   : [['is','p11_P'], ['has','p12_P']],
#             'E2'   : [['is','p13_P']]}

def Tests():
    """Pruebas iniciales para comprobar funcionamiento de la memoria."""
    init() # Inicializa vector 'null' y matriz de símbolos
    SaveConcepts(Dict_defs)  #Almacena conceptos individuales (auto-asociativo)
    Definitions(Dict_defs)   #Almacena definiciones (hetero-asociativo)

    assert Dict['coca_1'].dist( Dict['coca'].getPointer().ip(1)) < 0.4 * N
    assert Dict['soda'].dist( Dict['coca_1'].getPointer() * Dict['is'] ) < 0.4 * N
    assert Dict['text_P'].dist( Dict['coca_1'].getPointer() * Dict['has'] ) < 4000
    assert HDvector.getLabel(Dict['can_1'].getPointer() * Dict['size']) == 'medium_P'
    assert HDvector.getLabel(Dict['soda_1'] ** Dict['is']) == 'can'
    print "All tests passed!"


def Srch_perm (concepto):
    """Dado una etiqueta, realiza permutaciones inversas al Pointer de su vector
    hasta encontrar todas las etiquetas almacenadas.
    Ejemplo: HD-vector de 'coca' -> regresa, etiqueta 'coca_1', 'coca_2', ... """
    labels = []
    i = 1
    while True:
        lab = Dict[concepto] / i
        if isinstance(lab, HDvector):
            lab = HDvector.getLabel(Dict[concepto] / i)
            if lab:
                i += 1
                labels.append(lab)
            else:
                break
        else:
            break
    return labels

def Srch_pairs (concepto):
    labels = []
    for p in Category:
        lab = Dict[concepto] ** Dict[p]
        if isinstance(lab, HDvector):
            lab = HDvector.getLabel(Dict[concepto] ** Dict[p])
            if lab:
                labels.append(p + ':' + lab)
    return labels


def join_lists(L1, L2):
    def join_l(L1, L2):
        "Dado [1,2] y [[3,4], [5,6]]  -> [[1,2,3,4],[1,2,5,6]]" 
        if flat_list(L2) == []:
            return L1
        elif L1 and L2:
            return map(lambda x: L1 + x, L2)
        elif not L1 and L2:
            return L2
        else:
            return L1
    # PROCESAMIENTO PARA CASOS ESPECIALES...
    L22 = []
    for l in L2:
        if 'or' in l:
            L22.append([l])
        else:
            L22.append(l)
    return join_l(L1,L22)


def JoinPercept(P, op):
    """ De: ['text_P', 'medium_P', 'red_P'], '^' ->  'text_P ^ medium_P ^ red_P' """
    PP = [x + ' '+op+' ' for x in P[:-1]] + [P[-1]]
    return ''.join(PP)
    
def PutOp(L, op):
    """ De: ['text_P', 'medium_P', 'red_P'] -> ['text_P', op, 'medium_P', op, 'red_P'] """
    R = []
    for p in L:
        R.append(p)
        R.append(op)
    return R

def Search (concp):
    """Funcion para realizar la búsqueda. Dado una cadena corrrespondiente a la etiqueta de un concepto
    se realiza una búsqueda en la memoria hasta encontrar todos los conceptos asociados a percepciones"""
    if not concp: #Lista vacía
        return None
    else:
        index = concp.find(':') + 1
        if index > 0:
            concp = concp[index:]
        # Busca permutaciones
        concepts = Srch_perm(concp)
        if not concepts: #no tiene acepciones, es de pares
            concepts = Srch_pairs(concp)
            percept = [c for c in concepts if c[-2:] == '_P']
            new_concepts = list(set(concepts) - set(percept))
            # Cadena de perceptos
            return percept + map(Search, new_concepts)  #PutOp(percept, 'and')
        return PutOp(map(Search, concepts), 'or')


def clean_list (L):
    def clean_None (L):
        "Dada una lista L elimina [] sin importar el nivel de anidación en el que esté"
        if L == []:
            return None
        elif len(L) == 1:
            return clean_None(L[0]) if isinstance(L[0], list) else L[0]
        else:
            R = []
            for el in L:
                if not isinstance(el,list):
                    R.append(el)
                elif isinstance(el,list) and len(el) > 0:
                    R.append(clean_None(el))
            return R   
    def clean_or(L):
        if L == []:
            return L
        R = []
        L = L[:-1] if L[-1] == 'or' else L
        for l in L:
            if isinstance(l,str):
                R.append(l)
            else:
                R.append(clean_or(l))
        return R
    C = clean_or(L)
    while ','.join(map(str,C)).find('[]') > -1: # Busca lista vacia [] en texto
        C = clean_None(C)
    return C


def response (L):
    """Recibe lista con respuesta, la regresa con nuevo formato"""
    def list_or(In):
        if isinstance(In, list):
            return any(map(lambda x: 'or' in x, In))
        return True
    def check_basicForm (List):
        "Revisa forma básica: en el primer nivel sólo strings != 'or' y listas con 'or'"
        i = 0
        for el in List:
            if el == 'or' or not list_or(el):
                return False
            elif isinstance(el,str): # DEBE DE HABER STRINGS PARA QUE SEA FORMA BÁSICA...
                i += 1
        return i > 0
    def remove_ors (L):
        return [x for x in L if x != 'or']
    def prepare_list (L):
        "Quita 'or's y convierte strings solitas a lista"
        R = []
        for x in L:
            if x != 'or' and isinstance(x,str):
                R.append([x])
            elif isinstance(x,list):
                R.append(x)
        return R
    
    if isinstance(L,str):
        return [L]
    elif all(map(lambda x: isinstance(x,str), L)):#Lista de cadenas
        return L 
    elif check_basicForm(L):
        # 1) Extraer strings
        str_list = [s for s in L if isinstance(s, str)]
        # 2) Extrae listas
        list_list = [l for l in L if isinstance(l, list)]
        # 3) Combina
        R = []
        for l in list_list:
            R.extend( join_lists(str_list, prepare_list(l)) )
        return response(R)
    else:
        return map(response, remove_ors(L))

def final_clean (L):
    def lista_cadenas (l):
        return all(map(lambda x: isinstance(x,str),l))
    "Elimina niveles de anidación innecesarios..."
    if all(map(lista_cadenas, L)):
        return L
    else:
        R = []
        for el in L:
            if lista_cadenas(el):#Lista de cadenas
                R.append(el)
            else:
                R.extend(el)
        return final_clean(R)
    
#Tests()
init() # Inicializa vector 'null' y matriz de símbolos
SaveConcepts(Dict_defs)  #Almacena conceptos individuales (auto-asociativo)
Definitions(Dict_defs)   #Almacena definiciones (hetero-asociativo)


list_search = Search('can')
clean = clean_list(list_search) 
#print '\nLIMPIO: ', clean
resp = response(clean)
#print '\nRESPUESTA: ', resp
print '\nFINAL FINAL:',final_clean(resp)


# LIMPIAR Y OPTIMIZAR TODO EL CÓDIGO....


















