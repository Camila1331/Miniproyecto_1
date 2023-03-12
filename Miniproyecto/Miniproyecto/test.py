import resPreg
import re

text_data = {\
    "id":"texto_ej", \
    "texto": "Felipe tiene una niña. Él tiene un niño. El niño tiene un carrito. La niña tiene una muñeca. El niño rie.", \
    "preguntas": ["¿Quién tiene una niña?", "¿Qué tiene el niño?"], \
    "respuestas": [["Felipe", "felipe"], ["un carrito", "carrito", "un carro", "carro"]]\
}

#formula = 'exists e e018 x z15 z17.(MASC(x) & Felipe(x) & FEME(z15) & NIÑA(z15) & TENER(e) & SUJETO(e,x) & OBJ_DIR(e,z15) & MASC(x3) & MASC(z17) & NIÑO(z17) & TENER(e018) & SUJETO(e018,x3) & OBJ_DIR(e018,z17))'
#m = re.search(f'SUJETO([^)]+?,x3)', formula)
#print(m)

resPreg.procesamiento(text_data=text_data, 
                      gramatica_texto='texto1_drs.fcfg', 
                      gramatica_preguntas='preguntas_texto1.fcfg',
                      verbose=False)