"""Question number extraction used by numerical grounding checks."""
from fractions import Fraction
import re

WORDS={'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,'eighty':80,'ninety':90,'hundred':100,'thousand':1000,'half':.5,'quarter':.25,'twice':2,'double':2,'doubled':2,'triple':3,'tripled':3,'thrice':3,'dozen':12}
def tokenize(question):
    q=question.lower().replace(',','')
    for tens in ('twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety'):
        for unit in ('one','two','three','four','five','six','seven','eight','nine'):
            q=re.sub(r'\b'+tens+r'[ -]'+unit+r'\b',str(WORDS[tens]+WORDS[unit]),q)
    pat=r'(?<![\w.])(?:\d+(?:\.\d+)?(?:/\d+)?|\.\d+)(?!\w)|\b(?:'+'|'.join(WORDS)+r')\b'
    numbers=[]
    def replace(m):
        token=m.group()
        try: value=float(WORDS[token]) if token in WORDS else float(Fraction(token))
        except (ValueError,ZeroDivisionError): return token
        numbers.append(value)
        return f' quantity{len(numbers)-1} '
    normalized=re.sub(pat,replace,q)
    return numbers,normalized

