"""Conservative arithmetic and text features; no reference access or code execution."""
import ast
from fractions import Fraction
import operator
import re
import numpy as np
from common import parse_answer
from submission import features

OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}

def numeric(expr):
    expr=expr.strip().rstrip('.')
    if not expr or len(expr)>160:return None
    try:
        tree=ast.parse(expr,mode='eval')
        def visit(t,depth=0):
            if depth>12:raise ValueError()
            if isinstance(t,ast.Expression):return visit(t.body,depth+1)
            if isinstance(t,ast.Constant) and type(t.value) in (int,float):v=float(t.value)
            elif isinstance(t,ast.UnaryOp) and isinstance(t.op,(ast.USub,ast.UAdd)):v=(-1 if isinstance(t.op,ast.USub) else 1)*visit(t.operand,depth+1)
            elif isinstance(t,ast.BinOp) and type(t.op) in OPS:v=OPS[type(t.op)](visit(t.left,depth+1),visit(t.right,depth+1))
            else:raise ValueError()
            if not np.isfinite(v) or abs(v)>1e15:raise ValueError()
            return v
        return visit(tree)
    except (SyntaxError,ValueError,ZeroDivisionError,OverflowError,RecursionError):return None


def equations(text):
    s=text.replace(',','').replace('−','-').replace('×','*').replace('÷','/')
    s=s.replace('\\times','*').replace('\\cdot','*').replace('\\div','/').replace('\\left','').replace('\\right','')
    for _ in range(3):
        s=re.sub(r'\\(?:d?frac)\{([\d.\s+*/()\-]+)\}\{([\d.\s+*/()\-]+)\}',r'((\1)/(\2))',s)
    s=re.sub(r'(\d+(?:\.\d+)?)\s*\\?%',r'(\1/100)',s)
    # Only evaluate an entire contiguous numeric expression, never arbitrary text/code.
    pairs=[]
    for m in re.finditer('=',s):
        l=re.search(r'([\d.\s+*/()\-]+)$',s[:m.start()])
        r=re.match(r'([\d.\s+*/()\-]+)',s[m.end():])
        if l is None or r is None:continue
        left,right=l.group(1).strip(),r.group(1).strip()
        a,b=numeric(left),numeric(right)
        if a is not None and b is not None and any(op in left+right for op in '+*/-'):
            pairs.append((a,b))
    return pairs


def rich_features(bank):
    base,valid,votes=features(bank); rows=[]
    keywords=['but','instead','actually','mistake','incorrect','correct','assume','round','integer','cannot','not enough','double','twice','half','difference','each','per','left','more','less','discount','profit','total','remaining','altogether','how many','how much']
    for group in bank:
        q=group['question'].lower(); qnums=set(re.findall(r'\d+(?:\.\d+)?',q)); answers=[parse_answer(c['text']) for c in group['candidates']]
        for c,answer in zip(group['candidates'],answers):
            text=c['text']; low=text.lower(); eq=equations(text)
            errors=[abs(a-b)/max(1,abs(a),abs(b)) for a,b in eq]
            correct=sum(e<1e-6 for e in errors); wrong=len(eq)-correct
            val=float(Fraction(answer)) if answer is not None else 0.
            nums=re.findall(r'\d+(?:\.\d+)?',text)
            before=text.rsplit('####',1)[0] if '####' in text else text.rsplit('\\boxed',1)[0]
            numsbefore=[float(v) for v in re.findall(r'\d+(?:\.\d+)?',before)]
            row=[len(eq),correct,wrong,wrong/max(1,len(eq)),max(errors,default=0),sum(errors)/max(1,len(errors)),
                 len(nums),len(set(nums)-qnums),len(qnums & set(nums)),len(qnums),
                 sum(abs(v-val)<1e-6 for v in numsbefore),
                 abs(numsbefore[-1]-val)/max(1,abs(val)) if numsbefore else 0,
                 len(re.findall(r'\\boxed\{',text)),len(text.splitlines()),text.count('```'),text.count('→')]
            row += [low.count(k) for k in keywords]
            row += [q.count(k) for k in keywords]
            rows.append(row)
    extra=np.asarray(rows,dtype=np.float32)
    # Numeric outliers cannot dominate the neural feature scale.
    extra=np.sign(extra)*np.log1p(np.abs(extra))
    return np.column_stack([base,extra]).astype(np.float32),valid,votes
