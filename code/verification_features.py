"""Conservative complete arithmetic spans and numerical grounding features."""
import ast
from fractions import Fraction
import re
import numpy as np
from common import parse_answer
from reasoning_features import numeric
from number_features import tokenize


def arithmetic_steps(text,allow_rhs_units=False):
    s=text.replace(',','').replace('−','-').replace('–','-').replace('×','*').replace('÷','/')
    for a,b in ((r'\times','*'),(r'\cdot','*'),(r'\div','/'),(r'\left',''),(r'\right',''),('**','')):s=s.replace(a,b)
    for _ in range(4):
        s=re.sub(r'\\(?:d?frac)\{([\d.\s+*/()\-]+)\}\{([\d.\s+*/()\-]+)\}',r'((\1)/(\2))',s)
    s=re.sub(r'\\boxed\{([\d.\s+*/()\-]+)\}',r'(\1)',s)
    # A unit is a boundary, not empty text: "6 hours = 360 minutes" is
    # a conversion, not a false numeric equality 6 = 360.
    s=re.sub(r'\\(?:text|mathrm)\{[^{}]*\}', '@', s)
    s=re.sub(r'\\[,;!]', '', s)
    s=re.sub(r'(\d+(?:\.\d+)?)\s*\\?%',r'(\1/100)',s)
    s=s.replace('\n','|')
    steps=[]
    for m in re.finditer('=',s):
        left=re.search(r'([\d. +*/()\-]+)$',s[:m.start()])
        right=re.match(r'([\d. +*/()\-]+)',s[m.end():])
        if left is None or right is None:continue
        lhs=left.group(1).strip();rhs=right.group(1).strip()
        if not lhs or not rhs:continue
        prefix=s[:left.start()].rstrip()
        suffix=s[m.end()+right.end():].lstrip()
        # Never evaluate a numeric suffix of a symbolic expression as the whole LHS.
        if lhs[0] in '+-*/' and prefix and (prefix[-1].isalnum() or prefix[-1] in '})]@'):continue
        if suffix and (suffix[0].isalpha() or suffix[0] in '{'):
            # A whitespace-separated result unit does not invalidate a pure
            # arithmetic LHS. Keep rejecting adjacent symbols such as 3x.
            separated=bool(right.group(1) and right.group(1)[-1].isspace())
            if not (allow_rhs_units and suffix[0].isalpha() and separated):continue
        a,b=numeric(lhs),numeric(rhs)
        if a is None or b is None or not any(c in lhs+rhs for c in '+*/-'):continue
        if '/100)' in rhs and abs(a-100*b)<1e-5*max(1,abs(a)):continue
        # Direct adjacency to an identifier, e.g. x2+3, is also symbolic.
        if left.start()>0 and not left.group(1).startswith(' ') and s[left.start()-1].isalnum():continue
        try:
            tree=ast.parse(lhs,mode='eval')
            if not any(isinstance(t,ast.BinOp) for t in ast.walk(tree)):continue
            leaves=[float(t.value) for t in ast.walk(tree) if isinstance(t,ast.Constant) and type(t.value) in (float,int)]
        except SyntaxError:continue
        steps.append(dict(left=a,right=b,leaves=leaves,position=m.start()/max(1,len(s)),lhs=lhs,rhs=rhs))
    return steps


def verification_features(bank,allow_rhs_units=False):
    rows=[]
    for row in bank:
        q=row['question'];qnums,_=tokenize(q)
        for c in row['candidates']:
            text=c['text'];steps=arithmetic_steps(text,allow_rhs_units=allow_rhs_units);parsed=parse_answer(text)
            answer=float(Fraction(parsed)) if parsed is not None else 0.
            known=list(qnums)+[0,1,2,3,4,7,10,12,24,30,52,60,100,365,1000,.5,.25,.01]
            known += [v/100 for v in qnums]
            errors=[];grounded=[];late=[];final=[];novel=[];computed=[]
            for step in steps:
                a,b=step['left'],step['right'];err=abs(a-b)>max(.05,.005*max(abs(a),abs(b)))
                errors.append(err);late.append(err and step['position']>.65)
                final.append(err and abs(b-answer)<1e-5*max(1,abs(answer)))
                missing=[v for v in step['leaves'] if not any(abs(v-k)<1e-5*max(1,abs(v)) for k in known)]
                grounded.append(not missing);novel.extend(missing)
                known.append(b);computed.append(b)
            n=len(steps);last=text[-180:].lower();lastnum=re.findall(r'[-+]?\d+(?:\.\d+)?',text[-100:])
            rowf=[n,sum(errors),sum(errors)/max(1,n),sum(late),sum(final),sum(grounded),
                  sum(e and g for e,g in zip(errors,grounded)),len(novel),len(set(novel)),
                  errors[-1] if n else 0, any(errors), any(final),
                  abs(computed[-1]-answer)/max(1,abs(answer)) if n else 0,
                  sum(abs(v-answer)<1e-5*max(1,abs(answer)) for v in computed),
                  text.rstrip().endswith('='),len(text)>3000,len(text)>3500,
                  bool(re.search(r'\\boxed\{[^{}]*[a-zA-Z][^{}]*\}',text)),
                  last.count('therefore'),last.count('answer'),last.count('####'),last.count('boxed'),
                  text.count('{')-text.count('}'),text.count('(')-text.count(')'),
                  len(qnums),len(set(qnums)),answer<0,abs(answer-int(answer))>1e-6,
                  abs(answer)>sum(abs(v) for v in qnums) if qnums else 0,
                  sum(abs(answer-v)<1e-6 for v in qnums),
                  bool(re.search(r'how many|number of',q.lower())) and abs(answer-int(answer))>1e-6,
                  bool(re.search(r'how many|how much|total|left|remaining',q.lower())) and answer<0]
            rows.append(rowf)
    x=np.array(rows,dtype='float32')
    return np.sign(x)*np.log1p(np.abs(x))


def repair_evidence(bank):
    """Support an available answer when it equals a recomputed final operation."""
    result=np.zeros((len(bank),4),dtype=np.float32)
    for i,row in enumerate(bank):
        parsed=[parse_answer(c['text']) for c in row['candidates']]
        answers=[float(Fraction(a)) if a is not None else np.nan for a in parsed]
        for source,answer in zip(row['candidates'],answers):
            if not np.isfinite(answer):continue
            steps=arithmetic_steps(source['text'])
            for step in steps:
                a,b=step['left'],step['right']
                if abs(b-answer)>1e-5*max(1,abs(answer)):continue
                if abs(a-b)<=max(.05,.005*max(abs(a),abs(b))):continue
                for j,target in enumerate(answers):
                    if np.isfinite(target) and abs(a-target)<1e-5*max(1,abs(target)):
                        result[i,j]=1.
    return result
