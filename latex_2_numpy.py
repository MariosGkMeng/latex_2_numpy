import os
import sys
import numpy as np
import re


D = dict()
D['pars'] = dict(
    {
            'brace-open': +1,
          'brace-closed': -1,
     'special-operators': dict({
                        'comma': '\🐯'
        })
    }
)

D['latex_functions'] = [
    ['\\frac{}{}',              '()/()'],
    [     '\\_{}',              '1']    
]

ID_EXPRESSION__FRAC         = 0
ID_EXPRESSION__POWER        = 1
ID_EXPRESSION__UNDERSCORE   = 2
ID_EXPRESSION__ARRAY        = 3

# 🕥 (When done --> delete the section that uses the same code) Sec. -1: Try with OOP ============================================================
def call_counter(func):
    def helper(*args, **kwargs):
        helper.calls += 1
        # print(helper.calls)
        return func(*args, **kwargs)
    helper.calls = 0
    return helper


class expression:
    
    @call_counter
    def __init__(self, s, parent=None):
        self.s = s
        self.parent = parent
        if not hasattr(self, 'token'): self.token = self.get_token()

    def get_token(self):
        # 💬 this function is seperate from __init__, so that the call of the latter is registered in the decorator first
        return self.__init__.calls
        
    def status(self):
        print('Status of expression is:' + '🍻🍻🍻🍻🍻')
        
    def get_in_brace_map(self):
        s = self.s
        indices_open   = [index for index in range(len(s)) if s.startswith('{', index)]
        indices_closed = [index for index in range(len(s)) if s.startswith('}', index)]

        dum1 = [D['pars']['brace-open']   for x in indices_open]
        dum2 = [D['pars']['brace-closed'] for x in indices_closed]


        Lx1 = len(indices_open)
        Lx2 = len(indices_closed)
        Lx = Lx1+Lx2

        brace_map = np.linalg.init_matrix([3, Lx])

        brace_map[0][:] = dum1 + dum2
        brace_map[1][:] = indices_open + indices_closed


        i_sort = np.argsort(brace_map[1][:])

        brace_map[0][:] = brace_map[0][i_sort]
        brace_map[1][:] = brace_map[1][i_sort]



        d = np.linalg.init_matrix([Lx])
        d0 = 0

        for i in range(len(d)): 
            d[i] = brace_map[0][i] + d0
            d0 = d[i]
            
        brace_map[2][:] = d


        # Get expressions

        d = np.linalg.init_matrix([1+Lx])
        d[1:] = brace_map[2][:]
        expressions = []
        position_expression = []
        df = np.diff(d)
        i = 0
        expression_number = []
        i_expr = 0
        i_expr_master = np.nan
        master_expressions = []
        while i < Lx:
            if df[i] == 1:
                if i == 0:
                    i_expr_master = i_expr
                    master_expressions.append(i_expr_master)
                elif brace_map[2][i] == 1 and brace_map[2][i-1]==0:
                    i_expr_master = i_expr
                    master_expressions.append(i_expr_master)
                expression_number.append(i_expr)
                i_expr += 1
                for j in range(i+1,Lx):
                    if brace_map[2][j] == brace_map[2][i]-1: 
                        break
                    
                if (brace_map[2][i] > brace_map[2][i-1]) and brace_map[2][i] > 1:
                    # Nested expression
                    nested_in = i_expr_master
                    after_expr = ''
                    brace_level = brace_map[2][i]-1
                else:
                    nested_in = -1
                    brace_level = 0
                    if i > 0:
                        after_expr = master_expressions[-2]
                    else:
                        after_expr = ''
                # Append info for the expression
                # ➕ Can have it show relative relations (e.g. "nested in expression 1, immediately after expression 2")    
                expressions.append([
                    s[int(brace_map[1][i])+1:int(brace_map[1][j])],
                    brace_map[1][i],
                    nested_in,
                    after_expr,
                    brace_level
                    ])
                position_expression.append(brace_map[1][i])
            i += 1
        
        self.in_brace_map = expressions        
            
    def get_children(self):
        
        # Need to see inside the braces in order to obtain the children
        if not hasattr(self, 'in_brace_map'): self.get_in_brace_map()
        
        BRACE_MAP = self.in_brace_map
        CHILDREN = []
        for M in BRACE_MAP: CHILDREN.append(expression(M[0], self.token))
        
        self.CHILDREN = CHILDREN
        
    
        

# ========================================================================================================================


def latex_arg_parser(s):
    '''
    Gets the arguments inside the current latex expression
    '''

    return 0

def check_error_cases(E):
    '''
    Checks the errors based on E, which is a list that contains lists with two elements: 
        - 0: is a bool: If True, then we raise the Exception
        - 1: is a string: The Exception Message
    '''
    for err in E:
        if err[0]: raise Exception(err[1])

def expression__numpy(x, id = 0):
    
    if id == ID_EXPRESSION__FRAC:
        
        if len(x) != 2: raise Exception('Length of input must be two')
        
        h = ['(' + y + ')' for y in x]
        y = '/'.join(h)
        
    elif id == ID_EXPRESSION__POWER:
        
        if len(x) != 1: raise Exception('Length of input must be one')
        y = '**(' + x[0] + ')'
        
        
    elif id == ID_EXPRESSION__UNDERSCORE:
        
        if len(x) != 1: raise Exception('Length of input must be one')
        y = '_' + x[0].replace(',', '_')        
     
    elif id == ID_EXPRESSION__ARRAY:
        
        y = 1    
        
    else:
        
        raise Exception('Nothing Coded for this case!')
    
    return y

def expression__latex(x, id = 0):
    if id == ID_EXPRESSION__FRAC:
        
        if len(x) != 2: raise Exception('Length of input must be two')
        
        h = ['{' + y + '}' for y in x]
        y = '\\frac' + ''.join(h)
        
    elif id == ID_EXPRESSION__POWER:
        
        if len(x) != 1: raise Exception('Length of input must be one')
        y = '^{' + x[0] + '}'
        
    elif id == ID_EXPRESSION__UNDERSCORE:
        
        if len(x) != 1: raise Exception('Length of input must be one')
        y = '_{' + x[0] + '}'
        
    elif id == ID_EXPRESSION__ARRAY:
        
        y = '1'
        
    else:
        
        raise Exception('Nothing Coded for this case!')
    
    return y
    
    
def get_brace_map(s, type):

    '''
    ➕ --> write description
    
    '''
    s0 = type[0]
    s1 = type[1]

    if isinstance(s, str):
        indices_open   = [index for index in range(len(s)) if s.startswith(s0, index)]
        indices_closed = [index for index in range(len(s)) if s.startswith(s1, index)]
    elif isinstance(s, list):
        indices_open   = [index for index in range(len(s)) if s[index] == s0]
        indices_closed = [index for index in range(len(s)) if s[index] == s1]
        

    dum1 = [D['pars']['brace-open']   for x in indices_open]
    dum2 = [D['pars']['brace-closed'] for x in indices_closed]


    Lx1 = len(indices_open)
    Lx2 = len(indices_closed)
    Lx = Lx1+Lx2

    brace_map = np.linalg.init_matrix([3, Lx])

    brace_map[0][:] = dum1 + dum2
    brace_map[1][:] = indices_open + indices_closed


    i_sort = np.argsort(brace_map[1][:])

    brace_map[0][:] = brace_map[0][i_sort]
    brace_map[1][:] = brace_map[1][i_sort]
    
    d = np.linalg.init_matrix([Lx])
    d0 = 0

    for i in range(len(d)): 
        d[i] = brace_map[0][i] + d0
        d0 = d[i]
        
    brace_map[2][:] = d 
    
    return brace_map

    

def categorize_parenthesis_opening(x):
    
    '''
    ➕ --> write description
    
    '''    
    comma_operator = D['pars']['special-operators']['comma'].replace('\\', '')

    
    idx = [i+1 for i,p in enumerate(x[1:]) if p[1]=='(' and x[i-1][2]]
    x1 = [y[1] for y in x]    
    brace_map = get_brace_map(x1,'()')
    # idx0 = [i+1 for i,p in enumerate(x[1:]) if p[1]=='(']
    # idx1 = [i+1 for i,p in enumerate(x[1:]) if p[1]==')']
    Lx = len(brace_map[0])
    parenthesis_category = []
    for i in range(Lx):
        B = [q[i] for q in brace_map]
        if B[0] == 1:
            # parenthesis open
            # search rest until parenthesis closed
            criterion = False
            for k in range(i, Lx):
                
                # ⬇️ parenthesis has closed
                if brace_map[2][k] == B[2]-1: break
                idxC = brace_map[1][k]
                # ⬇️ Check criterion
                
                
            for j in range(int(brace_map[1][i]), int(brace_map[1][k])):
                criterion = comma_operator in x1[j]    
                if criterion: break
            
            parenthesis_category.append([i, criterion])              
    
    return parenthesis_category
    
# 🗃️ Namespace 
# 1. ⚠️/Weak-Coding: Whenever I write something that has weak coding inside
# 2. 🔗: Links to whatever recedes it
    
    
# LOGIC:
# 1. Express number of input args and the way that they are written like in the case of "\frac{}{}". The algorithm should then search for the arguments inside the braces
#       - Should be recursive (braces inside the braces)
# 2. More systematic: 
#   1. get all arguments inside braces, categorize them based on {position}, {preceding functions}
#   2. get all latex functions and combine with point 1 above
#       - Perhaps remove whitespace first
#       - Catalog position of function --> check for match with position of beginning of argument ("{")
#   3. Create "expression object" --> e.g. expr.frac with 
#       - attributes
#           - String expression
#           - Position start
#           - Position end
#           - input arguments
#           - Transformation function
#               - e.g. in "\frac" function:
#                   "\frac{arg[0]}{arg[1]}" ----> "(arg[0])/(arg[1])" 
#       - methods
#           - transform
#           - transform in expression (i.e. replace previous text with new text in the correct position)

# ========================
#         RULES
# ========================
# 1. The tool needs to be intuitive ---> do not make the user to need to remember that line separation happens with ';' instead of ',' for example
#       (as long as this is possible)

# ========================
# ========================

# 🔴 ERROR CASES
# 1. Does not convert power operation when the power is not inside braces (e.g. x^2 instead of x^{2})
#    💡1
# 6. Example 14: "dot{x}" is not converted properly
# 7. Example 14: Multiplication not triggered before "("
# 8. Example 15: Requires a function that recognizes iteration index and creates an array, or 2 scalar variables
# 9. Example 16: 
#       a. Converts A^{*} to A**(*) ---SOLUTION---> recognize power to "*" and name "A_star"
#       b. For some reason the multiplication between number and variable is not recognized!
#               (perhaps the local version in the home-pc hasn't been uploaded)

## Completed
#---- 2.✔️ Example  8: Confuses "interpolate" for a variable and adds "*" in between
#---- 3.✔️ Example  8: Even if "interpolate" was a variable, the multiplication is done wrong
#---- 4.✔️ Example 11: Need fix 5✔️
#---- 5.✔️ Example 12: Ruined by 'Sec. 3'
#---- 7.✔️ Example 14: "\beta" is converted erratically

#           - See ⚒️-2
# ⚒️ Fixes
# ⚠️1. Example-13 fixed using patch: "⚠️/Weak-Coding-2"
#   2. Replace '\beta' with '\\beta'
# ⚠️3. Issue ✔️5 fixed using patch: "⚠️/Weak-Coding-3"
#      - It is a patch because it is not a systematic solution
#           - with this we avoid multiplying with an expression that starts with '(' (error case 🔴7)
#      - Ideas on how to fix:
#           - We could identify which parentheses correspond to a function definition and then assign them the True/False values
#               - Examples:
#                   - P(expr1, expr2) ---transf---> same
#                       - can just get all expressions (parent and children) and then see if there are commas inside themf
#                   - P(expr1+expr2) ---transf---> P*(expr1+expr2) (⚠️ unless P is a function as well!)



#     i,                       Comment, Status, Sec.,     Expression to convert
EXAMPLES = [
   [ '1',                           '', '✔️', '',  'proc_{7} = k_{h} \cdot fx(XU_{S_{BH}}, K_{X}) \cdot (S_{O_{rate}} + ny_{h} \cdot S_{NO_{rate}_{eq}}) \cdot X_{BH}'],
   [ '2',                           '', '✔️', '',  '\\frac{2}{\sqrt{x^{2}+2x+1}}'],
   [ '3',                           '', '✔️', '',  '\\dot{P}= \\frac{1}{x^{2} + exp(x)} \\beta P - P \\frac{\phi^{exp(x^{2^{2}})} \phi^{T}}{m^{2}}P'],
   [ '4',     'Testing Array creation', '✔️', '',  'proc = \left[ \begin{array}{cc}  S_{S_{X_{BH}}} \cdot S_{O_{rate}} \\ S_{S_{X_{BH}}} \cdot S_{NO_{rate}} \cdot n_{y,g} \\ \mu_{A} \cdot fx(S_{NH}, K_{NH}) \cdot fx(S_{O}, K_{OA}) \cdot X_{BA} \\ b_{H} \cdot X_{BH} \\ b_{A} \cdot X_{BA} \\ proc_{7} \end{array} \right]'],
   [ '5',       'Test Matrix creation', '✔️', '',  'Y = \left[ {\begin{array}{cc}  0 & -\frac{1}{Y_{H}} & 0 & 0 & 1 & 0 & 0 & \frac{-(1-Y_{H})}{Y_{H}} & 0 & -i_{XB} & 0 & 0 & -\frac{i_{XB}}{14}\\ 0 & -\frac{1}{Y_{H}} & 0 & 0 & 1 & 0 & 0 & \frac{-(1-Y_{H})}{Y_{H}} & 0 & -i_{XB} & 0 & 0 & -\frac{i_{XB}}{14} \end{array} } \right]'],
   [ '6','Just some relevant equation', '✔️', '',  '\epsilon = \frac{{z-\theta^{T} \phi}}{1+\phi^{T}\phi}'],
   [ '7',          'Relevant equation', '✔️', '',  '\phi=\left[ {\begin{array}{cc}  {C_{f,3}} x_{\phi}+{D_{f,3}} y \\ {C_{f,2}}  x_{\phi}+D_{f,2} y \\ C_{f,1} x_{\phi}+{D_{f,1}} y \end{array} } \right]'],
   [ '8',          'Relevant equation', '✔️', '3', 'xu_{tot}=interpolate(X_{solv},y)'],
   [ '9',          'Relevant equation', '✔️', '',  'X_{solv}=g_{idx}'],
   ['10',     'Two equations in a row', '❌', '',  'A = \left[ {\begin{array}{cc}  -1 & 0 & 0 \\ 0 & -1  & 0\\ 0 & 1  & 0 \\ 0 & 0  & -1\end{array} } \right], \space \space \space b^{T}=\left[ {\begin{array}{cc}  10 & 0 & -1 & 0.1 \end{array} } \right]'],
   ['11',            'Pertaining ✔️4', '✔️', '',   'x \in \mathbb{R}^{n}'],
   ['12',                          '',  '✔️', '',  'b^{T}=\left[ {\begin{array}{cc}  10 & 0 & -1 & 0.1 \end{array} } \right]'],
   ['13',            'Pertaining ✔️4', '✔️', '',   'A \in \mathbb{R}^{n \times 2}'],
   ['14',                          '', '❌', '',   '\dot{P}=\beta P - P \frac{\phi \phi^{T}}{m^{2}}P + interpolate(X_{solv},y)'],
   ['15',       'Requires new section', '❌','',   'x_{k+1}=x_{k}+a_{k} \frac{dy(x_{k})}{dx}'],
   ['16',                           '', '❌', '',  'A^{*} =s^{4}+s^{3}+96.8s^{2}-314.78s+522']
]

I__TEST_EXAMPLE = 2

s = EXAMPLES[I__TEST_EXAMPLE-1][-1]

# ➕➕ 
# 0➕ Fix (1) X '⚠️/Weak-Coding'
# 1➕ Get the algorithm I used in order to count parenthesis open and close
# 2➕ Braces are still there after all operations
#       - Can just replace them with parentheses after the algorithm is done with all special functions
# 3✔️ Multiplications' identification
#       - Create condition that checks "continuity" between expressions ---> whenever continuity is not justified
#          (i.e. by another mathematical operation), it must be because of multiplication
# 4✔️ Translate 'x \in \mathbb{R}^{n}' to 'x = np.linalg.init_matrix([...])'
#     - 🔗 Example 11
# 5✔️ Adjust function definition so that there's no confusion with the "missing *" test
#     - 🔗 Examples 
#           - ✔️ 8 
#           - ✔️11
# 6➕ Multiple lines
#     - ⚠️ Keep RULE-1 in mind
#     - 💡 3   
# 7➕ What other errors can I identify that are similar to Error-5?

# 💡
# 1. Use a more OOP approach (Sec. -1:)
#   - An expression can be an object that has undergone sequential character-transformations. Those transformations can overlap each other.
#     We can have applications based on parent-child relationships between expressions (e.g. with nested expressions)
# 2. How can decorators help me?
#   - Whenever I complete a section of text conversion, the decorators can help me track which ones I have completed, so that the sections can more easily communicate with each other
# 3. Can I do a "ML" like approach? Like, give a few samples and say how the lines are separated

s = s.replace('\frac', '\\frac')
s = s.replace('\beta', '\\beta')


# Sec. 0: Map brace encounters and get in-brace expressions ================================================================

brace_map = get_brace_map(s, '{}')
Lx = len(brace_map[0])

# Get expressions using brace_map
d = np.linalg.init_matrix([1+Lx])
d[1:] = brace_map[2][:]
expressions = []
position_expression = []
df = np.diff(d)
i = 0
expression_number = []
i_expr = 0
i_expr_master = np.nan
master_expressions = []
while i < Lx:
    
    # ⬇️ If parenthesis/bracket/brace opens
    if df[i] == 1:
             
        if (i == 0) or (brace_map[2][i] == 1 and brace_map[2][i-1]==0):
             i_expr_master = i_expr
             master_expressions.append(i_expr_master)
        # else:
        #     raise Exception('Nothing Coded Here')

        expression_number.append(i_expr)
        i_expr += 1
        for j in range(i+1,Lx):
            
            brace_closes_here = brace_map[2][j] == brace_map[2][i] - 1
            
            if brace_closes_here: break
            
        if (brace_map[2][i] > brace_map[2][i-1]) and brace_map[2][i] > 1:
            # Nested expression
            nested_in = i_expr_master
            after_expr = ''
            brace_level = brace_map[2][i]-1
        else:
            nested_in = -1
            brace_level = 0
            if i > 0:
                after_expr = master_expressions[-2]
            else:
                after_expr = ''
        # Append info for the expression
        # ➕ Can have it show relative relations (e.g. "nested in expression 1, immediately after expression 2")    
        expressions.append([
            s[int(brace_map[1][i])+1:int(brace_map[1][j])],
            brace_map[1][i],
            nested_in,
            after_expr,
            brace_level
            ])
        position_expression.append(brace_map[1][i])
    i += 1
        
# ================================================================================================================================================================================================


# Initialize matrices based on the 'x \in \mathbb{R}^{n}' syntax ================================================================================================================================
signature = '\in \mathbb{R}^'

i_sig = s.find(signature)
if i_sig != -1:
    domain_str = s[i_sig + len(signature):]
    
    # ⚠️/Weak-Coding-2: We are allowing for any type of string inside the braces
    #   The developer might write something wrong and we are not preventing it
    domain = re.findall('\{[\w,\W]*\}', domain_str)[0].replace('{', '').replace('}', '')
    #
    
    
    t = ' \times '
    if t in domain:
        domain = domain.split(t)
    else:
        domain = ['1', domain]
    
    s = s[:i_sig]  + ' = np.linalg.init_matrix([' + ', '.join(domain) + '])'
    

# ================================================================================================================================




frac_expr = []
sff = [index for index in range(len(s)) if s.startswith('\\frac', index)]
sff = []

funcs       = ['\\frac',            '^',                    '_'                       ] #  ,                        '\begin{array}']
id__funcs   = [ID_EXPRESSION__FRAC, ID_EXPRESSION__POWER,   ID_EXPRESSION__UNDERSCORE] #,  ID_EXPRESSION__ARRAY]
for idxF, f in enumerate(funcs):
    for i in range(len(s)):
        if s.startswith(f, i):
            sff.append([id__funcs[idxF], i])
    
expr_len = []
for sf1 in sff:
    sf = sf1[1]
    
    if sf1[0] == ID_EXPRESSION__FRAC:
        N = 2
    elif sf1[0] == ID_EXPRESSION__POWER:
        N = 1
    elif sf1[0] == ID_EXPRESSION__UNDERSCORE:
        N = 1
    elif sf1[0] == ID_EXPRESSION__ARRAY:
        N = 3
    else:
        raise Exception('Nothing Coded here!')
        
    if sf != -1:
        curr = sf
        # search for receding expression
        # pp = [x-sf for x in position_expression if (x-sf)>0]
        for ix, x in enumerate(expressions):
            if x[1] - sf > 0: 
                break
        idx__receding_expression = ix
        
        brace_level = expressions[idx__receding_expression][4]
        
        
        expressions_frac = []
        n = 0
        for i in range(idx__receding_expression, len(expressions)):
            if expressions[i][4] == brace_level:
                # is at correct level
                n += 1
                expressions_frac.append(expressions[i][0])
            if n>=N: break

        new_expression      = expression__numpy(expressions_frac, id=sf1[0])
        previous_expression = expression__latex(expressions_frac, id=sf1[0])
        
        frac_expr.append([expressions_frac, new_expression, previous_expression])
        expr_len.append(len(previous_expression))


idx_sort_expressions = np.array(np.argsort(expr_len)[::-1])

frac_expr1 = []
for i in idx_sort_expressions: frac_expr1.append(frac_expr[i])    

frac_expr = frac_expr1
s1 = s
for ff in frac_expr:
    s1 = s1.replace(ff[2], ff[1])


# Straightforward replacements ================================================================
MAP_REPLACEMENTS = [
    ['\\beta',     'beta'],
    ['\\phi',      'phi'],
    ['\\theta',    'theta'],
    ['\theta',     'theta'],
    ['\\epsilon',  'epsilon'],
    ['\epsilon',   'epsilon'],
    ['\\mu',       'mu'],
    ['\\left',     ''],
    ['\\right',    ''],
    ['\left',      ''],
    ['\right',     ''],
    ['\\cdot',     '*']
]

for M in MAP_REPLACEMENTS: s1 = s1.replace(M[0], M[1])
# ================================================================================================================================


# Matrices ================================================================================================================================

im0 = [index for index in range(len(s1)) if s1.startswith('\begin{array}', index)]
im1 = [index for index in range(len(s1)) if s1.startswith('\end{array}', index)]

matrices = []

## Check error cases 
check_error_cases([
    [                 len(im0) != len(im1), 'You forgot to close or open the matrix declaration in your latex equation'],
    [np.any(np.array(im0) > np.array(im1)), 'the "begin{array}" and "end{array}" expression do not follow correct sequences']
])
## 

for i in range(len(im0)):
    str_replace = '\begin{array}'
    strI = s1[im0[i]:im1[i]].replace(str_replace, '')    
    i1 = strI.find('}')
    str_replace += strI[:i1+1]
    matrices.append([strI[i1+1:], '', str_replace])
    
## Convert Matrices
is_vector = False

for iM, M1 in enumerate(matrices):
    M = M1[0]
    rows = M.split('\\')
    mat_numpy = 'np.array('
    R = rows[0]
    cols = R.split('&')
    if len(cols) == 1: is_vector = True

    if is_vector:
        mat_numpy += '[' + ', '.join(rows) + ']'
    else:
        mat_numpy += '['
        colsR = []
        for R in rows:
            cols = R.split('&')

            # mat_numpy += '[' + ', '.join(cols) + ']'
            colsR.append('[' + ', '.join(cols) + ']')
        
        mat_numpy += ', '.join(colsR)
    mat_numpy += '])'
    matrices[iM][1] = mat_numpy


##

## Replace Matrices

for M in matrices:
    s1 = s1.replace(M[2] + M[0] + '\end{array}', M[1])

##

# ================================================================================================================================

# Sec. 3: Multiplications without the symbol ==================================================================================================

## Identify multiplications
# Ways to do this:
#  - Can separate all variables-expressions and see which ones do NOT have any operation between them
#       - ⚠ Cannot always accept the multiplication: e.g. if there's "12 12" somewhere, then it is wrong when writing math alltogether

### Examples
ex1_done = 'S_P S_H+ a1A+Bb**2 +B / (12 + A B)'
comma_operator = D['pars']['special-operators']['comma'] # because regex doesn't let me!
s1 = s1.replace(',', comma_operator)
patt1 = r"\b[a-zA-Z]\w*"
patt2 = '['+comma_operator+',a-z,A-Z,0-9,\-,\+,\*\*,\(,\),\{,\},/,\_,=,\.,\[,\]]*'
popo=re.findall(patt2,s1)
popo = [x for x in popo if len(x)>0]
#### The result is: ['S_P', 'S_H+', 'a1A+Bb**2', '+B', '/', '(12', '+', '52A)']
##### We have a multiplication whenever there's variables that don't have any operation between them
###### How to check this:
comma_operator = comma_operator.replace('\\', '')
operators = "+", "-", "**", "*", "/", "(", ")", "{", "}", " ", "==", "=", "[", "]", comma_operator
regex_pattern = '|'.join(map(re.escape, operators))

# ⚠️/Weak-Coding-3: we are not removing '(', because it might be a start to a function call. It's a patch for problem: ✔️5 

toRemove1 = ['{', '}'] 
toRemove = toRemove1 + [')'] # instead of: toRemove = ['(', ')', '{', '}']
#

listOperators = ['np.array', '**', '*', '/', '+', '-', '==', '=', ',', comma_operator, '(', "[", "]"]
has_variable = False
to_append_multiplication_after = []
pcT = []


for i, p in enumerate(popo):
    pc = re.split('('+regex_pattern+')', p)
    
    
    if has_variable and len(pc[0]) > 0:
        # has_variable = False
        to_append_multiplication_after.append(i-1)
        
    if len(pc[-1]) > 0:
        has_variable = True
    else:
        has_variable = False
        
    pc = [x for x in pc if len(x)>0 and x != ' ']
    pcT += pc




pc1 = [[i, x, (x in listOperators)] for i, x in enumerate(pcT) if not x in toRemove]
pc2 = [[i, x, (x in listOperators)] for i, x in enumerate(pcT) if not x in toRemove1]

# Categorize parenthesis opening to expression and function

parenthesis_category = categorize_parenthesis_opening(pc2)


#

pc1Conds = [x[2] for x in pc1]
for i_pc in range(len(pc1Conds[:-1])):
    if not (pc1Conds[i_pc] or pc1Conds[i_pc+1]):
        ii0 = pc1[i_pc][0]+1
        while pcT[ii0] in toRemove:
            ii0 += 1
        pcT[ii0] = '*' + pcT[ii0]
        
        
# for i in to_append_multiplication_after: popo[i] += '*'
s2 = ''.join(pcT).replace(comma_operator, ', ')
##

# ================================================================================================================================


# ⚠️/Weak-Coding: Remove remaining braces ================================================================================================================================
s2 = s2.replace('{', '(').replace('}', ')')
# ================================================================================================================================



print(s2)
